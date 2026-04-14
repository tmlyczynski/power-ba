from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .capture import PulseAudioCapture
from .config import AppConfig
from .context import ConversationContext
from .llm import LlmClient, build_llm_client
from .stt import SttEngineError, build_stt_engine


@dataclass
class RuntimeState:
    paused: bool = False
    mic_enabled: bool = True
    ignore_remote_until: float = 0.0
    stop_requested: bool = False
    snapshot_requested: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def toggle_pause(self) -> bool:
        with self._lock:
            self.paused = not self.paused
            return self.paused

    def toggle_mic(self) -> bool:
        with self._lock:
            self.mic_enabled = not self.mic_enabled
            return self.mic_enabled

    def set_ignore_remote_for(self, seconds: int) -> float:
        with self._lock:
            self.ignore_remote_until = time.time() + max(1, seconds)
            return self.ignore_remote_until

    def should_ignore_remote(self, now: float | None = None) -> bool:
        current = now or time.time()
        with self._lock:
            return current < self.ignore_remote_until

    def request_stop(self) -> None:
        with self._lock:
            self.stop_requested = True

    def should_stop(self) -> bool:
        with self._lock:
            return self.stop_requested

    def is_paused(self) -> bool:
        with self._lock:
            return self.paused

    def is_mic_enabled(self) -> bool:
        with self._lock:
            return self.mic_enabled

    def request_snapshot(self) -> None:
        with self._lock:
            self.snapshot_requested = True

    def consume_snapshot_request(self) -> bool:
        with self._lock:
            if self.snapshot_requested:
                self.snapshot_requested = False
                return True
            return False


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._file = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def log(self, event_name: str, payload: dict) -> None:
        row = {
            "ts": time.time(),
            "event": event_name,
            **payload,
        }
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


def run_session(
    config: AppConfig,
    question_interval_override: int | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    max_runtime: int | None = None,
    interactive_controls: bool = True,
) -> None:
    config.sanitize()

    question_interval = question_interval_override or config.question_interval_seconds
    if question_interval < 5:
        question_interval = 5

    state = RuntimeState(mic_enabled=config.mic_listening_enabled)
    context = ConversationContext(window_seconds=max(config.context_window_seconds, question_interval))
    llm_client = build_llm_client(
        provider=config.provider,
        model=config.model,
        openai_api_key=config.openai_api_key,
        anthropic_api_key=config.anthropic_api_key,
    )

    logger: JsonlLogger | None = None
    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = output_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = JsonlLogger(out_dir / "session.jsonl")

    print("Consent reminder: use this app only when participants consent to recording/transcription.")
    if interactive_controls and sys.stdin.isatty():
        _start_controls_thread(state)
    else:
        print("Interactive controls disabled. Use CLI flags to control runtime.")

    start_time = time.time()
    next_question_time = start_time + question_interval

    try:
        if dry_run:
            _run_dry_session(
                state=state,
                context=context,
                llm_client=llm_client,
                logger=logger,
                main_prompt=config.main_prompt,
                question_interval=question_interval,
                output_dir=out_dir,
                start_time=start_time,
                max_runtime=max_runtime,
            )
        else:
            _run_live_session(
                config=config,
                state=state,
                context=context,
                llm_client=llm_client,
                logger=logger,
                main_prompt=config.main_prompt,
                question_interval=question_interval,
                output_dir=out_dir,
                start_time=start_time,
                next_question_time=next_question_time,
                max_runtime=max_runtime,
            )
    finally:
        if logger is not None:
            logger.close()

    print("Session stopped.")


def _run_live_session(
    config: AppConfig,
    state: RuntimeState,
    context: ConversationContext,
    llm_client: LlmClient,
    logger: JsonlLogger | None,
    main_prompt: str,
    question_interval: int,
    output_dir: Path | None,
    start_time: float,
    next_question_time: float,
    max_runtime: int | None,
) -> None:
    if not config.monitor_source:
        raise RuntimeError("Monitor source is not set. Configure it in settings first.")

    if config.mic_listening_enabled and not config.mic_source:
        print("Mic listening is enabled but mic source is empty. Continuing with remote-only capture.")

    capture = PulseAudioCapture(
        mic_source=config.mic_source,
        monitor_source=config.monitor_source,
        sample_rate=16000,
        channels=1,
        chunk_ms=100,
        is_source_enabled=lambda source: state.is_mic_enabled() if source == "mic" else True,
    )

    try:
        stt_engine = build_stt_engine(
            backend=config.stt_backend,
            vosk_model_path=config.vosk_model_path,
            sample_rate=16000,
        )
    except SttEngineError as exc:
        raise RuntimeError(str(exc)) from exc

    capture.start(output_dir=output_dir)
    print("Capture started. Controls: p=pause/resume, m=mic on/off, i [sec]=ignore remote, s=snapshot, q=stop")

    try:
        while not state.should_stop():
            now = time.time()
            if max_runtime is not None and now - start_time >= max_runtime:
                state.request_stop()
                continue

            if state.is_paused():
                if state.consume_snapshot_request():
                    _save_snapshot(context, output_dir, logger)
                time.sleep(0.1)
                continue

            chunk = capture.get_chunk(timeout=0.25)
            if chunk is not None:
                if chunk.source == "mic" and not state.is_mic_enabled():
                    pass
                elif chunk.source == "remote" and state.should_ignore_remote(now):
                    pass
                else:
                    events = stt_engine.accept_audio(
                        source=chunk.source,
                        audio_chunk=chunk.data,
                        timestamp=chunk.timestamp,
                    )
                    for event in events:
                        if not event.is_final:
                            continue
                        _handle_transcript(event.source, event.text, context, logger)

            now = time.time()
            if now >= next_question_time:
                _emit_questions(context, main_prompt, llm_client, logger)
                next_question_time = now + question_interval

            if state.consume_snapshot_request():
                _save_snapshot(context, output_dir, logger)

        for event in stt_engine.flush():
            if event.text.strip():
                _handle_transcript(event.source, event.text, context, logger)
    finally:
        capture.stop()


def _run_dry_session(
    state: RuntimeState,
    context: ConversationContext,
    llm_client: LlmClient,
    logger: JsonlLogger | None,
    main_prompt: str,
    question_interval: int,
    output_dir: Path | None,
    start_time: float,
    max_runtime: int | None,
) -> None:
    print("Dry-run mode enabled. No real audio capture is used.")
    print("Controls: p=pause/resume, m=mic on/off, i [sec]=ignore remote, s=snapshot, q=stop")

    scripted = [
        ("remote", "Zacznijmy od celow projektu i zakresu MVP."),
        ("mic", "Mamy ograniczony budzet i dwa sprinty na wdrozenie."),
        ("remote", "Najwieksze ryzyko to opoznione integracje API partnera."),
        ("mic", "Potrzebujemy tez metryk sukcesu i planu rolloutu."),
    ]

    index = 0
    next_phrase_time = start_time
    next_question_time = start_time + question_interval

    while not state.should_stop():
        now = time.time()

        if max_runtime is not None and now - start_time >= max_runtime:
            state.request_stop()
            continue

        if state.is_paused():
            if state.consume_snapshot_request():
                _save_snapshot(context, output_dir, logger)
            time.sleep(0.1)
            continue

        if now >= next_phrase_time:
            source, text = scripted[index % len(scripted)]
            index += 1
            next_phrase_time = now + 2.0

            if source == "mic" and not state.is_mic_enabled():
                pass
            elif source == "remote" and state.should_ignore_remote(now):
                pass
            else:
                _handle_transcript(source, text, context, logger)

        if now >= next_question_time:
            _emit_questions(context, main_prompt, llm_client, logger)
            next_question_time = now + question_interval

        if state.consume_snapshot_request():
            _save_snapshot(context, output_dir, logger)

        time.sleep(0.05)


def _handle_transcript(
    source: str,
    text: str,
    context: ConversationContext,
    logger: JsonlLogger | None,
) -> None:
    cleaned = text.strip()
    if not cleaned:
        return

    stamp = time.strftime("%H:%M:%S")
    label = "JA" if source == "mic" else "MEET"
    print(f"[{stamp}] [{label}] {cleaned}")

    context.add_line(source=source, text=cleaned)
    if logger is not None:
        logger.log("transcript", {"source": source, "text": cleaned})


def _emit_questions(
    context: ConversationContext,
    main_prompt: str,
    llm_client: LlmClient,
    logger: JsonlLogger | None,
) -> None:
    payload = context.render_for_prompt(main_prompt=main_prompt)
    if not payload:
        print("[AI] Waiting for more conversation context.")
        return

    answer = llm_client.generate_questions(payload)
    print("\n[AI QUESTIONS]\n" + answer + "\n")

    if logger is not None:
        logger.log("ai_questions", {"text": answer})


def _save_snapshot(
    context: ConversationContext,
    output_dir: Path | None,
    logger: JsonlLogger | None,
) -> None:
    if output_dir is None:
        print("Snapshot skipped: output directory not set.")
        return

    snapshot_path = output_dir / f"snapshot-{int(time.time())}.txt"
    rendered = context.render_for_prompt(main_prompt="Snapshot")
    snapshot_path.write_text(rendered or "(no data)", encoding="utf-8")
    print(f"Snapshot saved: {snapshot_path}")

    if logger is not None:
        logger.log("snapshot", {"path": str(snapshot_path)})


def _start_controls_thread(state: RuntimeState) -> threading.Thread:
    def _loop() -> None:
        while not state.should_stop():
            try:
                raw = input().strip()
            except EOFError:
                break

            if not raw:
                continue

            parts = raw.split()
            cmd = parts[0].lower()

            if cmd == "p":
                paused = state.toggle_pause()
                print("Paused." if paused else "Resumed.")
            elif cmd == "m":
                mic_enabled = state.toggle_mic()
                print("Mic listening ON." if mic_enabled else "Mic listening OFF.")
            elif cmd == "i":
                seconds = 30
                if len(parts) > 1:
                    try:
                        seconds = int(parts[1])
                    except ValueError:
                        seconds = 30
                until = state.set_ignore_remote_for(seconds)
                print(f"Ignoring remote audio until {time.strftime('%H:%M:%S', time.localtime(until))}.")
            elif cmd == "s":
                state.request_snapshot()
                print("Snapshot requested.")
            elif cmd == "q":
                state.request_stop()
                print("Stop requested.")
            else:
                print("Unknown command. Use: p, m, i [sec], s, q")

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread
