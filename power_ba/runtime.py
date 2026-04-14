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
from .diarization import BaseSpeakerDiarizer, build_speaker_diarizer
from .llm import LlmClient, build_llm_client
from .stt import SttEngineError, build_stt_engine


@dataclass
class RuntimeState:
    paused: bool = False
    mic_enabled: bool = True
    ignore_remote_until: float = 0.0
    stop_requested: bool = False
    snapshot_requested: bool = False
    force_generation_requested: bool = False
    ignored_speakers: set[str] = field(default_factory=set)
    known_speakers: set[str] = field(default_factory=set)
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

    def request_force_generation(self) -> None:
        with self._lock:
            self.force_generation_requested = True

    def consume_force_generation_request(self) -> bool:
        with self._lock:
            if self.force_generation_requested:
                self.force_generation_requested = False
                return True
            return False

    def register_speaker(self, speaker: str) -> None:
        cleaned = speaker.strip()
        if not cleaned:
            return
        with self._lock:
            self.known_speakers.add(cleaned)

    def toggle_ignore_speaker(self, speaker: str) -> bool:
        cleaned = speaker.strip()
        if not cleaned:
            return False

        with self._lock:
            if cleaned in self.ignored_speakers:
                self.ignored_speakers.remove(cleaned)
                return False

            self.ignored_speakers.add(cleaned)
            return True

    def is_speaker_ignored(self, speaker: str | None) -> bool:
        if not speaker:
            return False
        with self._lock:
            return speaker in self.ignored_speakers

    def list_known_speakers(self) -> list[str]:
        with self._lock:
            return sorted(self.known_speakers)

    def list_ignored_speakers(self) -> list[str]:
        with self._lock:
            return sorted(self.ignored_speakers)


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
    interval_enabled_override: bool | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    max_runtime: int | None = None,
    interactive_controls: bool = True,
) -> None:
    config.sanitize()

    question_interval_enabled = config.question_interval_enabled
    if interval_enabled_override is not None:
        question_interval_enabled = interval_enabled_override

    question_interval = config.question_interval_seconds
    if question_interval_override is not None:
        if question_interval_override <= 0:
            question_interval_enabled = False
        else:
            question_interval = question_interval_override

    if question_interval < 5:
        question_interval = 5

    state = RuntimeState(mic_enabled=config.mic_listening_enabled)
    required_window = question_interval if question_interval_enabled else 60
    context = ConversationContext(window_seconds=max(config.context_window_seconds, required_window))
    llm_client = build_llm_client(
        provider=config.provider,
        model=config.model,
        openai_api_key=config.openai_api_key,
        anthropic_api_key=config.anthropic_api_key,
    )
    diarizer = build_speaker_diarizer(
        enabled=config.diarization_enabled,
        backend=config.diarization_backend,
        hf_token=config.pyannote_hf_token,
        model_name=config.pyannote_model_name,
        sample_rate=16000,
        interval_seconds=config.diarization_interval_seconds,
        max_buffer_seconds=config.diarization_max_buffer_seconds,
    )

    logger: JsonlLogger | None = None
    out_dir: Path | None = None
    if output_dir is not None:
        out_dir = output_dir.expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        logger = JsonlLogger(out_dir / "session.jsonl")

    print("Consent reminder: use this app only when participants consent to recording/transcription.")
    if diarizer.status_message:
        print(f"Diarization: {diarizer.status_message}")
    if interactive_controls and sys.stdin.isatty():
        _start_controls_thread(state)
    else:
        print("Interactive controls disabled. Use CLI flags to control runtime.")

    start_time = time.time()
    next_question_time = start_time + question_interval if question_interval_enabled else None

    try:
        if dry_run:
            _run_dry_session(
                state=state,
                context=context,
                llm_client=llm_client,
                logger=logger,
                main_prompt=config.main_prompt,
                question_interval=question_interval,
                question_interval_enabled=question_interval_enabled,
                output_dir=out_dir,
                start_time=start_time,
                max_runtime=max_runtime,
            )
        else:
            _run_live_session(
                config=config,
                state=state,
                context=context,
                diarizer=diarizer,
                llm_client=llm_client,
                logger=logger,
                main_prompt=config.main_prompt,
                question_interval=question_interval,
                question_interval_enabled=question_interval_enabled,
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
    diarizer: BaseSpeakerDiarizer,
    llm_client: LlmClient,
    logger: JsonlLogger | None,
    main_prompt: str,
    question_interval: int,
    question_interval_enabled: bool,
    output_dir: Path | None,
    start_time: float,
    next_question_time: float | None,
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
            whisper_cpp_model_path=config.whisper_cpp_model_path,
            whisper_cpp_binary=config.whisper_cpp_binary,
            whisper_cpp_chunk_seconds=config.whisper_cpp_chunk_seconds,
            whisper_cpp_language=config.whisper_cpp_language,
        )
    except SttEngineError as exc:
        raise RuntimeError(str(exc)) from exc

    capture.start(output_dir=output_dir)
    print(
        "Capture started. Controls: p=pause/resume, m=mic on/off, i [sec]=ignore remote, "
        "x <speaker>=ignore speaker, k=list speakers, g=generate now, s=snapshot, q=stop"
    )
    if question_interval_enabled:
        print(f"Auto AI generation enabled every {question_interval}s.")
    else:
        print("Auto AI generation disabled. Use `g` to generate questions on demand.")

    try:
        while not state.should_stop():
            now = time.time()
            if max_runtime is not None and now - start_time >= max_runtime:
                state.request_stop()
                continue

            if state.is_paused():
                if state.consume_force_generation_request():
                    _emit_questions(context, main_prompt, llm_client, logger)
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
                    if chunk.source == "remote":
                        diarizer.add_audio(chunk.data, chunk.timestamp)
                        diarizer.run_if_due(now)
                        for speaker in diarizer.known_speakers():
                            state.register_speaker(speaker)

                    events = stt_engine.accept_audio(
                        source=chunk.source,
                        audio_chunk=chunk.data,
                        timestamp=chunk.timestamp,
                    )
                    for event in events:
                        if not event.is_final:
                            continue

                        speaker_label: str | None = event.speaker
                        if event.source == "remote":
                            speaker_label = speaker_label or diarizer.label_for_timestamp(event.timestamp)
                            if speaker_label:
                                state.register_speaker(speaker_label)
                            if state.is_speaker_ignored(speaker_label):
                                continue

                        _handle_transcript(
                            event.source,
                            event.text,
                            context,
                            logger,
                            speaker_label=speaker_label,
                        )

            if state.consume_force_generation_request():
                _emit_questions(context, main_prompt, llm_client, logger)

            now = time.time()
            if question_interval_enabled and next_question_time is not None and now >= next_question_time:
                _emit_questions(context, main_prompt, llm_client, logger)
                next_question_time = now + question_interval

            if state.consume_snapshot_request():
                _save_snapshot(context, output_dir, logger)

        for event in stt_engine.flush():
            if event.text.strip():
                speaker_label: str | None = event.speaker
                if event.source == "remote":
                    speaker_label = speaker_label or diarizer.label_for_timestamp(event.timestamp)
                    if speaker_label:
                        state.register_speaker(speaker_label)
                    if state.is_speaker_ignored(speaker_label):
                        continue

                _handle_transcript(
                    event.source,
                    event.text,
                    context,
                    logger,
                    speaker_label=speaker_label,
                )
    finally:
        capture.stop()


def _run_dry_session(
    state: RuntimeState,
    context: ConversationContext,
    llm_client: LlmClient,
    logger: JsonlLogger | None,
    main_prompt: str,
    question_interval: int,
    question_interval_enabled: bool,
    output_dir: Path | None,
    start_time: float,
    max_runtime: int | None,
) -> None:
    print("Dry-run mode enabled. No real audio capture is used.")
    print(
        "Controls: p=pause/resume, m=mic on/off, i [sec]=ignore remote, "
        "x <speaker>=ignore speaker, k=list speakers, g=generate now, s=snapshot, q=stop"
    )
    if question_interval_enabled:
        print(f"Auto AI generation enabled every {question_interval}s.")
    else:
        print("Auto AI generation disabled. Use `g` to generate questions on demand.")

    scripted = [
        ("remote", "SPEAKER_00", "Zacznijmy od celow projektu i zakresu MVP."),
        ("mic", None, "Mamy ograniczony budzet i dwa sprinty na wdrozenie."),
        ("remote", "SPEAKER_01", "Najwieksze ryzyko to opoznione integracje API partnera."),
        ("mic", None, "Potrzebujemy tez metryk sukcesu i planu rolloutu."),
    ]

    index = 0
    next_phrase_time = start_time
    next_question_time = start_time + question_interval if question_interval_enabled else None

    while not state.should_stop():
        now = time.time()

        if max_runtime is not None and now - start_time >= max_runtime:
            state.request_stop()
            continue

        if state.is_paused():
            if state.consume_force_generation_request():
                _emit_questions(context, main_prompt, llm_client, logger)
            if state.consume_snapshot_request():
                _save_snapshot(context, output_dir, logger)
            time.sleep(0.1)
            continue

        if now >= next_phrase_time:
            source, speaker_label, text = scripted[index % len(scripted)]
            index += 1
            next_phrase_time = now + 2.0

            if source == "mic" and not state.is_mic_enabled():
                pass
            elif source == "remote" and state.should_ignore_remote(now):
                pass
            elif source == "remote" and state.is_speaker_ignored(speaker_label):
                pass
            else:
                if speaker_label:
                    state.register_speaker(speaker_label)
                _handle_transcript(source, text, context, logger, speaker_label=speaker_label)

        if state.consume_force_generation_request():
            _emit_questions(context, main_prompt, llm_client, logger)

        if question_interval_enabled and next_question_time is not None and now >= next_question_time:
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
    speaker_label: str | None = None,
) -> None:
    cleaned = text.strip()
    if not cleaned:
        return

    stamp = time.strftime("%H:%M:%S")
    if source == "mic":
        label = "JA"
    elif speaker_label:
        label = f"MEET:{speaker_label}"
    else:
        label = "MEET"
    print(f"[{stamp}] [{label}] {cleaned}")

    context.add_line(source=source, text=cleaned, speaker=speaker_label)
    if logger is not None:
        logger.log(
            "transcript",
            {
                "source": source,
                "speaker": speaker_label,
                "text": cleaned,
            },
        )


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
            elif cmd == "x":
                if len(parts) < 2:
                    print("Usage: x <speaker_id>")
                    continue

                speaker_id = parts[1].strip()
                enabled = state.toggle_ignore_speaker(speaker_id)
                if enabled:
                    print(f"Speaker ignored: {speaker_id}")
                else:
                    print(f"Speaker unignored: {speaker_id}")
            elif cmd == "k":
                known = state.list_known_speakers()
                ignored = state.list_ignored_speakers()
                print("Known speakers: " + (", ".join(known) if known else "none"))
                print("Ignored speakers: " + (", ".join(ignored) if ignored else "none"))
            elif cmd == "g":
                state.request_force_generation()
                print("Manual AI generation requested.")
            elif cmd == "s":
                state.request_snapshot()
                print("Snapshot requested.")
            elif cmd == "q":
                state.request_stop()
                print("Stop requested.")
            else:
                print("Unknown command. Use: p, m, i [sec], x <speaker>, k, g, s, q")

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread
