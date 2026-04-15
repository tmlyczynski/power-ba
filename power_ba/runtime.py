from __future__ import annotations

import json
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .capture import (
    PulseAudioCapture,
    choose_default_mic_source,
    choose_default_monitor_source,
    list_mic_sources,
    list_monitor_sources,
)
from .config import AppConfig
from .context import ConversationContext
from .diarization import BaseSpeakerDiarizer, build_speaker_diarizer
from .llm import LlmClient, build_llm_client
from .stt import SttEngineError, build_stt_engine

CONTROL_REMINDER_SECONDS = 30.0
OutputEmitter = Callable[[str], None]


def _default_emit(message: str) -> None:
    print(message)


def _emit(emit: OutputEmitter, message: str) -> None:
    emit(message)


def _parse_ai_language(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized in {"en", "english", "angielski"}:
        return "en"
    if normalized in {"pl", "polski", "polish"}:
        return "pl"
    return None


def _parse_context_window_spec(value: str) -> int | None:
    cleaned = value.strip().lower().replace(" ", "")
    if not cleaned:
        raise ValueError("empty value")

    if cleaned in {"all", "full", "whole", "max", "unlimited", "nolimit", "no-limit"}:
        return None

    match = re.fullmatch(r"(\d+)([a-z]*)", cleaned)
    if not match:
        raise ValueError("invalid format")

    amount = int(match.group(1))
    if amount <= 0:
        raise ValueError("value must be > 0")

    unit = match.group(2)
    multipliers = {
        "": 1,
        "s": 1,
        "sec": 1,
        "secs": 1,
        "second": 1,
        "seconds": 1,
        "m": 60,
        "min": 60,
        "mins": 60,
        "minute": 60,
        "minutes": 60,
        "h": 3600,
        "hr": 3600,
        "hrs": 3600,
        "hour": 3600,
        "hours": 3600,
    }
    if unit not in multipliers:
        raise ValueError("unsupported unit")

    return amount * multipliers[unit]


def _format_context_window_spec(seconds: int | None) -> str:
    if seconds is None:
        return "full session context (no time limit)"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h ({seconds}s)"
    if seconds % 60 == 0:
        return f"{seconds // 60}m ({seconds}s)"
    return f"{seconds}s"


def _resolve_initial_context_window(
    config: AppConfig,
    required_window: int,
    emit: OutputEmitter,
) -> int | None:
    fallback_seconds = max(config.context_window_seconds, required_window)
    raw_default = config.ai_context_window_default.strip()
    if not raw_default:
        return fallback_seconds

    try:
        parsed = _parse_context_window_spec(raw_default)
    except ValueError:
        _emit(
            emit,
            f"Invalid default AI context window '{raw_default}'. Falling back to {fallback_seconds}s.",
        )
        return fallback_seconds

    if parsed is not None and parsed < 5:
        return 5

    return parsed


@dataclass
class RuntimeState:
    paused: bool = False
    mic_enabled: bool = True
    ignore_remote_until: float = 0.0
    stop_requested: bool = False
    snapshot_requested: bool = False
    force_generation_requested: bool = False
    custom_query_requests: list[str] = field(default_factory=list)
    ai_language: str = "pl"
    ai_style_instruction: str = ""
    ignored_speakers: set[str] = field(default_factory=set)
    known_speakers: set[str] = field(default_factory=set)
    recent_answer_requests: list[int | None] = field(default_factory=list)
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

    def request_custom_query(self, query: str) -> bool:
        cleaned = query.strip()
        if not cleaned:
            return False

        with self._lock:
            self.custom_query_requests.append(cleaned)
        return True

    def consume_custom_query_requests(self) -> list[str]:
        with self._lock:
            if not self.custom_query_requests:
                return []

            pending = list(self.custom_query_requests)
            self.custom_query_requests.clear()
            return pending

    def set_ai_language(self, language: str) -> str | None:
        cleaned = language.strip()
        if not cleaned:
            return None

        normalized = _parse_ai_language(cleaned)
        if normalized is None:
            return None

        with self._lock:
            self.ai_language = normalized
        return normalized

    def get_ai_language(self) -> str:
        with self._lock:
            return self.ai_language

    def set_ai_style_instruction(self, instruction: str) -> str:
        cleaned = instruction.strip()
        with self._lock:
            self.ai_style_instruction = cleaned
            return self.ai_style_instruction

    def get_ai_style_instruction(self) -> str:
        with self._lock:
            return self.ai_style_instruction

    def request_recent_answers(self, window_seconds: int | None) -> bool:
        with self._lock:
            self.recent_answer_requests.append(window_seconds)
        return True

    def consume_recent_answer_requests(self) -> list[int | None]:
        with self._lock:
            if not self.recent_answer_requests:
                return []

            pending = list(self.recent_answer_requests)
            self.recent_answer_requests.clear()
            return pending

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


def _print_controls_help(
    question_interval_enabled: bool,
    emit: OutputEmitter,
    reminder: bool = False,
) -> None:
    if reminder:
        _emit(emit, "\n=== Controls reminder ===")
    else:
        _emit(emit, "\n=== Session controls ===")

    _emit(emit, "  p          pause or resume processing (STT + AI)")
    _emit(emit, "  m          toggle microphone listening on/off")
    _emit(emit, "  i <sec>    ignore all remote audio for N seconds")
    _emit(emit, "  x <id>     ignore/unignore selected remote speaker id")
    _emit(emit, "  k          show known speakers and ignored speakers")
    _emit(emit, "  g          generate AI questions now (manual trigger)")
    _emit(emit, "  a <text>   send custom AI query / refine follow-up")
    _emit(emit, "  ra [<20s|2m|Xh|all>]  answer direct questions from recent context (default 2m)")
    _emit(emit, "  lang <pl|en> set AI response language")
    _emit(emit, "  style <text> set persistent AI style for this session")
    _emit(emit, "  ctx <20m|2h|1200|all> set AI context time window")
    _emit(emit, "  s          save text snapshot (requires output directory)")
    _emit(emit, "  h          show this help again")
    _emit(emit, "  q          stop current session")

    if question_interval_enabled:
        _emit(emit, "Auto AI generation is ON (periodic interval).")
    else:
        _emit(emit, "Auto AI generation is OFF. Use `g` to generate on demand.")


def run_session(
    config: AppConfig,
    question_interval_override: int | None = None,
    interval_enabled_override: bool | None = None,
    output_dir: Path | None = None,
    save_audio_override: bool | None = None,
    save_transcript_override: bool | None = None,
    dry_run: bool = False,
    max_runtime: int | None = None,
    interactive_controls: bool = True,
    show_controls_help: bool = True,
    command_queue: queue.Queue[str] | None = None,
    event_callback: OutputEmitter | None = None,
) -> None:
    config.sanitize()
    emit = event_callback or _default_emit

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

    state = RuntimeState(
        mic_enabled=config.mic_listening_enabled,
        ai_language=config.ai_language,
    )
    required_window = question_interval if question_interval_enabled else 60
    initial_context_window = _resolve_initial_context_window(
        config=config,
        required_window=required_window,
        emit=emit,
    )
    context = ConversationContext(window_seconds=initial_context_window)
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
    out_dir = _resolve_output_directory(config=config, output_dir_override=output_dir)
    save_audio_to_files = config.save_audio_by_default if save_audio_override is None else save_audio_override
    save_transcript_to_files = (
        config.save_transcript_by_default if save_transcript_override is None else save_transcript_override
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    if (save_audio_to_files or save_transcript_to_files) and out_dir is None:
        _emit(
            emit,
            "File saving enabled but output directory is not set. Configure output path in settings or use --output.",
        )

    if save_transcript_to_files and out_dir is not None:
        logger = JsonlLogger(out_dir / "session.jsonl")

    _emit(emit, "Consent reminder: use this app only when participants consent to recording/transcription.")
    if diarizer.status_message:
        _emit(emit, f"Diarization: {diarizer.status_message}")

    controls_enabled = False
    if interactive_controls:
        if command_queue is not None:
            controls_enabled = True
        elif sys.stdin.isatty():
            controls_enabled = True
            _start_controls_thread(
                state,
                context=context,
                question_interval_enabled=question_interval_enabled,
                emit=emit,
            )

    if not controls_enabled:
        _emit(emit, "Interactive controls disabled. Use CLI flags to control runtime.")

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
                controls_enabled=controls_enabled,
                show_controls_help=show_controls_help,
                output_dir=out_dir,
                start_time=start_time,
                max_runtime=max_runtime,
                command_queue=command_queue,
                emit=emit,
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
                controls_enabled=controls_enabled,
                show_controls_help=show_controls_help,
                output_dir=out_dir,
                save_audio_to_files=save_audio_to_files,
                start_time=start_time,
                next_question_time=next_question_time,
                max_runtime=max_runtime,
                command_queue=command_queue,
                emit=emit,
            )
    finally:
        if logger is not None:
            logger.close()

    _emit(emit, "Session stopped.")


def _resolve_output_directory(config: AppConfig, output_dir_override: Path | None) -> Path | None:
    if output_dir_override is not None:
        return output_dir_override.expanduser().resolve()

    configured = config.default_output_dir.strip()
    if not configured:
        return None

    return Path(configured).expanduser().resolve()


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
    controls_enabled: bool,
    show_controls_help: bool,
    output_dir: Path | None,
    save_audio_to_files: bool,
    start_time: float,
    next_question_time: float | None,
    max_runtime: int | None,
    command_queue: queue.Queue[str] | None,
    emit: OutputEmitter,
) -> None:
    allow_prompt = sys.stdin.isatty()
    monitor_source, mic_source = _resolve_live_audio_sources(config, allow_prompt=allow_prompt, emit=emit)
    if config.mic_listening_enabled and not mic_source:
        _emit(emit, "Mic listening enabled but no mic selected. Continuing with remote-only capture.")

    capture = PulseAudioCapture(
        mic_source=mic_source,
        monitor_source=monitor_source,
        sample_rate=16000,
        channels=1,
        chunk_ms=100,
        is_source_enabled=lambda source: state.is_mic_enabled() if source == "mic" else True,
    )

    stt_backend = _resolve_stt_backend_with_fallback(config, emit=emit)

    try:
        stt_engine = build_stt_engine(
            backend=stt_backend,
            vosk_model_path=config.vosk_model_path,
            sample_rate=16000,
            whisper_cpp_model_path=config.whisper_cpp_model_path,
            whisper_cpp_binary=config.whisper_cpp_binary,
            whisper_cpp_chunk_seconds=config.whisper_cpp_chunk_seconds,
            whisper_cpp_language=config.whisper_cpp_language,
        )
    except SttEngineError as exc:
        raise RuntimeError(str(exc)) from exc

    capture.start(output_dir=output_dir if save_audio_to_files else None)
    _emit(emit, "Capture started.")
    if controls_enabled and show_controls_help:
        _print_controls_help(question_interval_enabled, emit=emit)

    next_controls_reminder = (
        time.time() + CONTROL_REMINDER_SECONDS if controls_enabled and show_controls_help else None
    )

    try:
        while not state.should_stop():
            now = time.time()
            _drain_command_queue(
                command_queue=command_queue,
                state=state,
                context=context,
                question_interval_enabled=question_interval_enabled,
                emit=emit,
            )
            _drain_custom_query_requests(
                state=state,
                context=context,
                main_prompt=main_prompt,
                llm_client=llm_client,
                logger=logger,
                emit=emit,
            )
            _drain_recent_answer_requests(
                state=state,
                context=context,
                main_prompt=main_prompt,
                llm_client=llm_client,
                logger=logger,
                emit=emit,
            )

            if max_runtime is not None and now - start_time >= max_runtime:
                state.request_stop()
                continue

            if state.is_paused():
                if state.consume_force_generation_request():
                    _emit_questions(
                        context,
                        main_prompt,
                        llm_client,
                        logger,
                        ai_language=state.get_ai_language(),
                        style_instruction=state.get_ai_style_instruction(),
                        emit=emit,
                    )
                if state.consume_snapshot_request():
                    _save_snapshot(context, output_dir, logger, emit=emit)

                if (
                    controls_enabled
                    and next_controls_reminder is not None
                    and now >= next_controls_reminder
                ):
                    _print_controls_help(question_interval_enabled, emit=emit, reminder=True)
                    next_controls_reminder = now + CONTROL_REMINDER_SECONDS

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
                            emit=emit,
                        )

            if state.consume_force_generation_request():
                _emit_questions(
                    context,
                    main_prompt,
                    llm_client,
                    logger,
                    ai_language=state.get_ai_language(),
                    style_instruction=state.get_ai_style_instruction(),
                    emit=emit,
                )

            now = time.time()
            if question_interval_enabled and next_question_time is not None and now >= next_question_time:
                _emit_questions(
                    context,
                    main_prompt,
                    llm_client,
                    logger,
                    ai_language=state.get_ai_language(),
                    style_instruction=state.get_ai_style_instruction(),
                    emit=emit,
                )
                next_question_time = now + question_interval

            if state.consume_snapshot_request():
                _save_snapshot(context, output_dir, logger, emit=emit)

            if (
                controls_enabled
                and next_controls_reminder is not None
                and now >= next_controls_reminder
            ):
                _print_controls_help(question_interval_enabled, emit=emit, reminder=True)
                next_controls_reminder = now + CONTROL_REMINDER_SECONDS

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
                    emit=emit,
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
    controls_enabled: bool,
    show_controls_help: bool,
    output_dir: Path | None,
    start_time: float,
    max_runtime: int | None,
    command_queue: queue.Queue[str] | None,
    emit: OutputEmitter,
) -> None:
    _emit(emit, "Dry-run mode enabled. No real audio capture is used.")
    if controls_enabled and show_controls_help:
        _print_controls_help(question_interval_enabled, emit=emit)

    scripted = [
        ("remote", "SPEAKER_00", "Zacznijmy od celow projektu i zakresu MVP."),
        ("mic", None, "Mamy ograniczony budzet i dwa sprinty na wdrozenie."),
        ("remote", "SPEAKER_01", "Najwieksze ryzyko to opoznione integracje API partnera."),
        ("mic", None, "Potrzebujemy tez metryk sukcesu i planu rolloutu."),
    ]

    index = 0
    next_phrase_time = start_time
    next_question_time = start_time + question_interval if question_interval_enabled else None
    next_controls_reminder = (
        time.time() + CONTROL_REMINDER_SECONDS if controls_enabled and show_controls_help else None
    )

    while not state.should_stop():
        now = time.time()

        _drain_command_queue(
            command_queue=command_queue,
            state=state,
            context=context,
            question_interval_enabled=question_interval_enabled,
            emit=emit,
        )
        _drain_custom_query_requests(
            state=state,
            context=context,
            main_prompt=main_prompt,
            llm_client=llm_client,
            logger=logger,
            emit=emit,
        )
        _drain_recent_answer_requests(
            state=state,
            context=context,
            main_prompt=main_prompt,
            llm_client=llm_client,
            logger=logger,
            emit=emit,
        )

        if max_runtime is not None and now - start_time >= max_runtime:
            state.request_stop()
            continue

        if state.is_paused():
            if state.consume_force_generation_request():
                _emit_questions(
                    context,
                    main_prompt,
                    llm_client,
                    logger,
                    ai_language=state.get_ai_language(),
                    style_instruction=state.get_ai_style_instruction(),
                    emit=emit,
                )
            if state.consume_snapshot_request():
                _save_snapshot(context, output_dir, logger, emit=emit)

            if (
                controls_enabled
                and next_controls_reminder is not None
                and now >= next_controls_reminder
            ):
                _print_controls_help(question_interval_enabled, emit=emit, reminder=True)
                next_controls_reminder = now + CONTROL_REMINDER_SECONDS

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
                _handle_transcript(source, text, context, logger, speaker_label=speaker_label, emit=emit)

        if state.consume_force_generation_request():
            _emit_questions(
                context,
                main_prompt,
                llm_client,
                logger,
                ai_language=state.get_ai_language(),
                style_instruction=state.get_ai_style_instruction(),
                emit=emit,
            )

        if question_interval_enabled and next_question_time is not None and now >= next_question_time:
            _emit_questions(
                context,
                main_prompt,
                llm_client,
                logger,
                ai_language=state.get_ai_language(),
                style_instruction=state.get_ai_style_instruction(),
                emit=emit,
            )
            next_question_time = now + question_interval

        if state.consume_snapshot_request():
            _save_snapshot(context, output_dir, logger, emit=emit)

        if (
            controls_enabled
            and next_controls_reminder is not None
            and now >= next_controls_reminder
        ):
            _print_controls_help(question_interval_enabled, emit=emit, reminder=True)
            next_controls_reminder = now + CONTROL_REMINDER_SECONDS

        time.sleep(0.05)


def _handle_transcript(
    source: str,
    text: str,
    context: ConversationContext,
    logger: JsonlLogger | None,
    speaker_label: str | None = None,
    emit: OutputEmitter = _default_emit,
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
    _emit(emit, f"[{stamp}] [{label}] {cleaned}")

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
    ai_language: str,
    style_instruction: str,
    emit: OutputEmitter,
) -> None:
    # Render prompt synchronously (safe snapshot of context) then run LLM in background
    payload = context.render_for_prompt(main_prompt=main_prompt, ai_language=ai_language)
    if not payload:
        _emit(emit, "[AI] Waiting for more conversation context.")
        return

    def _worker(prompt_payload: str) -> None:
        try:
            answer = llm_client.generate_questions(
                prompt_payload,
                ai_language=ai_language,
                style_instruction=style_instruction,
            )
        except Exception as exc:  # pragma: no cover - network/LLM errors
            _emit(emit, f"[AI error] Generation failed: {exc}")
            return

        _emit(emit, "\n[AI QUESTIONS]\n" + answer + "\n")
        if logger is not None:
            logger.log("ai_questions", {"text": answer})

    thread = threading.Thread(target=_worker, args=(payload,), daemon=True)
    thread.start()


def _drain_custom_query_requests(
    state: RuntimeState,
    context: ConversationContext,
    main_prompt: str,
    llm_client: LlmClient,
    logger: JsonlLogger | None,
    emit: OutputEmitter,
) -> None:
    for query in state.consume_custom_query_requests():
        ai_language = state.get_ai_language()
        style_instruction = state.get_ai_style_instruction()
        _emit_custom_query_response(
            query=query,
            context=context,
            main_prompt=main_prompt,
            llm_client=llm_client,
            logger=logger,
            ai_language=ai_language,
            style_instruction=style_instruction,
            emit=emit,
        )


def _drain_recent_answer_requests(
    state: RuntimeState,
    context: ConversationContext,
    main_prompt: str,
    llm_client: LlmClient,
    logger: JsonlLogger | None,
    emit: OutputEmitter,
) -> None:
    for window in state.consume_recent_answer_requests():
        ai_language = state.get_ai_language()
        style_instruction = state.get_ai_style_instruction()
        _emit_recent_answers(
            context=context,
            main_prompt=main_prompt,
            llm_client=llm_client,
            logger=logger,
            ai_language=ai_language,
            style_instruction=style_instruction,
            window_seconds=window,
            emit=emit,
        )


def _emit_recent_answers(
    context: ConversationContext,
    main_prompt: str,
    llm_client: LlmClient,
    logger: JsonlLogger | None,
    ai_language: str,
    style_instruction: str,
    window_seconds: int | None,
    emit: OutputEmitter,
) -> None:
    # Snapshot recent lines synchronously to avoid concurrent access races
    now = time.time()

    if window_seconds is None:
        lines = context.all_lines()
    else:
        cutoff = now - max(0, int(window_seconds))
        lines = [line for line in context.all_lines() if line.timestamp >= cutoff]

    if not lines:
        _emit(emit, f"[AI] No recent transcript in the selected window ({_format_context_window_spec(window_seconds)}).")
        return

    language = ai_language if ai_language in {"en", "pl"} else ("pl" if not _parse_ai_language(ai_language) else _parse_ai_language(ai_language))
    transcript_lines: list[str] = []
    for line in lines:
        label = context._label_for_line(line, ai_language=language)
        transcript_lines.append(f"[{label}] {line.text}")

    transcript = "\n".join(transcript_lines)

    if language == "en":
        prompt_payload = (
            f"MAIN ROLE PROMPT:\n{main_prompt.strip()}\n\n"
            f"RECENT CONVERSATION CONTEXT (last {_format_context_window_spec(window_seconds)}):\n{transcript}\n\n"
            "Identify any direct questions present in the conversation above and answer them directly and concisely. If there are no direct questions, reply with a short message stating that."
        )
    else:
        prompt_payload = (
            f"GLOWNY PROMPT ROLI:\n{main_prompt.strip()}\n\n"
            f"KONTEKST OSTATNIEJ ROZMOWY (ostatnie {_format_context_window_spec(window_seconds)}):\n{transcript}\n\n"
            "Znajdz bezposrednie pytania w powyzszym kontekscie i odpowiedz na nie krotko i rzeczowo. Jesli nie ma pytan, odpowiedz kroto, ze brak bezposrednich pytan."
        )

    def _worker(prompt_payload: str, win: int | None) -> None:
        try:
            answer = llm_client.generate_questions(
                prompt_payload,
                ai_language=ai_language,
                style_instruction=style_instruction,
            )
        except Exception as exc:  # pragma: no cover - network/LLM errors
            _emit(emit, f"[AI error] Recent-answers generation failed: {exc}")
            return

        _emit(emit, "\n[AI RECENT ANSWERS]\n" + answer + "\n")
        if logger is not None:
            logger.log("ai_recent_answers", {"window_seconds": win, "text": answer})

    thread = threading.Thread(target=_worker, args=(prompt_payload, window_seconds), daemon=True)
    thread.start()


def _emit_custom_query_response(
    query: str,
    context: ConversationContext,
    main_prompt: str,
    llm_client: LlmClient,
    logger: JsonlLogger | None,
    ai_language: str,
    style_instruction: str,
    emit: OutputEmitter,
) -> None:
    is_english = ai_language == "en"
    # Build a payload for custom user queries that does NOT ask the model
    # to propose follow-up questions (avoid mixing custom answers with
    # the periodic "g" question-generation behaviour).
    # Build prompt payload synchronously (snapshot) then call LLM in background
    transcript = context.render_full_transcript()
    if transcript:
        if is_english:
            base_payload = f"Main prompt:\n{main_prompt}\n\nConversation context:\n{transcript}"
        else:
            base_payload = f"Glowny prompt:\n{main_prompt}\n\nKontekst rozmowy:\n{transcript}"
    else:
        if is_english:
            base_payload = f"Main prompt:\n{main_prompt}\n\nConversation context:\n(no transcript yet)"
        else:
            base_payload = f"Glowny prompt:\n{main_prompt}\n\nKontekst rozmowy:\n(brak transkrypcji)"

    if is_english:
        prompt_payload = (
            f"{base_payload}\n\n"
            "User custom request:\n"
            f"{query}\n\n"
            "Answer this custom request directly and keep the answer practical."
        )
    else:
        prompt_payload = (
            f"{base_payload}\n\n"
            "Dodatkowe zapytanie uzytkownika:\n"
            f"{query}\n\n"
            "Odpowiedz bezposrednio na to zapytanie i trzymaj odpowiedz praktyczna."
        )

    def _worker(prompt_payload: str, user_query: str) -> None:
        try:
            answer = llm_client.generate_questions(
                prompt_payload,
                ai_language=ai_language,
                style_instruction=style_instruction,
            )
        except Exception as exc:  # pragma: no cover - network/LLM errors
            _emit(emit, f"[AI error] Custom request failed: {exc}")
            return

        _emit(emit, "\n[AI CUSTOM RESPONSE]\n" + answer + "\n")
        if logger is not None:
            logger.log(
                "ai_custom_query",
                {
                    "query": user_query,
                    "text": answer,
                },
            )

    thread = threading.Thread(target=_worker, args=(prompt_payload, query), daemon=True)
    thread.start()


def _save_snapshot(
    context: ConversationContext,
    output_dir: Path | None,
    logger: JsonlLogger | None,
    emit: OutputEmitter,
) -> None:
    if output_dir is None:
        _emit(emit, "Snapshot skipped: output directory not set.")
        return

    snapshot_path = output_dir / f"snapshot-{int(time.time())}.txt"
    rendered = context.render_for_prompt(main_prompt="Snapshot")
    snapshot_path.write_text(rendered or "(no data)", encoding="utf-8")
    _emit(emit, f"Snapshot saved: {snapshot_path}")

    if logger is not None:
        logger.log("snapshot", {"path": str(snapshot_path)})


def _start_controls_thread(
    state: RuntimeState,
    context: ConversationContext,
    question_interval_enabled: bool,
    emit: OutputEmitter,
) -> threading.Thread:
    def _loop() -> None:
        while not state.should_stop():
            try:
                raw = input()
            except EOFError:
                break

            cleaned = raw.strip()
            if not cleaned:
                continue

            _process_control_command(
                raw=cleaned,
                state=state,
                context=context,
                question_interval_enabled=question_interval_enabled,
                emit=emit,
            )

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


def _drain_command_queue(
    command_queue: queue.Queue[str] | None,
    state: RuntimeState,
    context: ConversationContext,
    question_interval_enabled: bool,
    emit: OutputEmitter,
) -> None:
    if command_queue is None:
        return

    while True:
        try:
            raw = command_queue.get_nowait()
        except queue.Empty:
            break

        _process_control_command(
            raw=raw,
            state=state,
            context=context,
            question_interval_enabled=question_interval_enabled,
            emit=emit,
        )


def _process_control_command(
    raw: str,
    state: RuntimeState,
    context: ConversationContext,
    question_interval_enabled: bool,
    emit: OutputEmitter,
) -> None:
    cleaned = raw.strip()
    if not cleaned:
        return

    parts = cleaned.split(maxsplit=1)
    cmd = parts[0].lower()

    if cmd in {"h", "help", "?"}:
        _print_controls_help(question_interval_enabled, emit=emit)
    elif cmd == "p":
        paused = state.toggle_pause()
        _emit(emit, "Paused." if paused else "Resumed.")
    elif cmd == "m":
        mic_enabled = state.toggle_mic()
        _emit(emit, "Mic listening ON." if mic_enabled else "Mic listening OFF.")
    elif cmd == "i":
        seconds = 30
        if len(parts) > 1:
            try:
                seconds = int(parts[1].split()[0])
            except ValueError:
                seconds = 30
        until = state.set_ignore_remote_for(seconds)
        _emit(emit, f"Ignoring remote audio until {time.strftime('%H:%M:%S', time.localtime(until))}.")
    elif cmd == "x":
        if len(parts) < 2:
            _emit(emit, "Usage: x <speaker_id>")
            return

        speaker_id = parts[1].strip()
        enabled = state.toggle_ignore_speaker(speaker_id)
        if enabled:
            _emit(emit, f"Speaker ignored: {speaker_id}")
        else:
            _emit(emit, f"Speaker unignored: {speaker_id}")
    elif cmd == "k":
        known = state.list_known_speakers()
        ignored = state.list_ignored_speakers()
        _emit(emit, "Known speakers: " + (", ".join(known) if known else "none"))
        _emit(emit, "Ignored speakers: " + (", ".join(ignored) if ignored else "none"))
    elif cmd == "g":
        state.request_force_generation()
        _emit(emit, "Manual AI generation requested.")
    elif cmd in {"a", "ask"}:
        query = parts[1].strip() if len(parts) > 1 else ""
        if not state.request_custom_query(query):
            _emit(emit, "Usage: a <your custom AI query>")
            return
        _emit(emit, "Custom AI query requested.")
    elif cmd in {"ra", "ans"}:
        # Recent-answers: answer direct questions found in recent context window.
        value = parts[1].strip() if len(parts) > 1 else ""
        # default to 2 minutes
        window: int | None = 120
        if value:
            token = value.split(maxsplit=1)[0]
            try:
                parsed = _parse_context_window_spec(token)
            except ValueError:
                _emit(emit, "Usage: ra <seconds|Xm|Xh|all> (examples: ra 120, ra 2m, ra all)")
                return
            window = parsed

        state.request_recent_answers(window)
        _emit(emit, f"Recent-answers requested for: {_format_context_window_spec(window)}")
    elif cmd in {"lang", "language", "l"}:
        language_value = ""
        if len(parts) > 1:
            language_value = parts[1].split()[0].strip().lower()
        if not language_value:
            current = state.get_ai_language()
            _emit(emit, f"Current AI language: {current}")
            _emit(emit, "Usage: lang <pl|en>")
            return

        updated = state.set_ai_language(language_value)
        if updated is None:
            _emit(emit, "Unsupported language. Use: pl or en")
            return

        _emit(emit, f"AI language set to: {updated}")
    elif cmd in {"style", "st"}:
        style_value = parts[1].strip() if len(parts) > 1 else ""
        if not style_value:
            current_style = state.get_ai_style_instruction()
            if current_style:
                _emit(emit, f"Current AI style: {current_style}")
            else:
                _emit(emit, "Current AI style: (not set)")
            _emit(emit, "Usage: style <text> or style clear")
            return

        if style_value.lower() in {"clear", "off", "none"}:
            state.set_ai_style_instruction("")
            _emit(emit, "AI style instruction cleared.")
            return

        state.set_ai_style_instruction(style_value)
        _emit(emit, "AI style instruction updated.")
    elif cmd in {"ctx", "context", "window"}:
        value = parts[1].strip() if len(parts) > 1 else ""
        if not value:
            current_window = context.get_window_seconds()
            _emit(emit, f"Current AI context window: {_format_context_window_spec(current_window)}")
            _emit(emit, "Usage: ctx <seconds|Xm|Xh|all> (examples: ctx 1200, ctx 20m, ctx all)")
            return

        try:
            parsed_window = _parse_context_window_spec(value)
        except ValueError:
            _emit(emit, "Invalid context window. Use e.g. ctx 1200, ctx 20m, ctx 2h, ctx all")
            return

        context.set_window_seconds(parsed_window)
        _emit(emit, f"AI context window set to: {_format_context_window_spec(parsed_window)}")
    elif cmd == "s":
        state.request_snapshot()
        _emit(emit, "Snapshot requested.")
    elif cmd == "q":
        state.request_stop()
        _emit(emit, "Stop requested.")
    else:
        _emit(
            emit,
            "Unknown command. Use: h, p, m, i [sec], x <speaker>, k, g, a <text>, ra [window], "
            "lang <pl|en>, style <text>, ctx <20m|all>, s, q",
        )


def _resolve_stt_backend_with_fallback(config: AppConfig, emit: OutputEmitter) -> str:
    backend = config.stt_backend.strip().lower()
    if backend != "vosk":
        return backend

    if config.vosk_model_path.strip():
        return "vosk"

    whisper_model = config.whisper_cpp_model_path.strip()
    if whisper_model:
        _emit(emit, "VOSK model path is empty. Falling back to whisper_cpp backend.")
        return "whisper_cpp"

    raise RuntimeError(
        "VOSK model path is empty and whisper_cpp model path is not configured. "
        "Set at least one STT model path in settings."
    )


def _resolve_live_audio_sources(
    config: AppConfig,
    allow_prompt: bool,
    emit: OutputEmitter,
) -> tuple[str, str]:
    monitors = list_monitor_sources()
    mics = list_mic_sources()

    monitor_source = config.monitor_source.strip()
    mic_source = config.mic_source.strip()

    if monitor_source and monitor_source not in monitors:
        _emit(emit, f"Configured monitor source not found: {monitor_source}")
        monitor_source = ""

    if not monitor_source:
        if not monitors:
            raise RuntimeError(
                "No monitor sources detected. Route meeting audio to a monitor source first (check pactl list short sources)."
            )

        if len(monitors) == 1:
            monitor_source = monitors[0]
            _emit(emit, f"Auto-selected monitor source: {monitor_source}")
        else:
            default_monitor = choose_default_monitor_source(monitors)
            if allow_prompt:
                monitor_source = _prompt_select_source(
                    "monitor",
                    monitors,
                    default_source=default_monitor,
                    allow_skip=False,
                    emit=emit,
                )
            else:
                monitor_source = default_monitor or monitors[0]
                _emit(emit, f"Auto-selected monitor source: {monitor_source}")

    if not config.mic_listening_enabled:
        return monitor_source, ""

    if mic_source and mic_source not in mics:
        _emit(emit, f"Configured mic source not found: {mic_source}")
        mic_source = ""

    if not mic_source:
        if not mics:
            _emit(emit, "No mic sources detected. Continuing with remote-only capture.")
            return monitor_source, ""

        if len(mics) == 1:
            mic_source = mics[0]
            _emit(emit, f"Auto-selected mic source: {mic_source}")
        else:
            default_mic = choose_default_mic_source(mics)
            if allow_prompt:
                mic_source = _prompt_select_source(
                    "microphone",
                    mics,
                    default_source=default_mic,
                    allow_skip=True,
                    emit=emit,
                )
            else:
                mic_source = default_mic or mics[0]
                _emit(emit, f"Auto-selected mic source: {mic_source}")

    return monitor_source, mic_source


def _prompt_select_source(
    kind: str,
    sources: list[str],
    default_source: str | None,
    allow_skip: bool,
    emit: OutputEmitter,
) -> str:
    _emit(emit, f"Detected multiple {kind} sources:")
    for index, source in enumerate(sources, start=1):
        marker = " (default)" if source == default_source else ""
        _emit(emit, f"  {index}. {source}{marker}")

    if allow_skip:
        _emit(emit, "  0. Disable this source")

    while True:
        prompt = f"Select {kind} source"
        if default_source:
            prompt += f" [Enter={default_source}]"
        prompt += ": "

        raw = input(prompt).strip()
        if not raw:
            if default_source:
                return default_source
            if not allow_skip:
                _emit(emit, "Selection required.")
                continue
            return ""

        try:
            selected = int(raw)
        except ValueError:
            _emit(emit, "Invalid selection. Enter a number.")
            continue

        if allow_skip and selected == 0:
            return ""
        if 1 <= selected <= len(sources):
            return sources[selected - 1]
        _emit(emit, "Selection out of range.")
