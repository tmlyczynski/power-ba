from __future__ import annotations

import getpass
from dataclasses import asdict
from pathlib import Path

import typer

from .capture import list_mic_sources, list_monitor_sources, list_pulse_sources
from .config import AppConfig, DEFAULT_CONFIG_PATH, load_config, save_config
from .runtime import run_session

app = typer.Typer(add_completion=False, no_args_is_help=False, help="Power BA meeting assistant CLI")


def run() -> None:
    app()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _default_ui(DEFAULT_CONFIG_PATH)


@app.command()
def menu(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Path to config YAML"),
) -> None:
    """Open interactive menu."""
    _menu_loop(config_path)


@app.command()
def tui(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Path to config YAML"),
) -> None:
    """Run modern TUI menu."""
    if not _launch_tui_or_fallback(config_path):
        _menu_loop(config_path)


@app.command()
def settings(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Path to config YAML"),
) -> None:
    """Edit settings without starting capture."""
    config = load_config(config_path)
    _settings_loop(config, config_path)


@app.command("list-sources")
def list_sources() -> None:
    """List PulseAudio/PipeWire source names."""
    sources = list_pulse_sources()
    if not sources:
        print("No sources found. Is pactl available?")
        return

    monitors = list_monitor_sources()
    mics = list_mic_sources()

    print("Monitor sources:")
    if monitors:
        for source in monitors:
            print(f"  - {source}")
    else:
        print("  (none)")

    print("Microphone sources:")
    if mics:
        for source in mics:
            print(f"  - {source}")
    else:
        print("  (none)")


@app.command()
def start(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Path to config YAML"),
    question_interval: int | None = typer.Option(None, "--question-interval", help="Seconds between AI question generation"),
    auto_interval: bool | None = typer.Option(None, "--auto-interval/--no-auto-interval", help="Enable or disable automatic interval generation"),
    provider: str | None = typer.Option(None, "--provider", help="openai or anthropic"),
    model: str | None = typer.Option(None, "--model", help="LLM model name"),
    ai_language: str | None = typer.Option(None, "--ai-language", help="AI response language: pl or en"),
    stt_backend: str | None = typer.Option(None, "--stt-backend", help="vosk or whisper_cpp"),
    whisper_model_path: str | None = typer.Option(None, "--whisper-model-path", help="Path to whisper.cpp model"),
    whisper_binary: str | None = typer.Option(None, "--whisper-binary", help="whisper.cpp binary name/path"),
    diarization: bool | None = typer.Option(None, "--diarization/--no-diarization", help="Enable or disable diarization"),
    mic_listening: bool | None = typer.Option(None, "--mic/--no-mic", help="Enable or disable microphone listening"),
    save_audio: bool | None = typer.Option(None, "--save-audio/--no-save-audio", help="Save captured audio files"),
    save_transcript: bool | None = typer.Option(
        None,
        "--save-transcript/--no-save-transcript",
        help="Save transcript and AI logs to jsonl",
    ),
    output: Path | None = typer.Option(None, "--output", help="Output directory for wav/jsonl logs"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run with simulated transcripts"),
    max_runtime: int | None = typer.Option(None, "--max-runtime", help="Stop automatically after N seconds"),
    no_controls: bool = typer.Option(False, "--no-controls", help="Disable interactive runtime controls"),
) -> None:
    """Start transcription + AI question session."""
    config = load_config(config_path)
    runtime_config = _copy_config(config)

    if question_interval is not None:
        runtime_config.question_interval_seconds = question_interval
        if auto_interval is None and question_interval > 0:
            runtime_config.question_interval_enabled = True
    if auto_interval is not None:
        runtime_config.question_interval_enabled = auto_interval
    if provider is not None:
        runtime_config.provider = provider.strip().lower()
    if model is not None and model.strip():
        runtime_config.model = model.strip()
    if ai_language is not None and ai_language.strip():
        runtime_config.ai_language = ai_language.strip().lower()
    if stt_backend is not None:
        runtime_config.stt_backend = stt_backend.strip().lower()
    if whisper_model_path is not None:
        runtime_config.whisper_cpp_model_path = whisper_model_path.strip()
    if whisper_binary is not None:
        runtime_config.whisper_cpp_binary = whisper_binary.strip()
    if diarization is not None:
        runtime_config.diarization_enabled = diarization
    if mic_listening is not None:
        runtime_config.mic_listening_enabled = mic_listening
    if save_audio is not None:
        runtime_config.save_audio_by_default = save_audio
    if save_transcript is not None:
        runtime_config.save_transcript_by_default = save_transcript

    runtime_config.sanitize()

    try:
        run_session(
            config=runtime_config,
            question_interval_override=question_interval,
            interval_enabled_override=auto_interval,
            output_dir=output,
            save_audio_override=save_audio,
            save_transcript_override=save_transcript,
            dry_run=dry_run,
            max_runtime=max_runtime,
            interactive_controls=not no_controls,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Session failed: {exc}")


def _default_ui(config_path: Path) -> None:
    if _launch_tui_or_fallback(config_path):
        return
    _menu_loop(config_path)


def _launch_tui_or_fallback(config_path: Path) -> bool:
    try:
        from .tui import launch_tui
    except Exception:
        print("TUI unavailable, switching to legacy menu.")
        return False

    launch_tui(config_path)
    return True


def _menu_loop(config_path: Path) -> None:
    config = load_config(config_path)

    while True:
        print("\n=== Power BA Menu ===")
        print("1. Start transcription + AI questions")
        print("2. Settings")
        print("3. Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            interval_override = _ask_optional_int(
                f"Question interval in seconds [{config.question_interval_seconds}]: "
            )
            output_default_label = config.default_output_dir or "no file output"
            output_raw = input(f"Output directory (blank = {output_default_label}): ").strip()
            if output_raw:
                output_dir = Path(output_raw).expanduser()
            elif config.default_output_dir:
                output_dir = Path(config.default_output_dir).expanduser()
            else:
                output_dir = None
            dry_run = _ask_yes_no("Dry-run mode? [y/N]: ", default=False)

            runtime_config = _copy_config(config)
            if interval_override is not None:
                runtime_config.question_interval_seconds = interval_override

            try:
                run_session(
                    config=runtime_config,
                    question_interval_override=interval_override,
                    interval_enabled_override=True if interval_override is not None else None,
                    output_dir=output_dir,
                    dry_run=dry_run,
                    interactive_controls=True,
                )
            except Exception as exc:
                print(f"Session failed: {exc}")

        elif choice == "2":
            config = _settings_loop(config, config_path)

        elif choice == "3":
            print("Bye.")
            break

        else:
            print("Invalid option.")


def _settings_loop(config: AppConfig, config_path: Path) -> AppConfig:
    working = _copy_config(config)

    while True:
        print("\n=== Settings ===")
        print(f"1. OpenAI API key        [{'set' if working.openai_api_key else 'empty'}]")
        print(f"2. Anthropic API key     [{'set' if working.anthropic_api_key else 'empty'}]")
        print(f"3. Provider              [{working.provider}]")
        print(f"4. Model                 [{working.model}]")
        print(f"5. AI language           [{working.ai_language}]")
        print(f"6. Mic listening         [{'on' if working.mic_listening_enabled else 'off'}]")
        print(f"7. Question interval     [{working.question_interval_seconds}s]")
        print(f"8. Auto interval         [{'on' if working.question_interval_enabled else 'off'}]")
        print(f"9. AI context window     [{working.ai_context_window_default}]")
        print(f"10. Main role prompt     [{working.main_prompt[:45]}...]")
        print(f"11. Mic source           [{working.mic_source or 'not set'}]")
        print(f"12. Monitor source       [{working.monitor_source or 'not set'}]")
        print(f"13. Output directory     [{working.default_output_dir or 'not set'}]")
        print(f"14. Save audio files     [{'on' if working.save_audio_by_default else 'off'}]")
        print(f"15. Save transcript logs [{'on' if working.save_transcript_by_default else 'off'}]")
        print(f"16. STT backend          [{working.stt_backend}]")
        print(f"17. VOSK model path      [{working.vosk_model_path or 'not set'}]")
        print(f"18. whisper model path   [{working.whisper_cpp_model_path or 'not set'}]")
        print(f"19. whisper binary       [{working.whisper_cpp_binary}]")
        print(f"20. whisper chunk sec    [{working.whisper_cpp_chunk_seconds}s]")
        print(f"21. Diarization          [{'on' if working.diarization_enabled else 'off'}]")
        print(f"22. pyannote token       [{'set' if working.pyannote_hf_token else 'empty'}]")
        print(f"23. pyannote model       [{working.pyannote_model_name}]")
        print(f"24. diarization interval [{working.diarization_interval_seconds}s]")
        print("25. Save and back")
        print("26. Cancel and back")

        choice = input("Select setting: ").strip()

        if choice == "1":
            key = getpass.getpass("OpenAI API key (blank clears): ").strip()
            working.openai_api_key = key
        elif choice == "2":
            key = getpass.getpass("Anthropic API key (blank clears): ").strip()
            working.anthropic_api_key = key
        elif choice == "3":
            new_provider = input("Provider [openai/anthropic]: ").strip().lower()
            if new_provider in {"openai", "anthropic"}:
                working.provider = new_provider
            else:
                print("Unsupported provider.")
        elif choice == "4":
            model = input("Model name: ").strip()
            if model:
                working.model = model
        elif choice == "5":
            language = input("AI language [pl/en]: ").strip().lower()
            if language in {"pl", "en"}:
                working.ai_language = language
            else:
                print("Unsupported language.")
        elif choice == "6":
            working.mic_listening_enabled = not working.mic_listening_enabled
        elif choice == "7":
            val = _ask_optional_int("Question interval in seconds: ")
            if val is not None:
                working.question_interval_seconds = val
        elif choice == "8":
            working.question_interval_enabled = not working.question_interval_enabled
        elif choice == "9":
            ctx_window = input("Default AI context window [e.g. 20m / 2h / 1200 / all]: ").strip()
            working.ai_context_window_default = ctx_window
        elif choice == "10":
            prompt = input("Main role prompt: ").strip()
            if prompt:
                working.main_prompt = prompt
        elif choice == "11":
            selected = _select_source("mic", working.mic_source)
            if selected is not None:
                working.mic_source = selected
        elif choice == "12":
            selected = _select_source("monitor", working.monitor_source)
            if selected is not None:
                working.monitor_source = selected
        elif choice == "13":
            output_dir = input("Default output directory (blank disables): ").strip()
            working.default_output_dir = output_dir
        elif choice == "14":
            working.save_audio_by_default = not working.save_audio_by_default
        elif choice == "15":
            working.save_transcript_by_default = not working.save_transcript_by_default
        elif choice == "16":
            backend = input("STT backend [vosk/whisper_cpp]: ").strip().lower()
            if backend in {"vosk", "whisper_cpp"}:
                working.stt_backend = backend
            else:
                print("Unsupported STT backend.")
        elif choice == "17":
            model_path = input("VOSK model path: ").strip()
            working.vosk_model_path = model_path
        elif choice == "18":
            model_path = input("whisper.cpp model path: ").strip()
            working.whisper_cpp_model_path = model_path
        elif choice == "19":
            binary = input("whisper.cpp binary (default whisper-cli): ").strip()
            if binary:
                working.whisper_cpp_binary = binary
        elif choice == "20":
            val = _ask_optional_int("whisper.cpp chunk seconds: ")
            if val is not None:
                working.whisper_cpp_chunk_seconds = val
        elif choice == "21":
            working.diarization_enabled = not working.diarization_enabled
        elif choice == "22":
            token = getpass.getpass("pyannote HF token (blank clears): ").strip()
            working.pyannote_hf_token = token
        elif choice == "23":
            name = input("pyannote model name: ").strip()
            if name:
                working.pyannote_model_name = name
        elif choice == "24":
            val = _ask_optional_int("Diarization interval in seconds: ")
            if val is not None:
                working.diarization_interval_seconds = val
        elif choice == "25":
            saved_path = save_config(working, config_path)
            print(f"Config saved: {saved_path}")
            return working
        elif choice == "26":
            print("Canceled settings changes.")
            return config
        else:
            print("Invalid option.")

        working.sanitize()


def _copy_config(config: AppConfig) -> AppConfig:
    return AppConfig.from_dict(asdict(config))


def _ask_optional_int(prompt: str) -> int | None:
    raw = input(prompt).strip()
    if not raw:
        return None

    try:
        value = int(raw)
    except ValueError:
        print("Invalid number.")
        return None

    if value < 5:
        print("Value too small. Using minimum 5.")
        return 5
    return value


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    raw = input(prompt).strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "t", "true", "1"}


def _select_source(kind: str, current: str) -> str | None:
    if kind == "monitor":
        sources = list_monitor_sources()
    elif kind == "mic":
        sources = list_mic_sources()
    else:
        sources = list_pulse_sources()

    if not sources:
        print(f"No {kind} sources found.")
        return None

    print(f"Available {kind} sources:")
    for index, source in enumerate(sources, start=1):
        marker = " (current)" if source == current else ""
        print(f"{index}. {source}{marker}")

    raw = input("Choose index (blank to cancel): ").strip()
    if not raw:
        return None

    try:
        index = int(raw)
    except ValueError:
        print("Invalid index.")
        return None

    if 1 <= index <= len(sources):
        return sources[index - 1]

    print("Index out of range.")
    return None


if __name__ == "__main__":
    run()
