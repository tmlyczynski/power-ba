from __future__ import annotations

import getpass
from dataclasses import asdict
from pathlib import Path

import typer

from .capture import list_pulse_sources
from .config import AppConfig, DEFAULT_CONFIG_PATH, load_config, save_config
from .runtime import run_session

app = typer.Typer(add_completion=False, no_args_is_help=False, help="Power BA meeting assistant CLI")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        _menu_loop(DEFAULT_CONFIG_PATH)


@app.command()
def menu(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Path to config YAML"),
) -> None:
    """Open interactive menu."""
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

    for source in sources:
        print(source)


@app.command()
def start(
    config_path: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", help="Path to config YAML"),
    question_interval: int | None = typer.Option(None, "--question-interval", help="Seconds between AI question generation"),
    provider: str | None = typer.Option(None, "--provider", help="openai or anthropic"),
    model: str | None = typer.Option(None, "--model", help="LLM model name"),
    mic_listening: bool | None = typer.Option(None, "--mic/--no-mic", help="Enable or disable microphone listening"),
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
    if provider is not None:
        runtime_config.provider = provider.strip().lower()
    if model is not None and model.strip():
        runtime_config.model = model.strip()
    if mic_listening is not None:
        runtime_config.mic_listening_enabled = mic_listening

    runtime_config.sanitize()

    try:
        run_session(
            config=runtime_config,
            question_interval_override=question_interval,
            output_dir=output,
            dry_run=dry_run,
            max_runtime=max_runtime,
            interactive_controls=not no_controls,
        )
    except Exception as exc:  # pragma: no cover
        print(f"Session failed: {exc}")


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
            output_raw = input("Output directory (blank = no file output): ").strip()
            output_dir = Path(output_raw).expanduser() if output_raw else None
            dry_run = _ask_yes_no("Dry-run mode? [y/N]: ", default=False)

            runtime_config = _copy_config(config)
            if interval_override is not None:
                runtime_config.question_interval_seconds = interval_override

            try:
                run_session(
                    config=runtime_config,
                    question_interval_override=interval_override,
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
        print(f"5. Mic listening         [{'on' if working.mic_listening_enabled else 'off'}]")
        print(f"6. Question interval     [{working.question_interval_seconds}s]")
        print(f"7. Main role prompt      [{working.main_prompt[:45]}...]")
        print(f"8. Mic source            [{working.mic_source or 'not set'}]")
        print(f"9. Monitor source        [{working.monitor_source or 'not set'}]")
        print(f"10. VOSK model path      [{working.vosk_model_path or 'not set'}]")
        print("11. Save and back")
        print("12. Cancel and back")

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
            working.mic_listening_enabled = not working.mic_listening_enabled
        elif choice == "6":
            val = _ask_optional_int("Question interval in seconds: ")
            if val is not None:
                working.question_interval_seconds = val
        elif choice == "7":
            prompt = input("Main role prompt: ").strip()
            if prompt:
                working.main_prompt = prompt
        elif choice == "8":
            selected = _select_source("mic", working.mic_source)
            if selected is not None:
                working.mic_source = selected
        elif choice == "9":
            selected = _select_source("monitor", working.monitor_source)
            if selected is not None:
                working.monitor_source = selected
        elif choice == "10":
            model_path = input("VOSK model path: ").strip()
            working.vosk_model_path = model_path
        elif choice == "11":
            saved_path = save_config(working, config_path)
            print(f"Config saved: {saved_path}")
            return working
        elif choice == "12":
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
    sources = list_pulse_sources()
    if not sources:
        print("No PulseAudio/PipeWire sources found.")
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
    app()
