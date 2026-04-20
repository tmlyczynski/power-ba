from power_ba.config import AppConfig
from power_ba.runtime import run_session


def main() -> None:
    cfg = AppConfig(
        provider="openai",
        model="gpt-5.4-nano",
        openai_api_key="",
        anthropic_api_key="",
        mic_listening_enabled=True,
        question_interval_seconds=5,
        context_window_seconds=60,
        main_prompt="Jestes business analitykiem. Przygotuj trafne pytania.",
        stt_backend="vosk",
        vosk_model_path="",
        mic_source="",
        monitor_source="",
    )

    run_session(
        config=cfg,
        question_interval_override=5,
        output_dir=None,
        dry_run=True,
        max_runtime=12,
        interactive_controls=False,
    )


if __name__ == "__main__":
    main()
