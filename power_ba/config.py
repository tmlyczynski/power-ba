from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path.home() / ".config" / "power-ba"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.yaml"
DEFAULT_MAIN_PROMPT = (
    "Jestes doswiadczonym business analitykiem technicznym. "
    "Analizuj rozmowe i przygotowuj konkretne, pomocne pytania doprecyzowujace."
)


@dataclass
class AppConfig:
    provider: str = "openai"
    model: str = "gpt-5-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    mic_listening_enabled: bool = True
    question_interval_seconds: int = 30
    context_window_seconds: int = 120
    main_prompt: str = DEFAULT_MAIN_PROMPT
    stt_backend: str = "vosk"
    vosk_model_path: str = ""
    mic_source: str = ""
    monitor_source: str = ""
    save_audio_by_default: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        allowed = set(cls.__dataclass_fields__.keys())
        filtered = {key: value for key, value in data.items() if key in allowed}
        config = cls(**filtered)
        config.sanitize()
        return config

    def sanitize(self) -> None:
        if self.provider not in {"openai", "anthropic"}:
            self.provider = "openai"
        if not self.model:
            self.model = "gpt-5-mini"
        if self.question_interval_seconds < 5:
            self.question_interval_seconds = 5
        if self.context_window_seconds < self.question_interval_seconds:
            self.context_window_seconds = max(self.question_interval_seconds, 60)
        if self.stt_backend not in {"vosk", "whisper_cpp"}:
            self.stt_backend = "vosk"
        if not self.main_prompt.strip():
            self.main_prompt = DEFAULT_MAIN_PROMPT


def resolve_config_path(config_path: Path | None = None) -> Path:
    if config_path is None:
        return DEFAULT_CONFIG_PATH
    return config_path.expanduser().resolve()


def load_config(config_path: Path | None = None) -> AppConfig:
    path = resolve_config_path(config_path)
    if not path.exists():
        config = AppConfig()
        save_config(config, path)
        return config

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        data = {}

    return AppConfig.from_dict(data)


def save_config(config: AppConfig, config_path: Path | None = None) -> Path:
    path = resolve_config_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config.sanitize()
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(asdict(config), file, sort_keys=False, allow_unicode=True)

    try:
        path.chmod(0o600)
    except PermissionError:
        pass

    return path
