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
    question_interval_enabled: bool = True
    question_interval_seconds: int = 30
    context_window_seconds: int = 120
    ai_language: str = "pl"
    main_prompt: str = DEFAULT_MAIN_PROMPT
    stt_backend: str = "vosk"
    vosk_model_path: str = ""
    whisper_cpp_model_path: str = ""
    whisper_cpp_binary: str = "whisper-cli"
    whisper_cpp_chunk_seconds: int = 5
    whisper_cpp_language: str = "pl"
    mic_source: str = ""
    monitor_source: str = ""
    default_output_dir: str = ""
    save_audio_by_default: bool = False
    save_transcript_by_default: bool = False
    diarization_enabled: bool = False
    diarization_backend: str = "pyannote"
    pyannote_hf_token: str = ""
    pyannote_model_name: str = "pyannote/speaker-diarization-3.1"
    diarization_interval_seconds: int = 15
    diarization_max_buffer_seconds: int = 180

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
        normalized_language = self.ai_language.strip().lower()
        if normalized_language in {"en", "english"}:
            self.ai_language = "en"
        else:
            self.ai_language = "pl"
        required_window = self.question_interval_seconds if self.question_interval_enabled else 60
        if self.context_window_seconds < required_window:
            self.context_window_seconds = max(required_window, 60)
        if self.stt_backend not in {"vosk", "whisper_cpp"}:
            self.stt_backend = "vosk"
        if self.whisper_cpp_chunk_seconds < 2:
            self.whisper_cpp_chunk_seconds = 2
        if not self.whisper_cpp_binary.strip():
            self.whisper_cpp_binary = "whisper-cli"
        if not self.whisper_cpp_language.strip():
            self.whisper_cpp_language = "pl"
        if self.diarization_backend not in {"pyannote"}:
            self.diarization_backend = "pyannote"
        if self.diarization_interval_seconds < 5:
            self.diarization_interval_seconds = 5
        if self.diarization_max_buffer_seconds < 30:
            self.diarization_max_buffer_seconds = 30
        if not self.main_prompt.strip():
            self.main_prompt = DEFAULT_MAIN_PROMPT
        self.default_output_dir = self.default_output_dir.strip()


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
