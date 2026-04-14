from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Static, Switch

from .config import AppConfig, load_config, save_config


def _copy_config(config: AppConfig) -> AppConfig:
    return AppConfig.from_dict(asdict(config))


class MainMenuScreen(Screen[None]):
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Label("Power BA", id="title"),
            Static("Meeting transcription + AI follow-up questions", id="subtitle"),
            Button("1. Start", id="start", variant="success"),
            Button("2. Settings", id="settings", variant="primary"),
            Button("3. Exit", id="exit", variant="error"),
            Static(
                "During session controls: p, m, i [sec], x <speaker>, k, g, s, q",
                id="hints",
            ),
            id="menu",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        app = self.app
        if not isinstance(app, PowerBATui):
            return

        if event.button.id == "start":
            app.result_action = "start"
            app.exit()
        elif event.button.id == "settings":
            app.push_screen(SettingsScreen(_copy_config(app.current_config), app.config_path))
        elif event.button.id == "exit":
            app.result_action = "quit"
            app.exit()


class SettingsScreen(Screen[None]):
    BINDINGS = [("escape", "back", "Back")]

    def __init__(self, config: AppConfig, config_path: Path) -> None:
        super().__init__()
        self._working = config
        self._config_path = config_path

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Label("Settings", id="settings_title"),
            Input(value=self._working.provider, id="provider", placeholder="provider: openai/anthropic"),
            Input(value=self._working.model, id="model", placeholder="model name"),
            Input(
                value=self._working.openai_api_key,
                id="openai_api_key",
                placeholder="OpenAI API key",
                password=True,
            ),
            Input(
                value=self._working.anthropic_api_key,
                id="anthropic_api_key",
                placeholder="Anthropic API key",
                password=True,
            ),
            Input(
                value=str(self._working.question_interval_seconds),
                id="question_interval_seconds",
                placeholder="question interval seconds",
            ),
            Static("Auto interval"),
            Switch(value=self._working.question_interval_enabled, id="question_interval_enabled"),
            Input(value=self._working.main_prompt, id="main_prompt", placeholder="main role prompt"),
            Input(value=self._working.mic_source, id="mic_source", placeholder="mic source"),
            Input(value=self._working.monitor_source, id="monitor_source", placeholder="monitor source"),
            Input(value=self._working.stt_backend, id="stt_backend", placeholder="stt backend: vosk/whisper_cpp"),
            Input(value=self._working.vosk_model_path, id="vosk_model_path", placeholder="vosk model path"),
            Input(
                value=self._working.whisper_cpp_model_path,
                id="whisper_cpp_model_path",
                placeholder="whisper.cpp model path",
            ),
            Input(
                value=self._working.whisper_cpp_binary,
                id="whisper_cpp_binary",
                placeholder="whisper.cpp binary (default whisper-cli)",
            ),
            Input(
                value=str(self._working.whisper_cpp_chunk_seconds),
                id="whisper_cpp_chunk_seconds",
                placeholder="whisper.cpp chunk seconds",
            ),
            Input(
                value=self._working.whisper_cpp_language,
                id="whisper_cpp_language",
                placeholder="whisper.cpp language (pl/en)",
            ),
            Static("Mic listening"),
            Switch(value=self._working.mic_listening_enabled, id="mic_listening_enabled"),
            Static("Diarization enabled"),
            Switch(value=self._working.diarization_enabled, id="diarization_enabled"),
            Input(
                value=self._working.pyannote_hf_token,
                id="pyannote_hf_token",
                placeholder="pyannote HF token",
                password=True,
            ),
            Input(
                value=self._working.pyannote_model_name,
                id="pyannote_model_name",
                placeholder="pyannote model name",
            ),
            Input(
                value=str(self._working.diarization_interval_seconds),
                id="diarization_interval_seconds",
                placeholder="diarization run interval seconds",
            ),
            Input(
                value=str(self._working.diarization_max_buffer_seconds),
                id="diarization_max_buffer_seconds",
                placeholder="diarization max buffer seconds",
            ),
            Button("Save", id="save", variant="success"),
            Button("Back", id="back", variant="warning"),
            id="settings_form",
        )
        yield Footer()

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
            return

        if event.button.id != "save":
            return

        self._working.provider = self.query_one("#provider", Input).value.strip().lower()
        self._working.model = self.query_one("#model", Input).value.strip()
        self._working.openai_api_key = self.query_one("#openai_api_key", Input).value.strip()
        self._working.anthropic_api_key = self.query_one("#anthropic_api_key", Input).value.strip()
        self._working.main_prompt = self.query_one("#main_prompt", Input).value.strip()
        self._working.mic_source = self.query_one("#mic_source", Input).value.strip()
        self._working.monitor_source = self.query_one("#monitor_source", Input).value.strip()
        self._working.stt_backend = self.query_one("#stt_backend", Input).value.strip()
        self._working.vosk_model_path = self.query_one("#vosk_model_path", Input).value.strip()
        self._working.whisper_cpp_model_path = self.query_one("#whisper_cpp_model_path", Input).value.strip()
        self._working.whisper_cpp_binary = self.query_one("#whisper_cpp_binary", Input).value.strip()
        self._working.whisper_cpp_language = self.query_one("#whisper_cpp_language", Input).value.strip()
        self._working.pyannote_hf_token = self.query_one("#pyannote_hf_token", Input).value.strip()
        self._working.pyannote_model_name = self.query_one("#pyannote_model_name", Input).value.strip()

        self._working.mic_listening_enabled = self.query_one("#mic_listening_enabled", Switch).value
        self._working.diarization_enabled = self.query_one("#diarization_enabled", Switch).value
        self._working.question_interval_enabled = self.query_one("#question_interval_enabled", Switch).value

        self._working.question_interval_seconds = _safe_int(
            self.query_one("#question_interval_seconds", Input).value,
            self._working.question_interval_seconds,
        )
        self._working.whisper_cpp_chunk_seconds = _safe_int(
            self.query_one("#whisper_cpp_chunk_seconds", Input).value,
            self._working.whisper_cpp_chunk_seconds,
        )
        self._working.diarization_interval_seconds = _safe_int(
            self.query_one("#diarization_interval_seconds", Input).value,
            self._working.diarization_interval_seconds,
        )
        self._working.diarization_max_buffer_seconds = _safe_int(
            self.query_one("#diarization_max_buffer_seconds", Input).value,
            self._working.diarization_max_buffer_seconds,
        )

        self._working.sanitize()
        save_config(self._working, self._config_path)

        app = self.app
        if isinstance(app, PowerBATui):
            app.current_config = _copy_config(self._working)

        self.app.pop_screen()


class PowerBATui(App[None]):
    CSS = """
    Screen {
        align: center middle;
    }

    #menu, #settings_form {
        width: 88;
        max-height: 90%;
        padding: 1 2;
        border: round #4f7f6f;
        overflow-y: auto;
    }

    #title {
        text-style: bold;
        color: #9be5c2;
    }

    #subtitle {
        color: #cfd7d4;
        margin-bottom: 1;
    }

    Button {
        margin: 0 0 1 0;
    }
    """

    def __init__(self, config_path: Path) -> None:
        super().__init__()
        self.config_path = config_path
        self.current_config = load_config(config_path)
        self.result_action = "quit"

    def on_mount(self) -> None:
        self.push_screen(MainMenuScreen())


def _safe_int(value: str, fallback: int) -> int:
    raw = value.strip()
    if not raw:
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def launch_tui(config_path: Path) -> str:
    app = PowerBATui(config_path=config_path)
    app.run()
    return app.result_action
