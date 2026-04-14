from __future__ import annotations

import queue
import threading
from dataclasses import asdict
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static, Switch

from .capture import list_mic_sources, list_monitor_sources
from .config import AppConfig, load_config, save_config
from .runtime import run_session


def _copy_config(config: AppConfig) -> AppConfig:
    return AppConfig.from_dict(asdict(config))


SESSION_COMMANDS_TEXT = (
    "Komendy sesji (wpisz w dolnym polu i Enter):\n"
    "h        pomoc\n"
    "p        pauza/wznowienie\n"
    "m        mikrofon on/off\n"
    "i 30     ignoruj zdalny glos przez 30s\n"
    "x ID     ignoruj/odblokuj mowce (np. x SPEAKER_00)\n"
    "k        pokaz znanych i ignorowanych mowcow\n"
    "g        wygeneruj pytania AI teraz\n"
    "a tekst  doprecyzuj/wyslij nowe zapytanie do AI\n"
    "lang en  zmien jezyk AI (pl/en)\n"
    "style t  ustaw styl odpowiedzi AI na cala sesje\n"
    "s        zapisz snapshot (gdy ustawiono output)\n"
    "q        zatrzymaj sesje"
)


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
                "During session controls: h, p, m, i [sec], x <speaker>, k, g, a <text>, lang <pl|en>, style <text>, s, q",
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
            app.push_screen(SessionScreen(_copy_config(app.current_config)))
        elif event.button.id == "settings":
            app.push_screen(SettingsScreen(_copy_config(app.current_config), app.config_path))
        elif event.button.id == "exit":
            app.result_action = "quit"
            app.exit()


class SessionScreen(Screen[None]):
    BINDINGS = [
        ("escape", "back", "Back"),
    ]

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._commands: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._running = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Vertical(
            Label("Session Running", id="session_title"),
            Static(SESSION_COMMANDS_TEXT, id="session_commands"),
            RichLog(id="session_log", wrap=True, markup=False, highlight=False),
            Input(
                placeholder="Wpisz komende (g, a doprecyzuj..., lang en, style tylko ryzyka, q)",
                id="session_input",
            ),
            id="session_view",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._running = True
        self._write_log("Uruchamianie sesji...")

        self._worker = threading.Thread(target=self._session_worker, daemon=True)
        self._worker.start()

    def action_back(self) -> None:
        if self._running:
            self._commands.put("q")
            self._write_log("Wyslano komende stop (q). Poczekaj na zakonczenie sesji.")
            return
        self.app.pop_screen()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        raw = event.value.strip()
        event.input.value = ""
        if not raw:
            return

        self._write_log(f"> {raw}")
        self._commands.put(raw)

    def _session_worker(self) -> None:
        try:
            run_session(
                config=self._config,
                interactive_controls=True,
                show_controls_help=False,
                command_queue=self._commands,
                event_callback=self._emit_from_thread,
            )
        except Exception as exc:  # pragma: no cover
            self._emit_from_thread(f"Session failed: {exc}")
        finally:
            self._running = False
            self._emit_from_thread("Sesja zakonczona. ESC aby wrocic do menu.")

    def _emit_from_thread(self, message: str) -> None:
        self.app.call_from_thread(self._write_log, message)

    def _write_log(self, message: str) -> None:
        log_widget = self.query_one("#session_log", RichLog)
        for line in message.splitlines() or [""]:
            log_widget.write(line)


class SourcePickerScreen(Screen[str | None]):
    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        kind_label: str,
        sources: list[str],
        current: str,
        allow_disable: bool,
    ) -> None:
        super().__init__()
        self._kind_label = kind_label
        self._sources = sources
        self._current = current
        self._allow_disable = allow_disable

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        rows = [
            Label(f"Select {self._kind_label} source", id="picker_title"),
        ]

        if not self._sources:
            rows.append(Static("No sources detected.", id="picker_empty"))
        else:
            for index, source in enumerate(self._sources, start=1):
                marker = " (current)" if source == self._current else ""
                rows.append(Button(f"{index}. {source}{marker}", id=f"pick_{index}"))

        if self._allow_disable:
            rows.append(Button("Disable source", id="disable_source", variant="warning"))

        rows.append(Button("Cancel", id="cancel_picker", variant="error"))

        yield Vertical(*rows, id="source_picker")
        yield Footer()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "cancel_picker":
            self.dismiss(None)
            return

        if button_id == "disable_source":
            self.dismiss("")
            return

        if button_id.startswith("pick_"):
            try:
                idx = int(button_id.split("_", 1)[1]) - 1
            except ValueError:
                self.dismiss(None)
                return

            if 0 <= idx < len(self._sources):
                self.dismiss(self._sources[idx])
                return

        self.dismiss(None)


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
            Input(value=self._working.ai_language, id="ai_language", placeholder="AI language (pl/en)"),
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
            Static("Monitor source", id="monitor_source_label"),
            Static(self._working.monitor_source or "(not set)", id="monitor_source_value"),
            Button("Select monitor source", id="pick_monitor_source", variant="primary"),
            Static("Microphone source", id="mic_source_label"),
            Static(self._working.mic_source or "(not set)", id="mic_source_value"),
            Button("Select microphone source", id="pick_mic_source", variant="primary"),
            Input(
                value=self._working.default_output_dir,
                id="default_output_dir",
                placeholder="output directory for saved files",
            ),
            Static("Save audio files"),
            Switch(value=self._working.save_audio_by_default, id="save_audio_by_default"),
            Static("Save transcript/AI logs"),
            Switch(value=self._working.save_transcript_by_default, id="save_transcript_by_default"),
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

    def on_mount(self) -> None:
        self._refresh_source_labels()

    def action_back(self) -> None:
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
            return

        if event.button.id == "pick_monitor_source":
            sources = list_monitor_sources()
            self.app.push_screen(
                SourcePickerScreen(
                    kind_label="monitor",
                    sources=sources,
                    current=self._working.monitor_source,
                    allow_disable=False,
                ),
                self._on_monitor_source_picked,
            )
            return

        if event.button.id == "pick_mic_source":
            sources = list_mic_sources()
            self.app.push_screen(
                SourcePickerScreen(
                    kind_label="microphone",
                    sources=sources,
                    current=self._working.mic_source,
                    allow_disable=True,
                ),
                self._on_mic_source_picked,
            )
            return

        if event.button.id != "save":
            return

        self._working.provider = self.query_one("#provider", Input).value.strip().lower()
        self._working.model = self.query_one("#model", Input).value.strip()
        self._working.ai_language = self.query_one("#ai_language", Input).value.strip().lower()
        self._working.openai_api_key = self.query_one("#openai_api_key", Input).value.strip()
        self._working.anthropic_api_key = self.query_one("#anthropic_api_key", Input).value.strip()
        self._working.main_prompt = self.query_one("#main_prompt", Input).value.strip()
        self._working.default_output_dir = self.query_one("#default_output_dir", Input).value.strip()
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
        self._working.save_audio_by_default = self.query_one("#save_audio_by_default", Switch).value
        self._working.save_transcript_by_default = self.query_one("#save_transcript_by_default", Switch).value

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

    def _on_monitor_source_picked(self, selected: str | None) -> None:
        if selected is None:
            return
        self._working.monitor_source = selected
        self._refresh_source_labels()

    def _on_mic_source_picked(self, selected: str | None) -> None:
        if selected is None:
            return
        self._working.mic_source = selected
        self._refresh_source_labels()

    def _refresh_source_labels(self) -> None:
        self.query_one("#monitor_source_value", Static).update(self._working.monitor_source or "(not set)")
        self.query_one("#mic_source_value", Static).update(self._working.mic_source or "(not set)")


class PowerBATui(App[None]):
    CSS = """
    Screen {
        align: center middle;
    }

    #menu, #settings_form, #session_view {
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

    #session_title {
        text-style: bold;
        color: #9be5c2;
        margin-bottom: 1;
    }

    #session_commands {
        color: #d6e2dd;
        margin-bottom: 1;
    }

    #session_log {
        height: 1fr;
        border: round #3f5f56;
        margin-bottom: 1;
    }

    #session_input {
        dock: bottom;
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
