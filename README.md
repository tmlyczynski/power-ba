# Power BA

Power BA is a Linux-first command line meeting assistant for developers and business analysts.

It can:
- capture system audio (`monitor`) and optionally microphone (`mic`)
- transcribe with local STT (`vosk` or `whisper.cpp`)
- generate follow-up questions every N seconds with OpenAI or Anthropic
- pause/resume in runtime
- ignore all remote audio for a period, or ignore selected speaker labels
- run in modern TUI (`textual`) or legacy menu mode

Audio source handling:
- automatically detects available monitor and microphone sources
- auto-selects a default source when one is available
- shows an interactive numbered list when multiple sources are available

## Flow

Default UI (`python3 -m power_ba.cli`) starts TUI.

Main options:
1. Start transcription + AI questions
2. Settings
3. Exit

Settings include:
- OpenAI API key / Anthropic API key
- provider/model switch
- STT backend switch (`vosk` / `whisper_cpp`)
- whisper.cpp model path and binary path
- mic listening on/off
- auto interval on/off (can disable periodic generation completely)
- question interval (default 30s, used when auto interval is on)
- main prompt (role prompt)
- mic and monitor source
- diarization on/off + pyannote token/model

Runtime controls during session:
- `h` show commands help with descriptions
- `p` pause/resume
- `m` mic listening on/off
- `i 30` ignore all remote audio for 30 seconds
- `x SPEAKER_00` ignore/unignore specific speaker label
- `k` list known and ignored speakers
- `g` force immediate AI question generation (without waiting for interval)
- `s` save context snapshot (if `--output` is set)
- `q` stop session

Controls help is printed at session start and reminded periodically during the session.

## Requirements (Linux)

- Python 3.10+
- `pactl` and `parec` (PulseAudio/PipeWire compatibility)
- local model for selected STT backend

Optional for diarization:
- `pyannote.audio` dependencies + Hugging Face token

Quick environment checks:
```bash
pactl list short sources
which parec
which whisper-cli
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not activate `.venv`, run commands with explicit interpreter:
```bash
.venv/bin/python -m power_ba.cli
```

## Global Install (Linux)

Use installer script (recommended):
```bash
./install.sh
```

Then run from anywhere:
```bash
power-bi
```

Alternative with pipx:
```bash
pipx install .
power-bi
```

To remove user installation:
```bash
./uninstall.sh
```

Optional diarization dependencies:
```bash
pip install -r requirements-diarization.txt
```

## Run

Default (TUI):
```bash
python3 -m power_ba.cli
```

Global command (after installer):
```bash
power-bi
```

If you get `ModuleNotFoundError` (for example missing `typer`), use:
```bash
source .venv/bin/activate
python3 -m power_ba.cli
```
or:
```bash
.venv/bin/python -m power_ba.cli
```

Explicit TUI:
```bash
python3 -m power_ba.cli tui
```

Legacy menu:
```bash
python3 -m power_ba.cli menu
```

Direct start with 30s question interval:
```bash
python3 -m power_ba.cli start --question-interval 30
```

Disable periodic generation and use only manual `g` trigger:
```bash
python3 -m power_ba.cli start --no-auto-interval
```

Dry-run (no real audio capture):
```bash
python3 -m power_ba.cli start --dry-run --question-interval 5 --max-runtime 15 --no-controls
```

List available audio sources:
```bash
python3 -m power_ba.cli list-sources
```

When you start session and monitor/mic is not configured:
- app auto-detects sources
- if many are available, it asks you to choose from a numbered list

Start with whisper.cpp backend:
```bash
python3 -m power_ba.cli start \
	--stt-backend whisper_cpp \
	--whisper-model-path /path/to/ggml-model.bin \
	--whisper-binary whisper-cli
```

## Notes

- Audio files are written only when `--output <dir>` is set.
- Without API keys, session still runs and AI output is disabled gracefully.
- Diarization falls back to noop mode when pyannote/token is unavailable.
- If `stt_backend=vosk` but `vosk_model_path` is empty, app automatically falls back to `whisper_cpp` when `whisper_cpp_model_path` is configured.

## Smoke Test

```bash
python3 smoke_test.py
```

This runs a short simulated session to validate runtime flow.

## Legal / Privacy

Use recording/transcription only with participant consent and in compliance with local law.
