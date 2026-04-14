# Power BA CLI

CLI assistant for meetings that can:
- capture mic + system audio (Meet/remote stream)
- transcribe speech with local STT (VOSK)
- generate follow-up questions every N seconds using OpenAI or Anthropic
- run with interactive controls: pause/resume, mic on/off, ignore remote audio window

## Main Flow

Interactive menu:
1. Start transcription + AI questions
2. Settings
3. Exit

Settings include:
- OpenAI API key
- Anthropic API key
- provider and model switch (for example `gpt-5-mini` <-> Anthropic model)
- mic listening on/off
- question interval (default 30s)
- main prompt
- mic source / monitor source
- VOSK model path

Runtime controls while session is running:
- `p` pause/resume processing
- `m` mic listening on/off
- `i 30` ignore remote audio for 30 seconds
- `s` save context snapshot (only if output dir is set)
- `q` stop session

## Requirements (Linux)

- Python 3.10+
- `pactl` and `parec` available (PulseAudio/PipeWire compatibility layer)
- local VOSK model directory

Quick checks:
```bash
pactl list short sources
which parec
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Open menu (default flow):
```bash
python3 -m power_ba.cli
```

Direct start command:
```bash
python3 -m power_ba.cli start --question-interval 30
```

Dry-run (no real audio capture):
```bash
python3 -m power_ba.cli start --dry-run --question-interval 5 --max-runtime 15 --no-controls
```

List audio sources:
```bash
python3 -m power_ba.cli list-sources
```

## Configure Audio Sources

In menu -> Settings choose `Mic source` and `Monitor source` from `pactl` list.
Monitor source is usually something like `xxx.monitor`.

## Notes

- Audio files are saved only when `--output <dir>` is provided.
- Without API key, AI generation is disabled gracefully (session still runs).
- `whisper_cpp` backend is planned but not implemented in this scaffold.

## Smoke Test

```bash
python3 smoke_test.py
```

This runs a 12-second simulated session and validates the runtime pipeline.

## Legal / Privacy

Use recording/transcription only with participant consent and in compliance with local law.
