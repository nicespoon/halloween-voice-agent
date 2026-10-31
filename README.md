# Halloween Voice Agent

Halloween demon voice prop: microphone input -> Gemini Live (speech to speech) -> spooky audio effects -> speakers.

## Setup

With `uv` (recommended):

```sh
uv venv && source .venv/bin/activate
uv pip install google-genai pyaudio numpy pedalboard
```

With a plain virtual environment:

```sh
python -m venv .venv && source .venv/bin/activate
pip install google-genai pyaudio numpy pedalboard
```

## System Requirements

- PortAudio, required to build PyAudio. On Debian or Ubuntu: `sudo apt install portaudio19-dev`.
- A Gemini API key in `GOOGLE_API_KEY` (environment or a gitignored `.env`).

## Run

```sh
python demon.py
```

The microphone is muted while the demon speaks to prevent speaker feedback. Set `MUTE_MIC_WHILE_SPEAKING = False` in `demon.py` to allow interruptions when using headphones or echo cancellation.

Note: Pedalboard is GPL-3.0 licensed.