# Halloween Voice Agent

Halloween demon voice prop: microphone input -> voice activity detection -> Whisper speech-to-text -> Ollama -> Piper text-to-speech -> spooky audio effects -> speakers.

## Setup

With `uv` (recommended):

```sh
uv venv && source .venv/bin/activate
uv pip install "pipecat-ai[silero,whisper,piper,local]" numpy pedalboard
```

With a plain virtual environment:

```sh
python -m venv .venv && source .venv/bin/activate
pip install "pipecat-ai[silero,whisper,piper,local]" numpy pedalboard
```

## System Requirements

- PortAudio, required to build PyAudio. On Debian or Ubuntu: `sudo apt install portaudio19-dev`.
- Ollama running locally with the configured model:

	```sh
	ollama serve
	ollama pull gemma3:4b
	```

## Run

```sh
python demon.py
```

Note: Piper (`piper-tts`) and Pedalboard are GPL-3.0 licensed.