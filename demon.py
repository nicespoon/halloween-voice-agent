#!/usr/bin/env python3
"""Halloween demon voice prop: mic -> VAD -> Whisper STT -> Ollama -> Piper TTS -> spooky DSP -> speakers.

Setup (uv, recommended):
    uv venv && source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
    uv pip install "pipecat-ai[silero,whisper,piper,local]" numpy

Setup (plain venv):
    python -m venv .venv && source .venv/bin/activate
    pip install "pipecat-ai[silero,whisper,piper,local]" numpy

System prerequisites:
    - PortAudio (needed to build PyAudio):  sudo apt install portaudio19-dev   |   brew install portaudio
    - Ollama running locally with a small, fast model:
          ollama serve &
          ollama pull llama3.2:1b

Run:
    python demon.py

Optional environment variables:
    OLLAMA_MODEL (default llama3.2:1b), OLLAMA_URL (default http://localhost:11434/v1),
    WHISPER_MODEL (default tiny.en), PIPER_VOICE (default en_US-ryan-high).
    Note: Piper (piper-tts) is GPL-3.0 licensed.
"""

import asyncio
import os

import numpy as np
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

SYSTEM_PROMPT = (
    "You are an ancient, mischievous Halloween demon guarding a driveway. "
    "Speak in short, spooky, witty sentences. Keep responses under 2 sentences."
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny.en")
PIPER_VOICE = os.getenv("PIPER_VOICE", "en_US-ryan-high")

# Seconds of silence before the visitor's turn is considered finished.
VAD_STOP_SECS = 0.3
# Cap reply length so generation (and speech) stays short and quick.
MAX_REPLY_TOKENS = 60


class SpookyVoiceFilter(FrameProcessor):
    """Streaming DSP that turns TTS audio into a demonic voice.

    Chain: pitch/speed drop (streaming linear resampler) -> ring modulation
    (growl) -> feedback echo (cavernous tail) -> soft clip. State is carried
    across chunks so there are no clicks at chunk boundaries.
    """

    def __init__(
        self,
        pitch: float = 0.78,
        ring_hz: float = 40.0,
        ring_mix: float = 0.35,
        echo_ms: float = 140.0,
        echo_feedback: float = 0.35,
        echo_mix: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pitch = pitch  # <1.0 = lower and slower
        self._ring_hz = ring_hz
        self._ring_mix = ring_mix
        self._echo_ms = echo_ms
        self._echo_feedback = echo_feedback
        self._echo_mix = echo_mix
        self._reset()

    def _reset(self):
        self._tail = np.zeros(0, dtype=np.float32)  # unread input samples
        self._pos = 0.0  # fractional read position within self._tail
        self._phase = 0
        self._echo_buf = None

    def _process_audio(self, pcm: bytes, rate: int) -> bytes:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        buf = np.concatenate([self._tail, x])

        # Pitch down: read input at `pitch` x speed using linear interpolation.
        n_out = int(np.floor((len(buf) - 1 - self._pos) / self._pitch)) + 1
        if n_out <= 0:
            self._tail = buf
            return b""
        idx = self._pos + np.arange(n_out) * self._pitch
        i0 = idx.astype(np.int64)
        frac = (idx - i0).astype(np.float32)
        y = buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac
        next_pos = self._pos + n_out * self._pitch
        keep = int(next_pos)
        self._tail = buf[keep:]
        self._pos = next_pos - keep

        # Ring modulation, phase-continuous.
        t = (self._phase + np.arange(len(y))) / rate
        self._phase += len(y)
        carrier = np.sin(2 * np.pi * self._ring_hz * t).astype(np.float32)
        y = y * (1.0 - self._ring_mix) + y * carrier * self._ring_mix

        # Feedback echo with a delay line that persists between chunks.
        delay = max(1, int(rate * self._echo_ms / 1000))
        if self._echo_buf is None:
            self._echo_buf = np.zeros(delay, dtype=np.float32)
        wet = np.empty_like(y)
        buf_e = self._echo_buf
        for start in range(0, len(y), delay):
            seg = y[start : start + delay]
            echoed = seg + buf_e[: len(seg)] * self._echo_feedback
            wet[start : start + len(seg)] = echoed
            buf_e = np.concatenate([buf_e[len(seg) :], echoed])
        self._echo_buf = buf_e
        y = y * (1.0 - self._echo_mix) + wet * self._echo_mix

        y = np.tanh(y * 1.4)  # soft clip for grit
        return (y * 32767.0).astype(np.int16).tobytes()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and frame.num_channels == 1:
            out = self._process_audio(frame.audio, frame.sample_rate)
            if out:
                frame.audio = out
                frame.num_frames = len(out) // 2
                await self.push_frame(frame, direction)
            return
        if isinstance(frame, TTSStoppedFrame):
            self._reset()
        await self.push_frame(frame, direction)


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    stt = WhisperSTTService(model=WHISPER_MODEL, device="auto", compute_type="int8")
    llm = OLLamaLLMService(
        base_url=OLLAMA_URL,
        settings=OLLamaLLMService.Settings(
            model=OLLAMA_MODEL,
            system_instruction=SYSTEM_PROMPT,
            max_tokens=MAX_REPLY_TOKENS,
            temperature=0.9,
        ),
    )
    tts = PiperTTSService(settings=PiperTTSService.Settings(voice=PIPER_VOICE))

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,  # sentence-streams LLM tokens into audio
            SpookyVoiceFilter(),
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(pipeline)
    logger.info("The demon awakens... speak, mortal.")
    await PipelineRunner().run(task)


if __name__ == "__main__":
    asyncio.run(main())
