#!/usr/bin/env python3
"""Halloween demon voice prop: mic -> VAD -> Whisper STT -> Ollama -> Piper TTS -> spooky DSP -> speakers."""

import asyncio

import numpy as np
from openai import AsyncOpenAI
from pedalboard import Chorus, Delay, Distortion, Pedalboard, Phaser, PitchShift, Reverb
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSSpeakFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
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
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter
from pipecat.turns.user_mute import AlwaysUserMuteStrategy
from pipecat.workers.runner import WorkerRunner

from discord_log import DiscordLogger, load_webhook_url

SYSTEM_PROMPT = (
    "You are a tall, spooky, mischievous Halloween tree with purple eyes who guards a house and greets trick-or-treaters. "
    "Be playful, welcoming, and spooky-funny, like a dramatic host at a haunted house. "
    "React to what visitors say and give out pretend supernatural praise for brave answers. "
    "Keep it family-friendly: no genuinely scary threats, insults, or adult jokes. "
    "Use natural spoken language and usually give two to four short sentences per reply. "
    "Respond directly, sometimes adding a playful detail or a friendly question to keep the conversation going. "
    "Output only the words you want spoken aloud. NEVER include stage directions, action nor emotion cues, "
    "sound-effect text, nor asides in parentheses, brackets, or asterisks. "
    "If you don't understand, playfully ask them to repeat themselves."
)

OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_KEEP_ALIVE = "1h"
WHISPER_MODEL = "small.en"
PIPER_VOICE = "en_GB-alan-medium"

# Keep detection responsive to quieter and higher-pitched voices.
VAD_CONFIDENCE = 0.40
# Ignore microphone input while the demon is speaking.
MUTE_USER_WHILE_SPEAKING = True
# Seconds of silence before the visitor's turn is considered finished.
VAD_STOP_SECS = 0.2
# Minimum measured audio volume required for speech detection.
VAD_MIN_VOLUME = 0.10
# Hard upper bound; the system prompt controls the usual reply length.
MAX_REPLY_TOKENS = 60
# Seconds after the last visitor turn before conversation history is cleared.
CONTEXT_RESET_SECS = 60
# Set to a number of silent seconds to enable an idle callout; None disables it.
IDLE_ATTRACT_SECS: int | None = 180
# Seconds of silence to insert between TTS sentences.
SENTENCE_PAUSE_SECS = 0
# Slow down the TTS voice to make it more spooky and otherworldly.
VOICE_SLOWDOWN = 1.12
# Max seconds of reverb/delay tail appended after speech; the visitor stays muted during it.
TAIL_MAX_SECS = 1.0
# Tail samples quieter than this (0-1 scale) are cut off.
TAIL_AUDIBLE_LEVEL = 0.005
# Webhook comes from DISCORD_WEBHOOK_URL or .env; unset disables Discord logging.
DISCORD_WEBHOOK_URL = load_webhook_url()


class SpookyVoiceFilter(FrameProcessor):
    """Turn-buffered Pedalboard effects that deepen and haunt the TTS voice."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._effects = Pedalboard(
            [
                PitchShift(semitones=-3.5),
                Chorus(
                    rate_hz=0.3,
                    depth=0.25,
                    centre_delay_ms=9.0,
                    feedback=0.1,
                    mix=0.22,
                ),
                Phaser(rate_hz=0.18, depth=0.55, feedback=0.15, mix=0.12),
                Distortion(drive_db=3.0),
                Delay(delay_seconds=0.26, feedback=0.2, mix=0.12),
                Reverb(room_size=0.75, damping=0.5, wet_level=0.12, dry_level=0.88),
            ]
        )
        self._audio_buffer = bytearray()
        self._emitted_sentence = False
        self._sample_rate: int | None = None
        self._reset()

    def _reset(self):
        self._effects.reset()
        self._audio_buffer.clear()
        self._emitted_sentence = False
        self._sample_rate = None

    def _process_sentence(self, rate: int) -> bytes:
        if not self._audio_buffer:
            return b""
        audio = np.frombuffer(self._audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
        if self._emitted_sentence:
            audio = np.concatenate([np.zeros(int(rate * SENTENCE_PAUSE_SECS), dtype=np.float32), audio])
        self._emitted_sentence = True
        # reset=True compensates PitchShift latency; reset=False would hold back the end of each sentence.
        processed = self._effects.process(audio, rate, reset=True)
        if processed.size:
            slowed_size = int(processed.size * VOICE_SLOWDOWN)
            processed = np.interp(
                np.linspace(0, processed.size - 1, slowed_size),
                np.arange(processed.size),
                processed,
            )
        self._audio_buffer.clear()
        # Let the reverb decay inside the chunk so the next one doesn't start on a hard cut.
        processed = np.concatenate([processed, self._flush_effects(rate)])
        return self._to_pcm(processed)

    @staticmethod
    def _to_pcm(audio: np.ndarray) -> bytes:
        if not audio.size:
            return b""
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    def _flush_effects(self, rate: int) -> np.ndarray:
        block_size = 1024
        silence_blocks = 0
        output_blocks = []
        max_blocks = max(1, int(rate * TAIL_MAX_SECS / block_size))

        for _ in range(max_blocks):
            silence = np.zeros(block_size, dtype=np.float32)
            processed = self._effects.process(silence, rate, reset=False)
            if not processed.size:
                continue
            output_blocks.append(processed)
            if np.max(np.abs(processed)) < TAIL_AUDIBLE_LEVEL:
                silence_blocks += 1
                if silence_blocks >= 2:
                    break
            else:
                silence_blocks = 0

        if not output_blocks:
            return np.zeros(0, dtype=np.float32)
        tail = np.concatenate(output_blocks)
        audible_samples = np.flatnonzero(np.abs(tail) > TAIL_AUDIBLE_LEVEL)
        if not audible_samples.size:
            return np.zeros(0, dtype=np.float32)
        return tail[: audible_samples[-1] + int(rate * 0.02)]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and frame.num_channels == 1:
            self._sample_rate = frame.sample_rate
            self._audio_buffer.extend(frame.audio)
            return
        if isinstance(frame, TTSTextFrame):
            # Piper pushes the text frame after the sentence's audio, so it marks a sentence boundary.
            if frame.will_be_spoken and self._sample_rate is not None:
                # Off the event loop: blocking it starves the audio writer and causes pops mid-playback.
                speech = await asyncio.to_thread(self._process_sentence, self._sample_rate)
                if speech:
                    await self.push_frame(
                        TTSAudioRawFrame(
                            audio=speech,
                            sample_rate=self._sample_rate,
                            num_channels=1,
                            context_id=frame.context_id,
                        ),
                        direction,
                    )
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, TTSStoppedFrame):
            if self._sample_rate is not None:
                speech = await asyncio.to_thread(self._process_sentence, self._sample_rate)
                if speech:
                    await self.push_frame(
                        TTSAudioRawFrame(
                            audio=speech,
                            sample_rate=self._sample_rate,
                            num_channels=1,
                            context_id=frame.context_id,
                        ),
                        direction,
                    )
            self._reset()
        await self.push_frame(frame, direction)


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(model=WHISPER_MODEL),
        think=False,
        device="auto",
        compute_type="int8",
        ttfs_p99_latency=1.0,
    )
    llm = OLLamaLLMService(
        base_url=OLLAMA_URL,
        settings=OLLamaLLMService.Settings(
            model=OLLAMA_MODEL,
            system_instruction=SYSTEM_PROMPT,
            max_tokens=MAX_REPLY_TOKENS,
            temperature=0.9,
            extra={"extra_body": {"keep_alive": OLLAMA_KEEP_ALIVE}},
        ),
    )
    attract_client = AsyncOpenAI(base_url=OLLAMA_URL, api_key="ollama")
    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=PIPER_VOICE),
        text_filters=[MarkdownTextFilter()],
    )

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_mute_strategies=(
                [AlwaysUserMuteStrategy()] if MUTE_USER_WHILE_SPEAKING else []
            ),
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    confidence=VAD_CONFIDENCE,
                    stop_secs=VAD_STOP_SECS,
                    min_volume=VAD_MIN_VOLUME,
                )
            ),
        ),
    )

    user_agg = aggregators.user()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_agg,
            llm,
            tts,  # sentence-streams LLM tokens into audio
            SpookyVoiceFilter(),
            transport.output(),
            aggregators.assistant(),
        ]
    )

    # Forget the previous visitor after sustained silence.
    reset_task: asyncio.Task | None = None
    attract_task: asyncio.Task | None = None

    async def reset_after_silence():
        await asyncio.sleep(CONTEXT_RESET_SECS)
        context.set_messages([])
        logger.info("Silence: conversation context cleared.")

    async def attract_after_silence():
        if IDLE_ATTRACT_SECS is None:
            return
        while True:
            await asyncio.sleep(IDLE_ATTRACT_SECS)
            try:
                response = await attract_client.chat.completions.create(
                    model=OLLAMA_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a playful, spooky Halloween tree welcoming trick-or-treaters. "
                                "Be family-friendly and inviting, with no threats. Output only spoken words."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Invent a fresh, one-sentence callout that invites nearby trick-or-treaters "
                                "to come closer to talk or get Halloween treats. Keep it under 16 words."
                            ),
                        },
                    ],
                    max_tokens=40,
                    temperature=0.9,
                    extra_body={"keep_alive": OLLAMA_KEEP_ALIVE},
                )
                line = response.choices[0].message.content
                if not line or not line.strip():
                    logger.warning("Ollama generated an empty idle callout.")
                    continue
                await tts.queue_frame(TTSSpeakFrame(line.strip(), append_to_context=False))
                logger.info("Silence: AI-generated idle callout queued.")
            except Exception:
                logger.exception("Could not generate idle callout.")

    def schedule_attraction():
        nonlocal attract_task
        if IDLE_ATTRACT_SECS is not None:
            if attract_task:
                attract_task.cancel()
            attract_task = asyncio.create_task(attract_after_silence())

    @user_agg.event_handler("on_user_turn_started")
    async def _on_started(*_):
        nonlocal reset_task
        if reset_task:
            reset_task.cancel()
        if attract_task:
            attract_task.cancel()

    discord = DiscordLogger(DISCORD_WEBHOOK_URL)
    discord.start()
    logger.info(f"Discord logging {'enabled' if discord.enabled else 'disabled (DISCORD_WEBHOOK_URL not set)'}")

    @aggregators.assistant().event_handler("on_assistant_turn_stopped")
    async def _on_bot_stopped(_agg, message):
        logger.debug(f"Discord log bot: {message.content!r}")
        discord.log_bot(message.content)

    @user_agg.event_handler("on_user_turn_stopped")
    async def _on_stopped(_agg, _strategy, message):
        logger.debug(f"Discord log user: {message.content!r}")
        discord.log_user(message.content)
        nonlocal reset_task
        if reset_task:
            reset_task.cancel()
        reset_task = asyncio.create_task(reset_after_silence())
        schedule_attraction()

    worker = PipelineWorker(pipeline, idle_timeout_secs=None)
    runner = WorkerRunner()
    await runner.add_workers(worker)
    schedule_attraction()
    logger.info("The demon awakens... speak, mortal.")
    try:
        await runner.run()
    finally:
        await attract_client.close()


if __name__ == "__main__":
    asyncio.run(main())
