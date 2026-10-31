#!/usr/bin/env python3
"""Gemini Live Halloween voice prop with local microphone, speaker, and effects."""

import asyncio
import logging
import os
import threading
from pathlib import Path

import numpy as np
import pyaudio
from google import genai
from google.genai import types
from pedalboard import Chorus, Delay, Distortion, Pedalboard, Phaser, Reverb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("demon")

SYSTEM_PROMPT = (
    "You are a tall, spooky, mischievous Halloween tree with purple eyes who guards a house and greets trick-or-treaters. "
    "Be playful, welcoming, and spooky-funny, like a dramatic host at a haunted house. "
    "Speak in a very deep, low-pitched, rumbling, resonant, gravelly voice at a very slow pace, with a spooky British English accent. "
    "Use British pronunciation and vocabulary. "
    "React to what visitors say and give out pretend supernatural praise for brave answers. "
    "Keep it family-friendly: no genuinely scary threats, insults, or adult jokes. "
    "Use natural spoken language and usually give one to three short sentences per reply. "
    "Respond directly, sometimes adding a playful detail or a friendly question to keep the conversation going. "
    "If you don't understand, playfully ask them to repeat themselves."
)

GEMINI_MODEL = "gemini-3.8-live"
GEMINI_VOICE = "Algenib"
MIC_RATE = 16_000
SPEAKER_RATE = 24_000
CHANNELS = 1
CHUNK_SAMPLES = 1024
TAIL_MAX_SECS = 0.6
TAIL_LEVEL = 0.005
MUTE_MIC_WHILE_SPEAKING = True


def load_api_key() -> str | None:
    if api_key := os.environ.get("GOOGLE_API_KEY"):
        return api_key
    env_file = Path(__file__).with_name(".env")
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator and name.strip() == "GOOGLE_API_KEY":
            return value.strip().strip("\"'") or None
    return None


class VoiceEffects:
    def __init__(self):
        self._board = Pedalboard(
            [
                Chorus(rate_hz=0.3, depth=0.25, centre_delay_ms=9.0, feedback=0.12, mix=0.14),
                Phaser(rate_hz=0.18, depth=0.5, feedback=0.18, mix=0.08),
                Distortion(drive_db=2.0),
                Delay(delay_seconds=0.26, feedback=0.18, mix=0.07),
                Reverb(room_size=0.7, damping=0.5, wet_level=0.1, dry_level=0.9),
            ]
        )
        self._reset_pending = True

    def reset(self):
        self._board.reset()
        self._reset_pending = True

    def process(self, pcm: bytes) -> bytes:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if not audio.size:
            return b""
        processed = self._board.process(audio, SPEAKER_RATE, reset=self._reset_pending)
        self._reset_pending = False
        return self._to_pcm(processed)

    def finish_turn(self) -> list[bytes]:
        blocks = []
        quiet_blocks = 0
        max_blocks = max(1, int(SPEAKER_RATE * TAIL_MAX_SECS / CHUNK_SAMPLES))
        for _ in range(max_blocks):
            silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            processed = self._board.process(silence, SPEAKER_RATE, reset=False)
            if not processed.size:
                continue
            if np.max(np.abs(processed)) < TAIL_LEVEL:
                quiet_blocks += 1
                if quiet_blocks >= 2:
                    break
            else:
                quiet_blocks = 0
                blocks.append(self._to_pcm(processed))
        self.reset()
        return blocks

    @staticmethod
    def _to_pcm(audio: np.ndarray) -> bytes:
        if not audio.size:
            return b""
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


async def send_microphone(
    session, mic_queue: asyncio.Queue[bytes], speaking: threading.Event
):
    while True:
        pcm = await mic_queue.get()
        if MUTE_MIC_WHILE_SPEAKING and speaking.is_set():
            pcm = bytes(len(pcm))
        await session.send_realtime_input(
            audio=types.Blob(data=pcm, mime_type=f"audio/pcm;rate={MIC_RATE}")
        )


async def play_audio(
    output_queue: asyncio.Queue[bytes | None], speaker, speaking: threading.Event
):
    while True:
        pcm = await output_queue.get()
        if pcm is None:
            speaking.clear()
            continue
        await asyncio.to_thread(speaker.write, pcm, exception_on_underflow=False)


def clear_queue(audio_queue: asyncio.Queue[bytes]):
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            break


async def receive_responses(
    session,
    output_queue: asyncio.Queue[bytes | None],
    effects: VoiceEffects,
    speaking: threading.Event,
):
    while True:
        async for response in session.receive():
            content = response.server_content
            if not content:
                continue
            if content.input_transcription and content.input_transcription.text:
                logger.info("Visitor: %s", content.input_transcription.text)
            if content.output_transcription and content.output_transcription.text:
                logger.info("Demon: %s", content.output_transcription.text)
            if content.interrupted:
                clear_queue(output_queue)
                effects.reset()
                speaking.clear()
            if content.model_turn:
                for part in content.model_turn.parts or []:
                    if part.inline_data and part.inline_data.data:
                        speaking.set()
                        pcm = effects.process(part.inline_data.data)
                        if pcm:
                            await output_queue.put(pcm)
            if content.turn_complete:
                for pcm in effects.finish_turn():
                    await output_queue.put(pcm)
                await output_queue.put(None)


async def run_live(api_key: str):
    client = genai.Client(api_key=api_key)
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_VOICE)
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
    )
    audio = pyaudio.PyAudio()
    mic = None
    speaker = None
    try:
        async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
            loop = asyncio.get_running_loop()
            mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=16)
            output_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=32)
            effects = VoiceEffects()
            speaking = threading.Event()

            def enqueue_mic(pcm: bytes):
                if mic_queue.full():
                    try:
                        mic_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                mic_queue.put_nowait(pcm)

            def mic_callback(in_data, _frame_count, _time_info, _status):
                if in_data:
                    if MUTE_MIC_WHILE_SPEAKING and speaking.is_set():
                        in_data = bytes(len(in_data))
                    loop.call_soon_threadsafe(enqueue_mic, in_data)
                return None, pyaudio.paContinue

            mic = audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=MIC_RATE,
                input=True,
                frames_per_buffer=CHUNK_SAMPLES,
                stream_callback=mic_callback,
                start=False,
            )
            speaker = audio.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SPEAKER_RATE,
                output=True,
                frames_per_buffer=CHUNK_SAMPLES,
                start=False,
            )
            mic.start_stream()
            speaker.start_stream()
            logger.info("The demon awakens... speak, mortal. Press Ctrl+C to stop.")
            sender = asyncio.create_task(send_microphone(session, mic_queue, speaking))
            player = asyncio.create_task(play_audio(output_queue, speaker, speaking))
            try:
                await receive_responses(session, output_queue, effects, speaking)
            finally:
                sender.cancel()
                player.cancel()
                await asyncio.gather(sender, player, return_exceptions=True)
    finally:
        if mic is not None:
            if mic.is_active():
                mic.stop_stream()
            mic.close()
        if speaker is not None:
            if speaker.is_active():
                speaker.stop_stream()
            speaker.close()
        audio.terminate()


def main():
    api_key = load_api_key()
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY is not set (environment or .env).")
    try:
        asyncio.run(run_live(api_key))
    except KeyboardInterrupt:
        logger.info("The demon returns to its tree.")


if __name__ == "__main__":
    main()
