"""Ordered conversation logging to a Discord webhook."""

import asyncio
import os
from pathlib import Path

import aiohttp
from loguru import logger

MAX_DISCORD_CHARS = 2000
VISITOR_NAME = "Trick-or-Treater \N{GHOST}"
DEMON_NAME = "Demon Tree \N{JACK-O-LANTERN}"


def load_webhook_url() -> str | None:
    """Env var first, then the gitignored .env next to this file."""
    if url := os.environ.get("DISCORD_WEBHOOK_URL"):
        return url
    env_file = Path(__file__).with_name(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            key, _, value = line.removeprefix("export ").partition("=")
            if key.strip() == "DISCORD_WEBHOOK_URL":
                return value.strip().strip("\"'") or None
    return None


class DiscordLogger:
    def __init__(self, webhook_url: str | None):
        self._url = webhook_url
        self._queue: asyncio.Queue[dict] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    def start(self):
        if self.enabled and not self._task:
            self._task = asyncio.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None

    def log_user(self, text: str | None):
        self._enqueue(VISITOR_NAME, text)

    def log_bot(self, text: str | None):
        self._enqueue(DEMON_NAME, text)

    def _enqueue(self, username: str, text: str | None):
        if not self.enabled or not text or not text.strip():
            return
        # allowed_mentions empty so spoken text can't ping anyone.
        self._queue.put_nowait(
            {
                "username": username,
                "content": text.strip()[:MAX_DISCORD_CHARS],
                "allowed_mentions": {"parse": []},
            }
        )

    async def _run(self):
        async with aiohttp.ClientSession() as session:
            while True:
                payload = await self._queue.get()
                try:
                    async with session.post(
                        self._url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 429:
                            retry = float((await resp.json()).get("retry_after", 1))
                            await asyncio.sleep(retry)
                            self._queue.put_nowait(payload)
                        elif resp.status >= 400:
                            logger.warning(f"Discord webhook failed: HTTP {resp.status}")
                except Exception as e:
                    logger.warning(f"Discord webhook error: {e}")
