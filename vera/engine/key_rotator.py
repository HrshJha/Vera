"""
Vera Multi-Key LLM Pool & Automatic Key Rotator.
================================================
Automatically loads all available Gemini & OpenRouter API keys from environment
and .env variables, distributes traffic via round-robin, and instantly fails over
to healthy keys whenever a 429 (Too Many Requests / Quota Exceeded) occurs.

Supported .env formats:
  - GEMINI_API_KEY=key1
  - GEMINI_API_KEY_1=key2
  - GEMINI_API_KEY_2=key3
  - GEMINI_API_KEYS=key1,key2,key3,key4
  - OPENROUTER_API_KEY=sk-...
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class KeyState:
    """Tracks runtime health and rate limits for an individual API key."""

    def __init__(self, key: str, index: int, provider: str = "gemini"):
        self.key = key.strip()
        self.index = index
        self.provider = provider
        self.cooldown_until: float = 0.0
        self.total_successes: int = 0
        self.total_429s: int = 0
        self.last_used: float = 0.0

    @property
    def is_available(self) -> bool:
        return time.time() >= self.cooldown_until

    @property
    def masked_key(self) -> str:
        if len(self.key) <= 8:
            return "***"
        return f"{self.key[:6]}...{self.key[-4:]}"

    def mark_429(self, cooldown_seconds: float = 60.0) -> None:
        self.total_429s += 1
        self.cooldown_until = time.time() + max(cooldown_seconds, 15.0)
        logger.warning(
            f"[KeyPool] Key #{self.index} ({self.masked_key}) hit 429/Quota. "
            f"Cooling down for {cooldown_seconds:.1f}s until {time.strftime('%H:%M:%S', time.localtime(self.cooldown_until))}"
        )

    def mark_success(self) -> None:
        self.total_successes += 1
        self.last_used = time.time()


class APIKeyPool:
    """
    Thread-safe manager for multiple Gemini / OpenRouter API keys.
    Automatically rotates keys and switches on failure.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: List[KeyState] = []
        self._current_index = 0
        self._client_cache: Dict[str, Any] = {}
        self.load_keys()

    def load_keys(self) -> int:
        """Scan environment and .env file for all API keys."""
        from dotenv import load_dotenv
        load_dotenv(override=True)

        found_keys: List[str] = []

        # 1. Comma-separated GEMINI_API_KEYS
        raw_list = os.environ.get("GEMINI_API_KEYS", "")
        if raw_list:
            for k in raw_list.split(","):
                k = k.strip()
                if k and k not in found_keys:
                    found_keys.append(k)

        # 2. Numbered keys GEMINI_API_KEY, GEMINI_API_KEY_1..20
        direct_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if direct_key and direct_key not in found_keys:
            found_keys.append(direct_key)

        for i in range(1, 21):
            k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
            if k and k not in found_keys:
                found_keys.append(k)

        # 3. Check extra keys in .env
        env_path = os.path.join(os.getcwd(), ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "GEMINI_API_KEY" in line and "=" in line:
                            _, val = line.split("=", 1)
                            val = val.strip().strip("'\"")
                            if val and val not in found_keys and "EXHAUSTED" not in line:
                                found_keys.append(val)
            except Exception as e:
                logger.warning(f"Error scanning .env for keys: {e}")

        with self._lock:
            self._keys = [KeyState(key=k, index=idx + 1) for idx, k in enumerate(found_keys)]
            self._client_cache.clear()

        logger.info(f"[KeyPool] Loaded {len(self._keys)} active Gemini API key(s) in pool.")
        for k in self._keys:
            logger.info(f"  Key #{k.index}: {k.masked_key} [READY]")

        return len(self._keys)

    @property
    def total_keys(self) -> int:
        with self._lock:
            return len(self._keys)

    @property
    def available_keys_count(self) -> int:
        with self._lock:
            return sum(1 for k in self._keys if k.is_available)

    def get_client_and_key(self) -> Tuple[Optional[Any], Optional[KeyState]]:
        """
        Get the next available healthy Gemini client & key state.
        Cycles in round-robin order across available keys.
        """
        with self._lock:
            if not self._keys:
                return None, None

            # Try to find next available key starting from current_index
            total = len(self._keys)
            for offset in range(total):
                idx = (self._current_index + offset) % total
                candidate = self._keys[idx]
                if candidate.is_available:
                    self._current_index = (idx + 1) % total
                    client = self._get_or_create_client(candidate.key)
                    return client, candidate

            # If all are cooling down, pick the one that will cool down earliest
            earliest = min(self._keys, key=lambda k: k.cooldown_until)
            wait_needed = max(0.0, earliest.cooldown_until - time.time())
            logger.warning(
                f"[KeyPool] All {total} keys are cooling down. "
                f"Earliest key #{earliest.index} ready in {wait_needed:.1f}s."
            )
            client = self._get_or_create_client(earliest.key)
            return client, earliest

    def _get_or_create_client(self, api_key: str) -> Any:
        if api_key in self._client_cache:
            return self._client_cache[api_key]

        from google import genai
        client = genai.Client(api_key=api_key)
        self._client_cache[api_key] = client
        return client

    def get_status(self) -> List[Dict[str, Any]]:
        """Return operational telemetry of the key pool."""
        with self._lock:
            now = time.time()
            return [
                {
                    "index": k.index,
                    "key": k.masked_key,
                    "available": k.is_available,
                    "cooldown_remaining_sec": max(0, int(k.cooldown_until - now)),
                    "successes": k.total_successes,
                    "429s": k.total_429s,
                }
                for k in self._keys
            ]


# Singleton global pool
key_pool = APIKeyPool()
