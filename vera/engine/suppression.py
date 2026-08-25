"""
Suppression engine.

Tracks what has already been sent to prevent:
- duplicate sends (same suppression_key)
- spam (same message body)
- post-decline outreach
- excessive unanswered nudges
- same trigger repeated

Uses simulated `now` from /v1/tick, not wall-clock time.
"""
from __future__ import annotations

import hashlib
import threading
from datetime import datetime
from typing import Dict, Optional, Set


# Minimum gap between sends to same merchant (in simulated seconds)
MIN_MERCHANT_GAP_SEC = 3600  # 1 hour between different sends to same merchant
SUPPRESSION_DEFAULT_TTL_SEC = 7 * 24 * 3600  # 7 days default TTL


class SuppressionRecord:
    __slots__ = ("suppression_key", "merchant_id", "customer_id", "trigger_id",
                 "body_hash", "sent_at_sim", "expires_sim")

    def __init__(self, suppression_key: str, merchant_id: str, body: str,
                 sent_at_sim: float, ttl_sec: float,
                 customer_id: Optional[str] = None,
                 trigger_id: Optional[str] = None) -> None:
        self.suppression_key = suppression_key
        self.merchant_id = merchant_id
        self.customer_id = customer_id
        self.trigger_id = trigger_id
        self.body_hash = hashlib.md5(body.strip().encode()).hexdigest()
        self.sent_at_sim = sent_at_sim
        self.expires_sim = sent_at_sim + ttl_sec


class SuppressionEngine:
    """Deterministic suppression based on simulated time."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # suppression_key -> SuppressionRecord
        self._by_key: Dict[str, SuppressionRecord] = {}
        # merchant_id -> last sent sim timestamp
        self._merchant_last_sent: Dict[str, float] = {}
        # merchant_id -> set of sent body hashes
        self._merchant_body_hashes: Dict[str, Set[str]] = {}
        # declined/hostile merchant IDs (permanently suppressed)
        self._declined: Set[str] = set()
        # trigger_id -> sent sim timestamp
        self._triggers_sent: Dict[str, float] = {}
        # current simulated time (seconds since epoch)
        self._sim_now: float = datetime.utcnow().timestamp()

    def update_sim_time(self, now_iso: str) -> None:
        """Update the simulated time reference from /v1/tick now field."""
        try:
            dt = datetime.fromisoformat(now_iso.replace("Z", "+00:00"))
            self._sim_now = dt.timestamp()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Check methods
    # ------------------------------------------------------------------

    def is_suppressed(self, suppression_key: str) -> bool:
        """Return True if this suppression key is still active."""
        with self._lock:
            rec = self._by_key.get(suppression_key)
            if not rec:
                return False
            return self._sim_now < rec.expires_sim

    def is_declined(self, merchant_id: str) -> bool:
        with self._lock:
            return merchant_id in self._declined

    def is_duplicate_body(self, merchant_id: str, body: str) -> bool:
        h = hashlib.md5(body.strip().encode()).hexdigest()
        with self._lock:
            return h in self._merchant_body_hashes.get(merchant_id, set())

    def is_trigger_sent(self, trigger_id: str) -> bool:
        with self._lock:
            return trigger_id in self._triggers_sent

    def merchant_gap_ok(self, merchant_id: str, min_gap_sec: float = MIN_MERCHANT_GAP_SEC) -> bool:
        """True if enough simulated time has passed since last send to this merchant."""
        with self._lock:
            last = self._merchant_last_sent.get(merchant_id)
            if last is None:
                return True
            return (self._sim_now - last) >= min_gap_sec

    def check(self, suppression_key: str, merchant_id: str,
              trigger_id: Optional[str] = None,
              body: Optional[str] = None,
              skip_gap_check: bool = False) -> Optional[str]:
        """
        Returns None if send is allowed, or a string reason if suppressed.
        """
        if self.is_declined(merchant_id):
            return "merchant_declined"
        if self.is_suppressed(suppression_key):
            return f"key_suppressed:{suppression_key}"
        if trigger_id and self.is_trigger_sent(trigger_id):
            return f"trigger_already_sent:{trigger_id}"
        if body and self.is_duplicate_body(merchant_id, body):
            return "duplicate_body"
        if not skip_gap_check and not self.merchant_gap_ok(merchant_id):
            return "merchant_gap_too_short"
        return None

    # ------------------------------------------------------------------
    # Record methods
    # ------------------------------------------------------------------

    def record_send(self, suppression_key: str, merchant_id: str, body: str,
                    ttl_sec: float = SUPPRESSION_DEFAULT_TTL_SEC,
                    customer_id: Optional[str] = None,
                    trigger_id: Optional[str] = None) -> None:
        """Record that a message was sent."""
        with self._lock:
            rec = SuppressionRecord(
                suppression_key=suppression_key,
                merchant_id=merchant_id,
                body=body,
                sent_at_sim=self._sim_now,
                ttl_sec=ttl_sec,
                customer_id=customer_id,
                trigger_id=trigger_id,
            )
            self._by_key[suppression_key] = rec
            self._merchant_last_sent[merchant_id] = self._sim_now

            hashes = self._merchant_body_hashes.setdefault(merchant_id, set())
            hashes.add(rec.body_hash)

            if trigger_id:
                self._triggers_sent[trigger_id] = self._sim_now

    def record_decline(self, merchant_id: str) -> None:
        with self._lock:
            self._declined.add(merchant_id)

    def teardown(self) -> None:
        with self._lock:
            self._by_key.clear()
            self._merchant_last_sent.clear()
            self._merchant_body_hashes.clear()
            self._declined.clear()
            self._triggers_sent.clear()
