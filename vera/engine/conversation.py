"""
Conversation state machine.

States: NEW → PITCHED → QUALIFYING → ENGAGED → ACTION_READY → WAITING →
        COMPLETED → DECLINED → AUTO_REPLY → ENDED

All transitions are deterministic. No LLM involvement in state decisions.
"""
from __future__ import annotations

import hashlib
import re
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class ConvState(str, Enum):
    NEW = "new"
    PITCHED = "pitched"
    QUALIFYING = "qualifying"
    ENGAGED = "engaged"
    ACTION_READY = "action_ready"
    WAITING = "waiting"
    COMPLETED = "completed"
    DECLINED = "declined"
    AUTO_REPLY = "auto_reply"
    ENDED = "ended"


# --- Intent classification patterns (deterministic, no LLM) ---

_ACCEPTANCE = re.compile(
    r"\b(yes|sure|ok\s*let['']?s?\s*do\s*it|go\s*ahead|do\s*it|proceed|haan|chalo|bilkul|"
    r"theek\s*hai|karein|karain|let['']?s\s*go|sounds?\s*good|great|perfect|"
    r"yes\s*please|please\s*do|i(?:'m|\s+am)\s*in|count\s*me\s*in)\b",
    re.IGNORECASE,
)

_DEFERRAL = re.compile(
    r"\b(later|tomorrow|next\s*week|some\s*other\s*time|not\s*now|remind\s*me|"
    r"baad\s*mein|kal|thodi\s*der|abhi\s*nahi|kuch\s*din)\b",
    re.IGNORECASE,
)

_DECLINE = re.compile(
    r"\b(no\s*thank[s]?|not\s*interested|stop|unsubscribe|nahi|nahin|band\s*karo|"
    r"don['']?t\s*contact|please\s*stop|leave\s*me|no\s*need|"
    r"not\s*required|hatao|mat\s*bhejo)\b",
    re.IGNORECASE,
)

_HOSTILE = re.compile(
    r"\b(spam|useless|waste\s*of\s*time|stop\s*messaging|harassment|report|"
    r"bakwas|band\s*karo\s*ye|annoying|irritating|fraud|scam)\b",
    re.IGNORECASE,
)

# Auto-reply indicators (common WA Business templates)
_AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "our team will respond",
    "automated",
    "aapki jaankari ke liye bahut-bahut shukriya",
    "will get back to you",
    "outside working hours",
    "business hours",
    "response time",
    "we have received your message",
    "humari team",
    "jald hi sampark karenge",
]


class Turn:
    __slots__ = ("from_role", "message", "ts", "msg_hash")

    def __init__(self, from_role: str, message: str, ts: Optional[str] = None) -> None:
        self.from_role = from_role
        self.message = message
        self.ts = ts or datetime.utcnow().isoformat() + "Z"
        self.msg_hash = hashlib.md5(message.strip().lower().encode()).hexdigest()


class Conversation:
    def __init__(self, conv_id: str, merchant_id: str,
                 customer_id: Optional[str] = None) -> None:
        self.conv_id = conv_id
        self.merchant_id = merchant_id
        self.customer_id = customer_id
        self.state = ConvState.NEW
        self.turns: List[Turn] = []
        self.trigger_id: Optional[str] = None
        self.created_at = datetime.utcnow().isoformat() + "Z"
        self.last_bot_body_hash: Optional[str] = None
        self.sent_bodies: List[str] = []
        # Track repeated incoming messages for auto-reply detection
        self._incoming_hash_counts: Dict[str, int] = {}

    @property
    def is_terminal(self) -> bool:
        return self.state in (ConvState.COMPLETED, ConvState.DECLINED, ConvState.ENDED)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def last_merchant_message(self) -> Optional[str]:
        for t in reversed(self.turns):
            if t.from_role in ("merchant", "customer"):
                return t.message
        return None

    def add_turn(self, from_role: str, message: str, ts: Optional[str] = None) -> None:
        turn = Turn(from_role, message, ts)
        self.turns.append(turn)

        if from_role in ("merchant", "customer"):
            h = turn.msg_hash
            self._incoming_hash_counts[h] = self._incoming_hash_counts.get(h, 0) + 1

    def add_bot_body(self, body: str) -> None:
        self.last_bot_body_hash = hashlib.md5(body.strip().encode()).hexdigest()
        self.sent_bodies.append(body.strip())

    def is_duplicate_body(self, body: str) -> bool:
        h = hashlib.md5(body.strip().encode()).hexdigest()
        return h == self.last_bot_body_hash

    # ------------------------------------------------------------------
    # Intent classification (deterministic)
    # ------------------------------------------------------------------

    def classify_incoming(self, message: str) -> str:
        """
        Return one of: 'accept', 'defer', 'decline', 'hostile', 'auto_reply', 'normal'
        """
        msg = message.strip()
        h = hashlib.md5(msg.lower().encode()).hexdigest()
        count = self._incoming_hash_counts.get(h, 0)

        # Auto-reply: same message 3+ times, or contains known phrases
        if count >= 3:
            return "auto_reply"
        msg_lower = msg.lower()
        for phrase in _AUTO_REPLY_PHRASES:
            if phrase in msg_lower:
                return "auto_reply"

        if _HOSTILE.search(msg):
            return "hostile"
        if _DECLINE.search(msg):
            return "decline"
        if _ACCEPTANCE.search(msg):
            return "accept"
        if _DEFERRAL.search(msg):
            return "defer"
        return "normal"

    def transition(self, intent: str) -> None:
        """Apply deterministic state transition from intent."""
        if self.state in (ConvState.COMPLETED, ConvState.ENDED):
            return

        if intent == "accept":
            self.state = ConvState.ACTION_READY
        elif intent == "defer":
            self.state = ConvState.WAITING
        elif intent in ("decline", "hostile"):
            self.state = ConvState.DECLINED
        elif intent == "auto_reply":
            if self.state == ConvState.AUTO_REPLY:
                self.state = ConvState.ENDED
            else:
                self.state = ConvState.AUTO_REPLY
        elif intent == "normal":
            if self.state == ConvState.NEW:
                self.state = ConvState.ENGAGED
            elif self.state == ConvState.PITCHED:
                self.state = ConvState.QUALIFYING
            # other states remain

    def mark_ended(self) -> None:
        self.state = ConvState.ENDED

    def mark_completed(self) -> None:
        self.state = ConvState.COMPLETED

    def after_send(self) -> None:
        """Update state after the bot sends a message."""
        if self.state == ConvState.NEW:
            self.state = ConvState.PITCHED
        elif self.state == ConvState.ACTION_READY:
            self.state = ConvState.COMPLETED

    def recent_turns(self, n: int = 2) -> List[dict]:
        """Return last n turns as simple dicts for LLM context."""
        return [{"from": t.from_role, "msg": t.message} for t in self.turns[-n:]]


class ConversationStore:
    """Thread-safe store for all in-flight conversations."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._convs: Dict[str, Conversation] = {}

    def get_or_create(self, conv_id: str, merchant_id: str,
                      customer_id: Optional[str] = None) -> Conversation:
        with self._lock:
            if conv_id not in self._convs:
                self._convs[conv_id] = Conversation(conv_id, merchant_id, customer_id)
            return self._convs[conv_id]

    def get(self, conv_id: str) -> Optional[Conversation]:
        with self._lock:
            return self._convs.get(conv_id)

    def active_convs_for_merchant(self, merchant_id: str) -> List[Conversation]:
        with self._lock:
            return [c for c in self._convs.values()
                    if c.merchant_id == merchant_id and not c.is_terminal]

    def teardown(self) -> None:
        with self._lock:
            self._convs.clear()
