"""
Versioned context store with atomic updates and O(1) index lookups.

Stores four context scopes: category, merchant, customer, trigger.
Maintains fast indexes: merchant_id -> category_slug, merchant_id -> customers.
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


VALID_SCOPES = {"category", "merchant", "customer", "trigger"}
MAX_PAYLOAD_BYTES = 500_000  # 500 KB


class ContextStore:
    """Thread-safe versioned store for all 4 context types."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # (scope, context_id) -> {"version": int, "payload": dict, "stored_at": str}
        self._store: Dict[Tuple[str, str], Dict[str, Any]] = {}

        # Fast indexes
        self._merchant_to_category: Dict[str, str] = {}   # merchant_id -> category_slug
        self._merchant_to_customers: Dict[str, list] = {} # merchant_id -> [customer_id]
        self._active_conversations: Dict[str, str] = {}   # conv_id -> merchant_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, scope: str, context_id: str, version: int, payload: dict) -> dict:
        """
        Store a context version.

        Returns:
            {"accepted": True, "ack_id": ..., "stored_at": ...}
            {"accepted": False, "reason": "stale_version", "current_version": N}
            {"accepted": False, "reason": "invalid_scope"}
            {"accepted": False, "reason": "payload_too_large"}
        """
        if scope not in VALID_SCOPES:
            return {"accepted": False, "reason": "invalid_scope",
                    "details": f"scope must be one of {VALID_SCOPES}"}

        # Size check (rough: json bytes estimate)
        size = len(str(payload).encode("utf-8"))
        if size > MAX_PAYLOAD_BYTES:
            return {"accepted": False, "reason": "payload_too_large",
                    "details": f"payload is {size} bytes; limit is {MAX_PAYLOAD_BYTES}"}

        with self._lock:
            key = (scope, context_id)
            existing = self._store.get(key)

            if existing is not None:
                if existing["version"] == version:
                    # Idempotent - already have this version
                    return {"accepted": True,
                            "ack_id": f"ack_{context_id}_v{version}_noop",
                            "stored_at": existing["stored_at"]}
                if existing["version"] > version:
                    return {"accepted": False, "reason": "stale_version",
                            "current_version": existing["version"]}

            stored_at = datetime.utcnow().isoformat() + "Z"
            self._store[key] = {"version": version, "payload": payload,
                                 "stored_at": stored_at}
            self._update_indexes(scope, context_id, payload)

            return {"accepted": True,
                    "ack_id": f"ack_{context_id}_v{version}",
                    "stored_at": stored_at}

    def get(self, scope: str, context_id: str) -> Optional[dict]:
        """Return payload dict or None if not found."""
        with self._lock:
            entry = self._store.get((scope, context_id))
            return entry["payload"] if entry else None

    def get_version(self, scope: str, context_id: str) -> Optional[int]:
        """Return current stored version or None."""
        with self._lock:
            entry = self._store.get((scope, context_id))
            return entry["version"] if entry else None

    def count_by_scope(self) -> Dict[str, int]:
        """Return {scope: count} for healthz."""
        counts: Dict[str, int] = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        with self._lock:
            for (scope, _) in self._store:
                if scope in counts:
                    counts[scope] += 1
        return counts

    def get_merchant(self, merchant_id: str) -> Optional[dict]:
        return self.get("merchant", merchant_id)

    def get_category_for_merchant(self, merchant_id: str) -> Optional[dict]:
        with self._lock:
            slug = self._merchant_to_category.get(merchant_id)
            if not slug:
                return None
        return self.get("category", slug)

    def get_customers_for_merchant(self, merchant_id: str) -> list:
        """Return list of customer payload dicts for this merchant."""
        with self._lock:
            customer_ids = list(self._merchant_to_customers.get(merchant_id, []))
        result = []
        for cid in customer_ids:
            c = self.get("customer", cid)
            if c:
                result.append(c)
        return result

    def get_customer(self, customer_id: str) -> Optional[dict]:
        return self.get("customer", customer_id)

    def get_trigger(self, trigger_id: str) -> Optional[dict]:
        return self.get("trigger", trigger_id)

    def get_category(self, slug: str) -> Optional[dict]:
        return self.get("category", slug)

    def teardown(self) -> None:
        """Wipe all state."""
        with self._lock:
            self._store.clear()
            self._merchant_to_category.clear()
            self._merchant_to_customers.clear()
            self._active_conversations.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_indexes(self, scope: str, context_id: str, payload: dict) -> None:
        """Maintain fast lookup indexes."""
        if scope == "merchant":
            slug = payload.get("category_slug")
            if slug:
                self._merchant_to_category[context_id] = slug

        elif scope == "customer":
            merchant_id = payload.get("merchant_id")
            if merchant_id:
                customers = self._merchant_to_customers.setdefault(merchant_id, [])
                if context_id not in customers:
                    customers.append(context_id)
