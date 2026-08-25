"""
Signal Ranker & Hard Gates.

All decisions are deterministic. The LLM never touches this module.

Pipeline:
  1. Resolve trigger context
  2. Apply hard gates
  3. Score candidates
  4. Rank deterministically
  5. Return top candidates (max 20 per tick)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from vera.engine.context_store import ContextStore
from vera.engine.suppression import SuppressionEngine
from vera.engine.conversation import ConversationStore, ConvState


MAX_ACTIONS_PER_TICK = 20


# Urgency multiplied weight for internal vs external triggers
INTERNAL_URGENCY_BONUS = 0.1  # small bonus for internal over external at same score


class Candidate:
    """A potential action that has passed hard gates."""

    __slots__ = ("trigger_id", "trigger", "merchant_id", "merchant", "category",
                 "customer_id", "customer", "score", "priority")

    def __init__(self, trigger_id: str, trigger: dict, merchant_id: str,
                 merchant: dict, category: dict, customer_id: Optional[str],
                 customer: Optional[dict]) -> None:
        self.trigger_id = trigger_id
        self.trigger = trigger
        self.merchant_id = merchant_id
        self.merchant = merchant
        self.category = category
        self.customer_id = customer_id
        self.customer = customer
        self.score: float = 0.0
        self.priority: float = 0.0


class HardGateError(Exception):
    pass


class SignalRanker:
    """
    Deterministic signal ranker implementing the priority formula:

    priority =
        20 * urgency
      + 20 * actionability
      + 15 * business_impact
      + 15 * merchant_fit
      + 10 * trigger_specificity
      + 10 * recency
      +  5 * offerability
      +  5 * conversation_alignment
      - 20 * fatigue
      - 25 * stale_or_conflicting
    """

    def __init__(self, ctx: ContextStore, conv_store: ConversationStore,
                 suppression: SuppressionEngine) -> None:
        self._ctx = ctx
        self._conv = conv_store
        self._sup = suppression

    def rank(self, available_trigger_ids: List[str],
             sim_now_iso: str) -> List[Candidate]:
        """
        Return ranked list of candidates that passed all hard gates.
        Maximum MAX_ACTIONS_PER_TICK candidates returned.
        """
        sim_now = _parse_iso(sim_now_iso)
        candidates: List[Candidate] = []

        # Track which merchants we've already selected an action for in this tick
        selected_merchant_convs: set = set()

        for trg_id in available_trigger_ids:
            try:
                candidate = self._evaluate_trigger(trg_id, sim_now)
                if candidate is not None:
                    key = f"{candidate.merchant_id}:{candidate.customer_id or 'none'}"
                    if key not in selected_merchant_convs:
                        candidates.append(candidate)
                        selected_merchant_convs.add(key)
            except HardGateError:
                continue
            except Exception:
                continue

        # Sort deterministically
        candidates.sort(key=lambda c: (
            -c.priority,
            0 if c.customer_id is None else 1,
            -c.trigger.get("urgency", 1),
            # internal before external at same priority
            0 if c.trigger.get("source") == "internal" else 1,
            c.trigger_id,  # lexicographic tiebreaker
        ))

        return candidates[:MAX_ACTIONS_PER_TICK]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evaluate_trigger(self, trg_id: str, sim_now: float) -> Optional[Candidate]:
        """Evaluate a single trigger through hard gates and scoring. Returns None to skip."""
        import logging
        logger = logging.getLogger(__name__)

        trigger = self._ctx.get_trigger(trg_id)
        if trigger is None:
            logger.debug(f"[gate_reject] {trg_id}: trigger_not_found")
            raise HardGateError(f"trigger_not_found:{trg_id}")

        # --- HARD GATES ---

        # 1. Trigger expired?
        expires_at = trigger.get("expires_at")
        if expires_at:
            exp_ts = _parse_iso(expires_at)
            if sim_now > exp_ts:
                logger.debug(f"[gate_reject] {trg_id}: trigger_expired")
                raise HardGateError(f"trigger_expired:{trg_id}")

        # 2. Resolve merchant
        merchant_id = trigger.get("merchant_id")
        if not merchant_id:
            logger.debug(f"[gate_reject] {trg_id}: no_merchant_id")
            raise HardGateError(f"no_merchant_id:{trg_id}")

        merchant = self._ctx.get_merchant(merchant_id)
        if merchant is None:
            logger.debug(f"[gate_reject] {trg_id}: merchant_not_found:{merchant_id}")
            raise HardGateError(f"merchant_not_found:{merchant_id}")

        # 3. Resolve category
        category = self._ctx.get_category_for_merchant(merchant_id)
        if category is None:
            logger.debug(f"[gate_reject] {trg_id}: category_not_found_for:{merchant_id}")
            raise HardGateError(f"category_not_found_for:{merchant_id}")

        # 4. Suppression check
        suppression_key = trigger.get("suppression_key", trg_id)
        if self._sup.is_suppressed(suppression_key):
            logger.debug(f"[gate_reject] {trg_id}: suppressed:{suppression_key}")
            raise HardGateError(f"suppressed:{suppression_key}")

        if self._sup.is_declined(merchant_id):
            logger.debug(f"[gate_reject] {trg_id}: merchant_declined:{merchant_id}")
            raise HardGateError(f"merchant_declined:{merchant_id}")

        if self._sup.is_trigger_sent(trg_id):
            logger.debug(f"[gate_reject] {trg_id}: trigger_sent:{trg_id}")
            raise HardGateError(f"trigger_sent:{trg_id}")

        # 5. Customer scope gates
        customer_id = trigger.get("customer_id")
        customer = None
        if customer_id:
            customer = self._ctx.get_customer(customer_id)
            if customer is None:
                logger.debug(f"[gate_reject] {trg_id}: customer_not_found:{customer_id}")
                raise HardGateError(f"customer_not_found:{customer_id}")

            # Verify merchant relationship
            if customer.get("merchant_id") != merchant_id:
                logger.debug(f"[gate_reject] {trg_id}: customer_merchant_mismatch:{customer_id}")
                raise HardGateError(f"customer_merchant_mismatch:{customer_id}")

            # Consent check for customer-scope triggers
            consent_scope = customer.get("consent", {}).get("scope", [])
            kind = trigger.get("kind", "")
            if not _consent_covers_trigger(kind, consent_scope):
                logger.debug(f"[gate_reject] {trg_id}: no_consent:{customer_id}:{kind}")
                raise HardGateError(f"no_consent:{customer_id}:{kind}")

        # 6. Existing conversation checks
        active_convs = self._conv.active_convs_for_merchant(merchant_id)
        for conv in active_convs:
            if conv.trigger_id == trg_id:
                logger.debug(f"[gate_reject] {trg_id}: trigger_already_in_flight")
                raise HardGateError(f"trigger_already_in_flight:{trg_id}")
            if conv.state == ConvState.DECLINED:
                logger.debug(f"[gate_reject] {trg_id}: merchant_declined_in_conv:{merchant_id}")
                raise HardGateError(f"merchant_declined_in_conv:{merchant_id}")
            if conv.state == ConvState.ENDED:
                continue

        # --- BUILD CANDIDATE ---
        cand = Candidate(trg_id, trigger, merchant_id, merchant, category,
                         customer_id, customer)
        cand.priority = self._score(cand, sim_now)

        logger.debug(f"[gate_pass] {trg_id}: priority={cand.priority:.2f}")

        return cand

    def _score(self, cand: Candidate, sim_now: float) -> float:
        """Compute priority score (higher = better)."""
        t = cand.trigger
        m = cand.merchant
        payload = t.get("payload", {})
        is_placeholder = payload.get("placeholder", False)

        # Urgency (0-1, normalized from 1-5)
        urgency_raw = t.get("urgency", 1)
        urgency = (urgency_raw - 1) / 4.0

        # Actionability: can the merchant do something right now?
        kind = t.get("kind", "")
        actionability = _actionability_score(kind, m)

        # Business impact: severity of the signal
        business_impact = _business_impact(t, m)

        # Merchant fit: how relevant is this to this specific merchant?
        merchant_fit = _merchant_fit(t, m, cand.category, cand.customer)

        # Trigger specificity: does the payload have concrete data?
        if is_placeholder:
            trigger_specificity = 0.1  # very low for placeholder triggers
        else:
            trigger_specificity = _trigger_specificity(payload)

        # Recency: how fresh is this signal?
        expires_at = t.get("expires_at")
        recency = _recency_score(expires_at, sim_now)

        # Offerability: does the merchant have active offers relevant to this trigger?
        active_offers = [o for o in m.get("offers", []) if o.get("status") == "active"]
        offerability = min(1.0, len(active_offers) * 0.5)

        # Conversation alignment: does existing conv state favor this trigger?
        conv_alignment = _conv_alignment(cand.merchant_id, kind,
                                          self._conv.active_convs_for_merchant(cand.merchant_id))

        # Fatigue: has the merchant received many messages recently?
        fatigue = 0.0  # suppression handles this; minor penalty for merchant signals
        if not self._sup.merchant_gap_ok(cand.merchant_id, min_gap_sec=1800):
            fatigue = 0.5

        # Stale or conflicting: is the trigger outdated?
        stale = 0.5 if is_placeholder else 0.0

        score = (
            20 * urgency
            + 20 * actionability
            + 15 * business_impact
            + 15 * merchant_fit
            + 10 * trigger_specificity
            + 10 * recency
            + 5 * offerability
            + 5 * conv_alignment
            - 20 * fatigue
            - 25 * stale
        )

        # Internal triggers get a small tie-breaking bonus
        if t.get("source") == "internal":
            score += INTERNAL_URGENCY_BONUS

        return score


# ------------------------------------------------------------------
# Scoring helpers (pure functions, deterministic)
# ------------------------------------------------------------------

_HIGH_ACTIONABILITY_KINDS = {
    "active_planning_intent", "recall_due", "renewal_due", "perf_dip",
    "appointment_tomorrow", "regulation_change", "review_theme_emerged",
    "winback_eligible",
}

_MEDIUM_ACTIONABILITY_KINDS = {
    "research_digest", "perf_spike", "festival_upcoming", "ipl_match_today",
    "wedding_package_followup", "milestone_reached", "seasonal_perf_dip",
}


def _actionability_score(kind: str, merchant: dict) -> float:
    if kind in _HIGH_ACTIONABILITY_KINDS:
        return 0.9
    if kind in _MEDIUM_ACTIONABILITY_KINDS:
        return 0.6
    return 0.3  # low for curious_ask_due, dormant etc.


def _business_impact(trigger: dict, merchant: dict) -> float:
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})

    # Renewal: imminent expiry = high impact
    days_rem = (payload.get("days_remaining")
                or merchant.get("subscription", {}).get("days_remaining", 99))
    if kind == "renewal_due":
        return max(0.1, 1.0 - days_rem / 30.0)

    # Performance dip: scale with delta magnitude
    delta_pct = payload.get("delta_pct")
    if delta_pct is not None and delta_pct < 0:
        return min(1.0, abs(delta_pct) * 1.5)

    # Recall due: always high impact
    if kind == "recall_due":
        return 0.8

    # Compliance change: highest urgency
    if kind == "regulation_change":
        return 1.0

    # Festival: low impact (awareness only)
    if kind in ("festival_upcoming", "ipl_match_today"):
        return 0.4

    return 0.5


def _merchant_fit(trigger: dict, merchant: dict, category: dict,
                   customer=None) -> float:
    """How relevant is this trigger to this specific merchant?"""
    kind = trigger.get("kind", "")
    payload = trigger.get("payload", {})
    signals = merchant.get("signals", [])

    fit = 0.5

    if kind == "research_digest":
        # Higher fit if merchant has relevant patient cohort
        ca = merchant.get("customer_aggregate", {})
        if ca.get("high_risk_adult_count", 0) > 50:
            fit = 0.9
        # Check if category matches
        if payload.get("category") == category.get("slug"):
            fit = max(fit, 0.7)

    elif kind == "perf_dip":
        # Signals-based fit
        if any("perf_dip" in s or "dormant" in s for s in signals):
            fit = 0.9

    elif kind == "renewal_due":
        sub = merchant.get("subscription", {})
        days_rem = sub.get("days_remaining", 99)
        if days_rem <= 14:
            fit = 1.0
        elif days_rem <= 30:
            fit = 0.8

    elif kind == "recall_due" and customer:
        # Customer must belong to merchant
        if customer.get("merchant_id") == merchant.get("merchant_id", ""):
            fit = 0.9

    elif kind == "dormant_with_vera":
        if any("dormant" in s for s in signals):
            fit = 0.85

    return min(1.0, fit)


def _trigger_specificity(payload: dict) -> float:
    """Does the payload have concrete, actionable data?"""
    if not payload:
        return 0.1
    has_numbers = any(isinstance(v, (int, float)) for v in payload.values()
                      if not isinstance(v, bool))
    has_dates = any("date" in k.lower() or "iso" in k.lower()
                    for k in payload.keys())
    has_text = any(isinstance(v, str) and len(v) > 10 for v in payload.values())
    score = (0.4 * has_numbers + 0.3 * has_dates + 0.3 * has_text)
    return max(0.1, score)


def _recency_score(expires_at: Optional[str], sim_now: float) -> float:
    """Higher score for triggers expiring soon (more urgent)."""
    if not expires_at:
        return 0.5
    try:
        exp_ts = _parse_iso(expires_at)
        time_left_sec = max(0, exp_ts - sim_now)
        # Triggers expiring within 24h get max recency
        if time_left_sec < 86400:
            return 1.0
        # Decays over 7 days
        return max(0.0, 1.0 - time_left_sec / (7 * 86400))
    except Exception:
        return 0.5


def _conv_alignment(merchant_id: str, kind: str, active_convs: list) -> float:
    """How well does this trigger align with existing conversation state?"""
    if not active_convs:
        return 0.5  # neutral when no active convs

    for conv in active_convs:
        if conv.state == ConvState.ACTION_READY and kind == "active_planning_intent":
            return 1.0
        if conv.state == ConvState.WAITING:
            return 0.3  # don't pile on if merchant asked for time
        if conv.state == ConvState.AUTO_REPLY:
            return 0.1  # auto-reply situation = very low alignment
    return 0.5


def _consent_covers_trigger(kind: str, consent_scope: list) -> bool:
    """Check if consent allows this trigger kind."""
    if not consent_scope:
        return False

    consent_map = {
        "recall_due": ["recall_reminders", "appointment_reminders"],
        "appointment_tomorrow": ["appointment_reminders"],
        "wedding_package_followup": ["marketing", "service_followup"],
        "customer_lapsed_soft": ["recall_reminders", "marketing"],
        "customer_lapsed_hard": ["recall_reminders", "marketing"],
        "winback_eligible": ["marketing", "winback"],
    }
    required = consent_map.get(kind, [])
    if not required:
        return True  # no consent requirement for merchant-scope triggers
    return any(r in consent_scope for r in required)


def _parse_iso(iso_str: str) -> float:
    """Parse ISO datetime to UTC timestamp."""
    try:
        s = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return 0.0
