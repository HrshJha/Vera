"""
Fact Packet Builder.

Converts raw context into a compact, verified evidence set for the LLM.
Every fact has a source path. No unverified facts are included.
"""
from __future__ import annotations

from typing import Any, List, Optional


def _safe(d: dict, *keys, default=None) -> Any:
    """Safe nested dict access."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is None:
            return default
    return cur


class Fact:
    __slots__ = ("id", "text", "source")

    def __init__(self, fid: str, text: str, source: str) -> None:
        self.id = fid
        self.text = text
        self.source = source

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "source": self.source}


class FactPacket:
    def __init__(self) -> None:
        self.primary_signal: str = ""
        self.facts: List[Fact] = []
        self.allowed_claims: List[str] = []
        self.forbidden_claims: List[str] = []

    def add(self, fid: str, text: str, source: str) -> None:
        self.facts.append(Fact(fid, text, source))

    def to_dict(self) -> dict:
        return {
            "primary_signal": self.primary_signal,
            "facts": [f.to_dict() for f in self.facts],
            "allowed_claims": self.allowed_claims,
            "forbidden_claims": self.forbidden_claims,
        }


class FactPacketBuilder:
    """Build verified fact packets for a specific (trigger, merchant, category, customer)."""

    def build(
        self,
        trigger: dict,
        merchant: dict,
        category: dict,
        customer: Optional[dict] = None,
        conv_turns: Optional[list] = None,
    ) -> FactPacket:
        fp = FactPacket()
        fp.forbidden_claims = list(category.get("voice", {}).get("vocab_taboo", []))
        fp.allowed_claims = list(category.get("voice", {}).get("vocab_allowed", []))

        kind = trigger.get("kind", "unknown")
        fp.primary_signal = kind

        idx = 1  # fact id counter

        def add(text: str, source: str) -> None:
            nonlocal idx
            fp.add(f"f{idx}", text, source)
            idx += 1

        # --- Merchant identity (always include) ---
        identity = merchant.get("identity", {})
        name = identity.get("name", "")
        owner = identity.get("owner_first_name", "")
        locality = identity.get("locality", "")
        city = identity.get("city", "")
        languages = identity.get("languages", ["en"])

        if owner:
            add(f"Owner first name: {owner}", "merchant.identity.owner_first_name")
        if name:
            add(f"Business name: {name}", "merchant.identity.name")
        if locality:
            add(f"Location: {locality}, {city}", "merchant.identity.locality")
        if languages:
            add(f"Language preference: {', '.join(languages)}", "merchant.identity.languages")

        # --- Subscription ---
        sub = merchant.get("subscription", {})
        sub_status = sub.get("status", "")
        days_rem = sub.get("days_remaining")
        plan = sub.get("plan", "")
        if days_rem is not None:
            add(f"Subscription: {plan} plan, {days_rem} days remaining, status={sub_status}",
                "merchant.subscription")

        # --- Performance ---
        perf = merchant.get("performance", {})
        views = perf.get("views")
        calls = perf.get("calls")
        ctr = perf.get("ctr")
        delta = perf.get("delta_7d", {})

        if views is not None:
            add(f"Google profile views (30d): {views}", "merchant.performance.views")
        if calls is not None:
            add(f"Calls received (30d): {calls}", "merchant.performance.calls")
        if ctr is not None:
            add(f"Click-through rate: {ctr:.1%}", "merchant.performance.ctr")
        if delta:
            views_pct = delta.get("views_pct")
            calls_pct = delta.get("calls_pct")
            if views_pct is not None:
                direction = "up" if views_pct >= 0 else "down"
                add(f"Views trend (7d): {direction} {abs(views_pct):.0%}",
                    "merchant.performance.delta_7d.views_pct")
            if calls_pct is not None:
                direction = "up" if calls_pct >= 0 else "down"
                add(f"Calls trend (7d): {direction} {abs(calls_pct):.0%}",
                    "merchant.performance.delta_7d.calls_pct")

        # --- Active offers ---
        offers = merchant.get("offers", [])
        active_offers = [o for o in offers if o.get("status") == "active"]
        if active_offers:
            titles = ", ".join(o["title"] for o in active_offers[:3])
            add(f"Active offers: {titles}", "merchant.offers[active]")

        # --- Signals ---
        signals = merchant.get("signals", [])
        if signals:
            add(f"Business signals: {', '.join(signals[:5])}", "merchant.signals")

        # --- Customer aggregate ---
        ca = merchant.get("customer_aggregate", {})
        total = ca.get("total_unique_ytd")
        lapsed = ca.get("lapsed_180d_plus")
        retention = ca.get("retention_6mo_pct")
        high_risk = ca.get("high_risk_adult_count")
        if total:
            add(f"Total unique customers YTD: {total}", "merchant.customer_aggregate.total_unique_ytd")
        if lapsed:
            add(f"Lapsed customers (180d+): {lapsed}", "merchant.customer_aggregate.lapsed_180d_plus")
        if retention:
            add(f"6-month retention rate: {retention:.0%}", "merchant.customer_aggregate.retention_6mo_pct")
        if high_risk:
            add(f"High-risk adult patient count: {high_risk}", "merchant.customer_aggregate.high_risk_adult_count")

        # --- Peer benchmarks ---
        peer = category.get("peer_stats", {})
        avg_ctr = peer.get("avg_ctr")
        if avg_ctr is not None and ctr is not None:
            if ctr < avg_ctr:
                add(f"CTR is {ctr:.1%}, below peer median of {avg_ctr:.1%}",
                    "category.peer_stats.avg_ctr vs merchant.performance.ctr")
            else:
                add(f"CTR is {ctr:.1%}, above peer median of {avg_ctr:.1%}",
                    "category.peer_stats.avg_ctr vs merchant.performance.ctr")

        # --- Trigger-specific facts ---
        self._add_trigger_facts(fp, trigger, category, add)

        # --- Customer context ---
        if customer:
            self._add_customer_facts(fp, customer, add)

        # --- Conversation history (last 2 turns) ---
        if conv_turns:
            for t in conv_turns[-2:]:
                add(f"Prev turn [{t.get('from')}]: {t.get('msg', '')[:100]}",
                    "conversation.recent_turn")

        # --- Review themes ---
        themes = merchant.get("review_themes", [])
        for theme in themes[:2]:
            sentiment = theme.get("sentiment", "")
            theme_name = theme.get("theme", "")
            occ = theme.get("occurrences_30d", 0)
            quote = theme.get("common_quote", "")
            if theme_name:
                add(f"Review theme '{theme_name}': {sentiment}, {occ}× in 30d. Quote: \"{quote}\"",
                    f"merchant.review_themes.{theme_name}")

        return fp

    def _add_trigger_facts(self, fp: FactPacket, trigger: dict,
                            category: dict, add_fn) -> None:
        kind = trigger.get("kind", "")
        payload = trigger.get("payload", {})
        is_placeholder = payload.get("placeholder", False)

        if is_placeholder:
            # Don't invent facts from placeholder triggers - return gracefully
            topic = payload.get("metric_or_topic", "")
            if topic:
                add_fn(f"Trigger topic area: {topic} (no specific data available)",
                       "trigger.payload.placeholder_topic")
            return

        if kind == "research_digest":
            item_id = payload.get("top_item_id")
            digest_items = category.get("digest", [])
            item = next((d for d in digest_items if d.get("id") == item_id), None)
            if item:
                title = item.get("title", "")
                source = item.get("source", "")
                trial_n = item.get("trial_n")
                segment = item.get("patient_segment", "")
                actionable = item.get("actionable", "")
                if title:
                    add_fn(f"Research finding: {title}", f"category.digest.{item_id}.title")
                if source:
                    add_fn(f"Source: {source}", f"category.digest.{item_id}.source")
                if trial_n:
                    add_fn(f"Trial size: {trial_n} patients", f"category.digest.{item_id}.trial_n")
                if segment:
                    add_fn(f"Patient segment: {segment}", f"category.digest.{item_id}.patient_segment")
                if actionable:
                    add_fn(f"Actionable step: {actionable}", f"category.digest.{item_id}.actionable")

        elif kind == "regulation_change":
            item_id = payload.get("top_item_id")
            deadline = payload.get("deadline_iso", "")
            digest_items = category.get("digest", [])
            item = next((d for d in digest_items if d.get("id") == item_id), None)
            if item:
                add_fn(f"Compliance: {item.get('title', '')}", f"category.digest.{item_id}.title")
                add_fn(f"Source: {item.get('source', '')}", f"category.digest.{item_id}.source")
                add_fn(f"Action needed: {item.get('actionable', '')}", f"category.digest.{item_id}.actionable")
            if deadline:
                add_fn(f"Compliance deadline: {deadline}", "trigger.payload.deadline_iso")

        elif kind == "recall_due":
            service_due = payload.get("service_due", "")
            last_date = payload.get("last_service_date", "")
            due_date = payload.get("due_date", "")
            slots = payload.get("available_slots", [])
            if service_due:
                add_fn(f"Service due: {service_due}", "trigger.payload.service_due")
            if last_date:
                add_fn(f"Last service date: {last_date}", "trigger.payload.last_service_date")
            if due_date:
                add_fn(f"Due date: {due_date}", "trigger.payload.due_date")
            for i, slot in enumerate(slots[:2]):
                add_fn(f"Available slot {i+1}: {slot.get('label', '')}", f"trigger.payload.available_slots[{i}]")

        elif kind in ("perf_dip", "perf_spike", "seasonal_perf_dip"):
            metric = payload.get("metric", "")
            delta_pct = payload.get("delta_pct")
            window = payload.get("window", "7d")
            if metric and delta_pct is not None:
                direction = "up" if delta_pct > 0 else "down"
                add_fn(f"{metric} is {direction} {abs(delta_pct):.0%} over {window}",
                       "trigger.payload.delta_pct")
            season_note = payload.get("season_note", "")
            if season_note:
                add_fn(f"Season context: {season_note}", "trigger.payload.season_note")

        elif kind == "renewal_due":
            days = payload.get("days_remaining")
            plan = payload.get("plan", "")
            amount = payload.get("renewal_amount")
            if days is not None:
                add_fn(f"Subscription expires in {days} days (plan: {plan})",
                       "trigger.payload.days_remaining")
            if amount:
                add_fn(f"Renewal amount: ₹{amount}", "trigger.payload.renewal_amount")

        elif kind == "festival_upcoming":
            festival = payload.get("festival", "")
            date = payload.get("date", "")
            days_until = payload.get("days_until")
            if festival:
                add_fn(f"Upcoming festival: {festival} on {date} ({days_until} days away)",
                       "trigger.payload.festival")

        elif kind == "ipl_match_today":
            match = payload.get("match", "")
            venue = payload.get("venue", "")
            if match:
                add_fn(f"IPL match today: {match} at {venue}", "trigger.payload.match")

        elif kind == "review_theme_emerged":
            theme = payload.get("theme", "")
            occ = payload.get("occurrences_30d", 0)
            trend = payload.get("trend", "")
            quote = payload.get("common_quote", "")
            if theme:
                add_fn(f"Review theme '{theme}' appeared {occ}× this month ({trend}), e.g.: \"{quote}\"",
                       "trigger.payload.theme")

        elif kind == "milestone_reached":
            metric = payload.get("metric", "")
            value_now = payload.get("value_now")
            milestone = payload.get("milestone_value")
            imminent = payload.get("is_imminent", False)
            if imminent and milestone:
                add_fn(f"Approaching milestone: {value_now}/{milestone} {metric}",
                       "trigger.payload.milestone_value")
            elif milestone:
                add_fn(f"Milestone reached: {milestone} {metric}",
                       "trigger.payload.milestone_value")

        elif kind == "active_planning_intent":
            topic = payload.get("intent_topic", "")
            last_msg = payload.get("merchant_last_message", "")
            if topic:
                add_fn(f"Merchant is actively planning: {topic}", "trigger.payload.intent_topic")
            if last_msg:
                add_fn(f"Merchant's last message: \"{last_msg}\"",
                       "trigger.payload.merchant_last_message")

        elif kind == "winback_eligible":
            days_exp = payload.get("days_since_expiry")
            lapsed = payload.get("lapsed_customers_added_since_expiry")
            if days_exp:
                add_fn(f"Subscription expired {days_exp} days ago", "trigger.payload.days_since_expiry")
            if lapsed:
                add_fn(f"New lapsed customers since expiry: {lapsed}",
                       "trigger.payload.lapsed_customers_added_since_expiry")

        elif kind == "wedding_package_followup":
            wedding_date = payload.get("wedding_date", "")
            days_to = payload.get("days_to_wedding")
            trial_done = payload.get("trial_completed", "")
            next_step = payload.get("next_step_window_open", "")
            if wedding_date:
                add_fn(f"Wedding date: {wedding_date} ({days_to} days away)",
                       "trigger.payload.wedding_date")
            if trial_done:
                add_fn(f"Bridal trial completed: {trial_done}", "trigger.payload.trial_completed")
            if next_step:
                add_fn(f"Next step window: {next_step.replace('_', ' ')}",
                       "trigger.payload.next_step_window_open")

        elif kind == "curious_ask_due":
            template = payload.get("ask_template", "")
            if template:
                add_fn(f"Curiosity topic: {template.replace('_', ' ')}",
                       "trigger.payload.ask_template")

        elif kind == "dormant_with_vera":
            days = payload.get("days_since_last_reply") or payload.get("days_dormant")
            if days:
                add_fn(f"No merchant reply for {days} days", "trigger.payload.days_dormant")

        elif kind == "appointment_tomorrow":
            count = payload.get("appointment_count")
            service = payload.get("service", "")
            if count:
                add_fn(f"{count} appointments tomorrow", "trigger.payload.appointment_count")
            if service:
                add_fn(f"Service: {service}", "trigger.payload.service")

    def _add_customer_facts(self, fp: FactPacket, customer: dict, add_fn) -> None:
        ci = customer.get("identity", {})
        name = ci.get("name", "")
        lang = ci.get("language_pref", "")
        rel = customer.get("relationship", {})
        state = customer.get("state", "")
        prefs = customer.get("preferences", {})

        if name:
            add_fn(f"Customer name: {name}", "customer.identity.name")
        if lang:
            add_fn(f"Customer language preference: {lang}", "customer.identity.language_pref")
        if state:
            add_fn(f"Customer state: {state}", "customer.state")

        last_visit = rel.get("last_visit", "")
        visits = rel.get("visits_total")
        services = rel.get("services_received", [])
        if last_visit:
            add_fn(f"Last visit: {last_visit}", "customer.relationship.last_visit")
        if visits:
            add_fn(f"Total visits: {visits}", "customer.relationship.visits_total")
        if services:
            add_fn(f"Services received: {', '.join(services[:4])}",
                   "customer.relationship.services_received")

        pref_slot = prefs.get("preferred_slots", "")
        if pref_slot:
            add_fn(f"Preferred booking time: {pref_slot.replace('_', ' ')}",
                   "customer.preferences.preferred_slots")

        consent = customer.get("consent", {})
        scope = consent.get("scope", [])
        if scope:
            add_fn(f"Consent scope: {', '.join(scope)}", "customer.consent.scope")
            fp.allowed_claims.append(f"customer_consented_to:{','.join(scope)}")
