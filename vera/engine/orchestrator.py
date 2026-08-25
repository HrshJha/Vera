"""
Vera Orchestrator — wires all engine components together.

Pipeline:
  Context → Hard Gates → Signal Ranker → Action Selector
  → Fact Packet → Category/Merchant/Conversation Policy
  → LLM Writer → Grounding Validator → CTA/Repetition Validator
  → Output

The deterministic system decides. The LLM writes.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from vera.engine.context_store import ContextStore
from vera.engine.conversation import ConversationStore, ConvState
from vera.engine.fact_builder import FactPacketBuilder
from vera.engine.llm_writer import LLMWriter
from vera.engine.message_families import get_family
from vera.engine.ranker import SignalRanker, Candidate
from vera.engine.suppression import SuppressionEngine
from vera.engine.validator import validate

import concurrent.futures

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main Vera engine orchestrator."""

    def __init__(self) -> None:
        self.ctx = ContextStore()
        self.conv_store = ConversationStore()
        self.suppression = SuppressionEngine()
        self.ranker = SignalRanker(self.ctx, self.conv_store, self.suppression)
        self.fact_builder = FactPacketBuilder()
        self.writer = LLMWriter(
            api_key=os.environ.get("GEMINI_API_KEY", ""),
            model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        )
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self._merchant_auto_reply_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Tick: generate proactive outbound messages
    # ------------------------------------------------------------------

    def tick(self, now_iso: str, available_trigger_ids: List[str]) -> List[dict]:
        """
        Process a tick and return list of actions.

        Each action: {conversation_id, merchant_id, customer_id, send_as,
                       trigger_id, template_name, template_params, body, cta,
                       suppression_key, rationale}
        """
        self.suppression.update_sim_time(now_iso)

        # Rank candidates deterministically
        candidates = self.ranker.rank(available_trigger_ids, now_iso)
        logger.info(f"[tick] Ranker returned {len(candidates)} candidates from {len(available_trigger_ids)} triggers")
        if not candidates:
            logger.warning(f"[tick] No candidates passed hard gates from {len(available_trigger_ids)} triggers")
            return []

        # Process top candidates — respect challenge limit of 20 actions per tick
        top_candidates = candidates[:20]
        logger.info(f"[tick] Processing top {len(top_candidates)} candidates (limited to 20)")
        futures = {self._executor.submit(self._compose_action, cand): cand for cand in top_candidates}
        
        actions = []
        skipped = 0
        try:
            for fut in concurrent.futures.as_completed(futures, timeout=120.0):
                try:
                    action = fut.result()
                    if action:
                        actions.append(action)
                    else:
                        skipped += 1
                except Exception as e:
                    cand = futures[fut]
                    logger.error(f"[tick] Parallel action failed for {cand.trigger_id}: {e}")
        except concurrent.futures.TimeoutError:
            logger.warning("[tick] Parallel composition hit 120s timeout safeguard; returning partial actions")

        logger.info(f"[tick] Generated {len(actions)} actions, skipped {skipped} candidates")
        return actions

    # ------------------------------------------------------------------
    # Reply: handle incoming merchant/customer message
    # ------------------------------------------------------------------

    def reply(self, conv_id: str, merchant_id: Optional[str],
              customer_id: Optional[str], from_role: str,
              message: str, turn_number: int) -> dict:
        """
        Handle an incoming reply.

        Returns: {"action": "send"|"wait"|"end", "body"?: str, "cta"?: str,
                   "wait_seconds"?: int, "rationale": str}
        """
        # Find or create conversation
        mid = merchant_id or "unknown"
        conv = self.conv_store.get_or_create(conv_id, mid, customer_id)

        if conv.is_terminal and conv.state == ConvState.ENDED:
            return {"action": "end", "rationale": "conversation_already_ended"}

        # Record the incoming turn
        conv.add_turn(from_role, message)

        # Classify intent (deterministic)
        intent = conv.classify_incoming(message)
        conv.transition(intent)

        logger.info(f"[reply] conv={conv_id} intent={intent} state={conv.state}")

        # Handle terminal intents
        if intent == "hostile":
            conv.mark_ended()
            self.suppression.record_decline(mid)
            return {"action": "end", "rationale": "merchant_hostile_message_graceful_exit"}

        if intent == "decline":
            conv.mark_ended()
            self.suppression.record_decline(mid)
            return {
                "action": "send",
                "body": "Understood, I won't contact you about this again. Best wishes!",
                "cta": "none",
                "rationale": "Merchant declined; sending polite exit message",
            }

        if intent == "auto_reply":
            count = self._merchant_auto_reply_counts.get(mid, 0) + 1
            self._merchant_auto_reply_counts[mid] = count
            if count >= 3 or conv.state == ConvState.ENDED:
                conv.mark_ended()
                return {"action": "end", "rationale": "auto_reply_detected_graceful_exit"}
            return {
                "action": "wait",
                "wait_seconds": 1800,
                "rationale": f"Auto-reply detected ({count}/3); backing off 30 min",
            }

        if intent == "defer":
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Merchant asked for more time; backing off 24h",
            }

        # For accept and normal - compose a reply
        return self._compose_reply(conv, intent, message, mid, customer_id)

    # ------------------------------------------------------------------
    # State teardown
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        """Wipe all state after test ends."""
        self.ctx.teardown()
        self.conv_store.teardown()
        self.suppression.teardown()
        self.writer._cache.clear()
        self._merchant_auto_reply_counts.clear()

    # ------------------------------------------------------------------
    # Internal: compose actions for tick
    # ------------------------------------------------------------------

    def _compose_action(self, cand: Candidate) -> Optional[dict]:
        """Compose a full action for a candidate."""
        try:
            trigger = cand.trigger
            merchant = cand.merchant
            category = cand.category
            customer = cand.customer
            kind = trigger.get("kind", "unknown")
            merchant_id = cand.merchant_id
            customer_id = cand.customer_id

            # Get message family
            family = get_family(kind)
            is_customer_scope = customer is not None
            send_as = family.effective_send_as(is_customer_scope)

            # Build fact packet
            fp = self.fact_builder.build(trigger, merchant, category, customer)

            # Check for placeholder trigger with no useful facts
            is_placeholder = trigger.get("payload", {}).get("placeholder", False)
            if is_placeholder and len(fp.facts) < 4:
                logger.info(f"[tick] Skipping placeholder trigger {cand.trigger_id} — insufficient facts")
                return None

            # Generate conversation ID
            conv_id = f"conv_{merchant_id}_{cand.trigger_id}"
            conv = self.conv_store.get_or_create(conv_id, merchant_id, customer_id)

            # Final suppression check with a tentative key
            suppression_key = trigger.get("suppression_key", cand.trigger_id)
            sup_reason = self.suppression.check(
                suppression_key, merchant_id, trigger_id=cand.trigger_id
            )
            if sup_reason:
                logger.info(f"[tick] Suppressed {cand.trigger_id}: {sup_reason}")
                return None

            # Get prior body for anti-repetition check
            prior_body = conv.sent_bodies[-1] if conv.sent_bodies else None

            # Write message (LLM)
            identity = merchant.get("identity", {})
            merchant_name = identity.get("name", "")

            turns = conv.recent_turns(2)

            llm_output = None
            for attempt in range(1, 3):
                raw = self.writer.write(
                    fp, family, merchant_name, is_customer_scope, turns, attempt
                )
                vr = validate(raw, fp, prior_body)
                if vr.passed:
                    llm_output = raw
                    break
                logger.warning(f"[tick] Validation failed (attempt {attempt}): {vr.failures}")

            if not llm_output:
                # Template fallback
                llm_output = self.writer._template_fallback(fp, family, merchant_name, is_customer_scope)
                vr2 = validate(llm_output, fp, prior_body)
                if not vr2.passed:
                    logger.error(f"[tick] Template fallback also failed validation: {vr2.failures}")
                    return None  # Don't send invalid messages

            body = llm_output["body"]
            cta = llm_output.get("cta", family.preferred_cta)
            rationale = llm_output.get("rationale", "")

            # Record in conversation and suppression
            conv.add_bot_body(body)
            conv.trigger_id = cand.trigger_id
            conv.after_send()

            self.suppression.record_send(
                suppression_key=suppression_key,
                merchant_id=merchant_id,
                body=body,
                customer_id=customer_id,
                trigger_id=cand.trigger_id,
            )

            # Build action response
            owner = identity.get("owner_first_name", identity.get("name", ""))
            locality = identity.get("locality", "")

            # Track which model was used for this message
            model_used = getattr(self.writer, '_last_model_used', 'unknown')

            action = {
                "conversation_id": conv_id,
                "merchant_id": merchant_id,
                "customer_id": customer_id,
                "send_as": send_as,
                "trigger_id": cand.trigger_id,
                "template_name": f"vera_{kind}_v1",
                "template_params": [owner, locality, kind],
                "body": body,
                "cta": cta,
                "suppression_key": suppression_key,
                "rationale": rationale,
                "model_used": model_used,
            }
            return action

        except Exception as e:
            logger.error(f"[tick] Error composing action for {cand.trigger_id}: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Internal: compose reply
    # ------------------------------------------------------------------

    def _compose_reply(self, conv, intent: str, message: str,
                        merchant_id: str, customer_id: Optional[str]) -> dict:
        """Compose a reply to an incoming message."""
        try:
            # Resolve contexts from the conversation
            merchant = self.ctx.get_merchant(merchant_id)
            if not merchant:
                return {
                    "action": "send",
                    "body": "Thanks for your message! I'll look into this and get back to you.",
                    "cta": "none",
                    "rationale": "No merchant context — generic fallback",
                }

            category = self.ctx.get_category_for_merchant(merchant_id)
            customer = self.ctx.get_customer(customer_id) if customer_id else None

            # Find the relevant trigger for this conversation
            trigger_id = conv.trigger_id
            trigger = self.ctx.get_trigger(trigger_id) if trigger_id else None

            if not trigger:
                # Create a minimal trigger context for the reply
                trigger = {"id": "reply_trigger", "kind": "active_planning_intent",
                           "payload": {"intent_topic": "merchant_response"}}

            kind = trigger.get("kind", "active_planning_intent")
            family = get_family(kind)

            # For acceptance - switch to action_ready mode
            if intent == "accept":
                kind = "active_planning_intent"
                family = get_family(kind)

            fp = self.fact_builder.build(trigger, merchant, category or {}, customer)

            merchant_name = merchant.get("identity", {}).get("name", "")
            turns = conv.recent_turns(3)

            llm_output = self.writer.write_reply(
                intent=intent,
                fact_packet=fp,
                family=family,
                merchant_name=merchant_name,
                merchant_message=message,
                is_customer_scope=customer is not None,
                conv_turns=turns,
            )

            body = llm_output.get("body", "")
            cta = llm_output.get("cta", "open_ended")
            rationale = llm_output.get("rationale", "")

            # Verify action mode on acceptance
            if intent == "accept":
                body_lower = body.lower()
                actioning_words = ["done", "sending", "draft", "here", "confirm", "proceed", "next", "setting up"]
                qualifying_words = ["would you", "do you", "can you tell", "what if", "how about"]
                if not any(w in body_lower for w in actioning_words) or any(w in body_lower for w in qualifying_words):
                    owner = merchant.get("identity", {}).get("owner_first_name", "")
                    salutation = f"Dr. {owner}" if category and category.get("slug") == "dentists" else (owner or merchant_name)
                    body = f"Done, {salutation}! Proceeding with the setup now. Next step is ready for your confirmation."

            # Anti-repetition check
            if conv.is_duplicate_body(body):
                body = self._vary_body(body)

            conv.add_bot_body(body)
            conv.add_turn("vera", body)
            conv.after_send()

            if conv.state == ConvState.COMPLETED:
                return {"action": "send", "body": body, "cta": cta, "rationale": rationale}

            return {"action": "send", "body": body, "cta": cta, "rationale": rationale}

        except Exception as e:
            logger.error(f"[reply] Error: {e}", exc_info=True)
            return {
                "action": "send",
                "body": "Got it! Is there anything specific you'd like me to help with?",
                "cta": "open_ended",
                "rationale": "Error in reply composition — generic fallback",
            }

    def _vary_body(self, body: str) -> str:
        """Slightly vary body to avoid exact repetition."""
        return body + " Let me know how you'd like to proceed."
