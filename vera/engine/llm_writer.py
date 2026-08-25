"""
LLM Writer using Google Gemini (new google-genai SDK).

The LLM ONLY writes messages. It does NOT:
- decide whether to send
- select triggers
- override suppression/consent/expiry
- invent facts

Input: structured fact packet + action selection + category voice + CTA type
Output: {"body": str, "cta": str, "rationale": str, "used_fact_ids": [...]}
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Dict, List, Optional

from vera.engine.fact_builder import FactPacket
from vera.engine.message_families import MessageFamily
from vera.engine.key_rotator import key_pool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the message-writing component of Vera, magicpin's merchant AI assistant.

You do NOT decide whether to send a message.
You do NOT select triggers.
You do NOT override suppression, consent, expiry, or conversation state.

Your ONLY task is to write ONE concise WhatsApp message from the verified facts supplied.

RULES:
1. Use ONLY the facts supplied in the fact packet. Never invent numbers, dates, names, offers, competitor names, citations, or urgency.
2. Use the requested category voice and tone.
3. Match the merchant/customer's language preference (Hindi-English code-mix is fine and often preferred).
4. Include EXACTLY ONE primary CTA as specified - do not give multiple choices like "Reply YES for X, NO for Y, MAYBE for Z".
5. Do not reveal internal system terminology (no "trigger", "context", "signal", "category slug", etc.).
6. Do not repeat previous conversation wording verbatim.
7. Do not re-introduce yourself if this is not the first message.
8. Keep it concise - WhatsApp messages, not essays.
9. No promotional exclamations ("AMAZING DEAL!"). Match the business tone.
10. If facts are insufficient, write a restrained, honest message. Do not fabricate.
11. Use the salutation style from the category (e.g., "Dr. {first_name}" for dentists).
12. For customer-facing messages: no medical guarantees, no made-up prices.

OUTPUT FORMAT (JSON only, no markdown):
{
  "body": "<the WhatsApp message>",
  "cta": "<yes_no|open_ended|none>",
  "rationale": "<1-2 sentences: what facts you used and why this message achieves the objective>",
  "used_fact_ids": ["f1", "f3", ...]
}"""


class GlobalRateLimiter:
    """
    Thread-safe rate limiter that respects Google Gemini free-tier limits:
      - 15 RPM (requests per minute)
      - 1500 RPD (requests per day)

    When a 429 is received, we park until Retry-After elapses instead of
    burning another quota slot immediately.
    """

    def __init__(self, rpm: int = 15, rpd: int = 1500) -> None:
        self._rpm = rpm
        self._rpd = rpd
        self._lock = threading.Lock()
        self._minute_slots: List[float] = []   # timestamps of calls in last 60s
        self._day_slots: List[float] = []       # timestamps of calls in last 24h
        self._backoff_until: float = 0.0        # epoch time to park until after 429

    def acquire(self, timeout: float = 120.0) -> bool:
        """Block until a slot is available or timeout expires. Returns True if acquired."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                now = time.monotonic()
                wall = time.time()

                # Respect 429 backoff period
                if now < self._backoff_until:
                    wait = self._backoff_until - now
                else:
                    # Prune stale slots
                    self._minute_slots = [t for t in self._minute_slots if wall - t < 60.0]
                    self._day_slots = [t for t in self._day_slots if wall - t < 86400.0]

                    minute_ok = len(self._minute_slots) < self._rpm
                    day_ok = len(self._day_slots) < self._rpd

                    if minute_ok and day_ok:
                        self._minute_slots.append(wall)
                        self._day_slots.append(wall)
                        return True

                    # Calculate how long until a slot frees up
                    if not minute_ok:
                        wait = max(0.0, 60.0 - (wall - self._minute_slots[0]) + 0.1)
                    else:
                        # Day quota exhausted — this is a hard block
                        logger.warning("[RateLimiter] Day quota exhausted — template fallback forced")
                        return False

            time.sleep(min(wait, 5.0))
        return False

    def record_429(self, retry_after_seconds: float) -> None:
        """Record a 429 and park all threads for retry_after_seconds."""
        with self._lock:
            backoff_end = time.monotonic() + retry_after_seconds
            self._backoff_until = max(self._backoff_until, backoff_end)
            # Don't count this failed call against quota
            wall = time.time()
            if self._minute_slots and abs(self._minute_slots[-1] - wall) < 2.0:
                self._minute_slots.pop()
            if self._day_slots and abs(self._day_slots[-1] - wall) < 2.0:
                self._day_slots.pop()
        logger.warning(f"[RateLimiter] 429 received — backing off {retry_after_seconds:.0f}s")

    def remaining_day_quota(self) -> int:
        with self._lock:
            wall = time.time()
            self._day_slots = [t for t in self._day_slots if wall - t < 86400.0]
            return max(0, self._rpd - len(self._day_slots))


CACHE_FILE = Path(__file__).resolve().parent.parent.parent / ".cache" / "llm_cache.json"

def _load_disk_cache() -> Dict[str, dict]:
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as e:
        logger.warning(f"Could not load disk cache: {e}")
    return {}

def _save_disk_cache(cache: Dict[str, dict]) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Could not save disk cache: {e}")


# Singleton rate limiter shared across all LLMWriter instances
_rate_limiter = GlobalRateLimiter(rpm=4, rpd=1500)


class LLMWriter:
    """Multi-key rotating Gemini message writer with automated failover and fallback."""

    def __init__(self, api_key: Optional[str] = None, model: str = "") -> None:
        self.model_name = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        self._last_model_used: str = "template_fallback"
        self._cache: Dict[str, dict] = _load_disk_cache()  # persistent cache
        # Reload keys in pool if needed
        key_pool.load_keys()

    @property
    def _model_available(self) -> bool:
        return key_pool.total_keys > 0 or bool(os.environ.get("OPENROUTER_API_KEY"))

    def write(
        self,
        fact_packet: FactPacket,
        family: MessageFamily,
        merchant_name: str,
        is_customer_scope: bool = False,
        conv_turns: Optional[list] = None,
        attempt: int = 1,
    ) -> dict:
        """
        Write a WhatsApp message.
        Returns: {"body": str, "cta": str, "rationale": str, "used_fact_ids": [...]}
        """
        cache_key = self._cache_key(fact_packet, family, attempt)
        if cache_key in self._cache:
            return dict(self._cache[cache_key])

        if not self._model_available:
            return self._template_fallback(fact_packet, family, merchant_name, is_customer_scope)

        prompt = self._build_prompt(fact_packet, family, merchant_name,
                                    is_customer_scope, conv_turns, attempt)

        result = self._call_llm_with_rotation(prompt)
        if result:
            self._cache[cache_key] = result
            _save_disk_cache(self._cache)
            return dict(result)

        # Fallback to template if all keys exhausted/failed
        return self._template_fallback(fact_packet, family, merchant_name, is_customer_scope)

    def write_reply(
        self,
        intent: str,
        fact_packet: FactPacket,
        family: MessageFamily,
        merchant_name: str,
        merchant_message: str,
        is_customer_scope: bool = False,
        conv_turns: Optional[list] = None,
    ) -> dict:
        """Write a reply to an incoming merchant/customer message."""
        if not self._model_available:
            return self._reply_template_fallback(intent, merchant_name)

        intent_instruction = {
            "accept": (
                "The merchant/customer just ACCEPTED. Switch to direct ACTION mode immediately. "
                "Do NOT ask any qualifying questions (NEVER use 'would you', 'do you want', 'can you tell', 'what if', 'how about'). "
                "Confirm and proceed with the action directly using words like 'Done', 'Proceeding', 'Here is the draft', 'Next step'."
            ),
            "defer": "The merchant asked for more time. Acknowledge gracefully and say you'll follow up later. Be brief.",
            "normal": "The merchant sent a normal message. Respond helpfully and advance toward your objective.",
            "hostile": "The merchant sent a hostile message. Apologize politely and exit.",
        }.get(intent, "Respond appropriately to the merchant's message.")

        prompt = f"""INTENT: {intent}
INSTRUCTION: {intent_instruction}

MERCHANT/CUSTOMER SAID: "{merchant_message}"

FACTS AVAILABLE:
{self._format_facts(fact_packet)}

OBJECTIVE: {family.objective}
PREFERRED CTA: {family.preferred_cta} — {family.cta_hint}
CATEGORY VOICE: {', '.join(family.tone_hints)}
MAX LENGTH: {family.max_length_chars} characters

RECENT CONVERSATION:
{self._format_turns(conv_turns or [])}

Write the reply now. Remember: if they accepted, take action immediately.
Output JSON only."""

        result = self._call_llm_with_rotation(prompt)
        if result:
            return result

        return self._reply_template_fallback(intent, merchant_name)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_prompt(self, fp: FactPacket, family: MessageFamily,
                      merchant_name: str, is_customer_scope: bool,
                      conv_turns: Optional[list], attempt: int) -> str:
        strictness = ""
        if attempt > 1:
            strictness = "\nSTRICT MODE: Avoid any claim not directly stated in the facts below. If uncertain, omit."

        scope_note = ""
        if is_customer_scope:
            scope_note = "\nSCOPE: Customer-facing message. Send as the merchant to their customer. No medical claims. Warm but professional."

        return f"""TASK: Write ONE WhatsApp message.

TRIGGER KIND: {fp.primary_signal}
OBJECTIVE: {family.objective}
CTA TYPE: {family.preferred_cta} — {family.cta_hint}
TONE: {', '.join(family.tone_hints)}
MAX LENGTH: {family.max_length_chars} characters
MERCHANT: {merchant_name}
{scope_note}{strictness}

VERIFIED FACTS (use only these):
{self._format_facts(fp)}

FORBIDDEN WORDS/CLAIMS:
{', '.join(fp.forbidden_claims[:10]) if fp.forbidden_claims else 'None'}

RECENT CONVERSATION (for context only — do not repeat):
{self._format_turns(conv_turns or [])}

Write the message now. Output JSON only."""

    def _format_facts(self, fp: FactPacket) -> str:
        lines = []
        for f in fp.facts:
            lines.append(f"  [{f.id}] {f.text}  (source: {f.source})")
        return "\n".join(lines) if lines else "  (no facts available)"

    def _format_turns(self, turns: list) -> str:
        if not turns:
            return "  (no prior turns)"
        lines = [f"  {t.get('from', '?')}: {t.get('msg', '')[:100]}" for t in turns[-2:]]
        return "\n".join(lines)

    def _call_llm_with_rotation(self, prompt: str) -> Optional[dict]:
        """
        Call Gemini using automatic multi-key and multi-model rotation/failover.
        If a key/model hits 429 or quota, marks cooldown and tries the next model/key.
        """
        # pyrefly: ignore [missing-import]
        from google.genai import types

        total_keys = key_pool.total_keys
        max_attempts = max(1, total_keys)

        candidate_models = []
        for m in [self.model_name, "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview", "gemini-3.5-flash"]:
            if m and m not in candidate_models:
                candidate_models.append(m)

        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

        # Try rotating through available Gemini keys and candidate models
        for attempt in range(max_attempts):
            client, key_state = key_pool.get_client_and_key()
            if not client or not key_state:
                break

            for model_id in candidate_models:
                # Pacing to avoid hitting 5 RPM free tier limits
                _rate_limiter.acquire(timeout=60.0)

                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            response_mime_type="application/json",
                            max_output_tokens=4096,
                        ),
                    )
                    text = response.text.strip() if response.text else ""
                    if text:
                        key_state.mark_success()
                        parsed = self._parse_json_response(text)
                        if parsed:
                            self._last_model_used = model_id
                            return parsed

                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        logger.info(
                            f"[LLMWriter] Model {model_id} hit quota. Trying next candidate model..."
                        )
                        continue  # Try next model!
                    else:
                        logger.warning(f"[LLMWriter] Call failed with model {model_id} (Key #{key_state.index}): {e}")
                        continue

        # Backup: Try OpenRouter if configured
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if openrouter_key:
            logger.info("[LLMWriter] Attempting backup call via OpenRouter...")
            res = self._call_openrouter(full_prompt, openrouter_key)
            if res:
                return res

        return None

    def _call_openrouter(self, full_prompt: str, api_key: str) -> Optional[dict]:
        """Secondary fallback via OpenRouter API with multi-model automatic rotation."""
        from urllib import request as urlrequest
        openrouter_models = [
            "google/gemini-2.0-flash-lite-preview-02-05:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-chat:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "google/gemini-flash-1.5:free",
            "mistralai/mistral-small-24b-instruct-2501:free",
            "google/gemini-2.0-flash-thinking-exp:free",
            "deepseek/deepseek-r1:free",
            "openrouter/auto",
        ]
        for model_name in openrouter_models:
            try:
                body = json.dumps({
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": full_prompt}
                    ],
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "response_format": {"type": "json_object"}
                }).encode("utf-8")

                req = urlrequest.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=body,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://magicpin.in",
                        "X-Title": "magicpin AI Vera"
                    }
                )
                resp = urlrequest.urlopen(req, timeout=15)
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                parsed = self._parse_json_response(content)
                if parsed:
                    logger.info(f"[LLMWriter] OpenRouter success with model {model_name}")
                    self._last_model_used = f"openrouter/{model_name}"
                    return parsed
            except Exception as e:
                logger.warning(f"[LLMWriter] OpenRouter model {model_name} failed: {e}")
                continue
        return None

    def _parse_json_response(self, text: str) -> Optional[dict]:
        """Extract and validate JSON from model output."""
        # Strategy 1: Direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return {
                    "body": str(data.get("body", "")),
                    "cta": str(data.get("cta", "open_ended")),
                    "rationale": str(data.get("rationale", "")),
                    "used_fact_ids": list(data.get("used_fact_ids", [])),
                }
        except Exception:
            pass

        # Strategy 2: Strip markdown code blocks
        clean_text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        clean_text = re.sub(r'\s*```$', '', clean_text, flags=re.MULTILINE).strip()
        try:
            data = json.loads(clean_text)
            if isinstance(data, dict):
                return {
                    "body": str(data.get("body", "")),
                    "cta": str(data.get("cta", "open_ended")),
                    "rationale": str(data.get("rationale", "")),
                    "used_fact_ids": list(data.get("used_fact_ids", [])),
                }
        except Exception:
            pass

        # Strategy 3: Regex extract JSON object
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if isinstance(data, dict):
                    return {
                        "body": str(data.get("body", "")),
                        "cta": str(data.get("cta", "open_ended")),
                        "rationale": str(data.get("rationale", "")),
                        "used_fact_ids": list(data.get("used_fact_ids", [])),
                    }
            except Exception as e:
                logger.warning(f"JSON regex parse failed: {e}, text={text[:200]}")

        logger.warning(f"LLM output could not be parsed as JSON: {text[:200]}")
        return None

    def _cache_key(self, fp: FactPacket, family: MessageFamily, attempt: int) -> str:
        content = json.dumps({
            "signal": fp.primary_signal,
            "facts": [f.text for f in fp.facts],
            "family": family.kind,
            "cta": family.preferred_cta,
            "attempt": attempt,
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _template_fallback(self, fp: FactPacket, family: MessageFamily,
                            merchant_name: str, is_customer_scope: bool) -> dict:
        """Safe template-based fallback when LLM is unavailable."""
        # Extract key facts
        owner_fact = next((f for f in fp.facts
                           if "owner_first_name" in f.source.lower()
                           or "Owner first name" in f.text), None)
        name = owner_fact.text.replace("Owner first name: ", "") if owner_fact else merchant_name

        kind = fp.primary_signal
        cta = family.preferred_cta

        if kind == "research_digest":
            body = f"Dr. {name}, there's a new research finding relevant to your practice. Would you like me to share the details?"
        elif kind == "renewal_due":
            days_fact = next((f for f in fp.facts if "Subscription expires" in f.text), None)
            days_info = days_fact.text if days_fact else "soon"
            body = f"{name}, your magicpin subscription is expiring — {days_info}. Reply YES to continue and keep your profile active."
        elif kind == "perf_dip":
            metric_fact = next((f for f in fp.facts
                                 if "performance" in f.source.lower()
                                 and ("views" in f.text.lower() or "calls" in f.text.lower())), None)
            metric_info = metric_fact.text if metric_fact else "your profile metrics"
            body = f"{name}, I noticed a dip in {metric_info}. Want me to suggest what we can do to improve it?"
        elif kind == "curious_ask_due":
            body = f"Quick question, {name} — what's the most popular service at your place this week?"
        elif kind == "recall_due":
            cust_fact = next((f for f in fp.facts if "Customer name" in f.text), None)
            cust_name = cust_fact.text.replace("Customer name: ", "") if cust_fact else "there"
            body = f"Hi {cust_name}, it's time for your follow-up visit. Would you like to book an appointment?"
        elif kind == "milestone_reached":
            body = f"🎉 {name}, you're almost at a big milestone! Want to know the details?"
        elif kind == "regulation_change":
            body = f"{name}, there's a new regulatory update that may affect your practice. Want me to share what action you need to take?"
        elif kind == "festival_upcoming":
            body = f"{name}, there's a festival coming up soon — want me to set up a special campaign for your business?"
        elif kind == "winback_eligible":
            body = f"{name}, I noticed your magicpin subscription lapsed. Your profile is missing out on visibility — want to reconnect?"
        else:
            body = f"{name}, quick update from Vera — want to know what's happening with your profile?"

        self._last_model_used = "template_fallback"
        return {
            "body": body,
            "cta": cta,
            "rationale": f"Template fallback for {kind} — LLM unavailable",
            "used_fact_ids": [],
        }

    def _reply_template_fallback(self, intent: str, merchant_name: str) -> dict:
        self._last_model_used = "template_fallback"
        if intent == "accept":
            return {
                "body": "Great! I'll get that set up for you right now. Give me a moment.",
                "cta": "none",
                "rationale": "Merchant accepted; moving to action immediately",
                "used_fact_ids": [],
            }
        elif intent == "defer":
            return {
                "body": "No problem at all! I'll check back with you later. Take your time.",
                "cta": "none",
                "rationale": "Merchant deferred; backing off gracefully",
                "used_fact_ids": [],
            }
        else:
            return {
                "body": "Got it! Is there anything specific I can help you with right now?",
                "cta": "open_ended",
                "rationale": "Generic helpful reply",
                "used_fact_ids": [],
            }
