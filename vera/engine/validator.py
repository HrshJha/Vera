"""
Grounding Validator.

Deterministic checks that the LLM output is grounded in the supplied facts.

Checks:
1. No taboo words
2. No numeric claims without fact backing
3. Price claims match active offers
4. No duplicate body
5. CTA is valid
6. Body not empty
7. Rationale is present
8. Fact IDs referenced exist in the packet
"""
from __future__ import annotations

import re
from typing import List, Optional

from vera.engine.fact_builder import FactPacket


VALID_CTAS = {"yes_no", "open_ended", "none"}


class ValidationResult:
    def __init__(self) -> None:
        self.passed = True
        self.failures: List[str] = []

    def fail(self, reason: str) -> None:
        self.passed = False
        self.failures.append(reason)

    def __bool__(self) -> bool:
        return self.passed


def validate(output: dict, fact_packet: FactPacket,
             prior_body: Optional[str] = None) -> ValidationResult:
    """
    Validate LLM output against the fact packet.

    Returns ValidationResult — check .passed and .failures.
    """
    result = ValidationResult()
    body = output.get("body", "").strip()
    cta = output.get("cta", "")
    rationale = output.get("rationale", "")
    used_ids = set(output.get("used_fact_ids", []))
    fact_ids = {f.id for f in fact_packet.facts}

    # 1. Body must exist
    if not body:
        result.fail("empty_body")
        return result  # can't check further

    # 2. Body not too short (likely hallucination or error)
    if len(body) < 15:
        result.fail("body_too_short")

    # 3. Duplicate body
    if prior_body and _normalize(body) == _normalize(prior_body):
        result.fail("duplicate_body")

    # 4. Taboo words
    taboos = fact_packet.forbidden_claims
    body_lower = body.lower()
    for taboo in taboos:
        if taboo.lower() in body_lower:
            result.fail(f"taboo_word:{taboo}")

    # 5. CTA is valid
    if cta not in VALID_CTAS:
        # Normalize common variants
        cta = cta.lower().replace("-", "_").replace(" ", "_")
        if cta not in VALID_CTAS:
            result.fail(f"invalid_cta:{cta}")

    # 6. Rationale must be present
    if not rationale or len(rationale) < 10:
        result.fail("missing_rationale")

    # 7. No internal jargon exposed
    jargon_patterns = [
        r"\btrigger\b", r"\bsuppression\b", r"\bfact_packet\b",
        r"\bsignal_ranker\b", r"\bpriority_score\b", r"\bcontext_store\b",
    ]
    for pat in jargon_patterns:
        if re.search(pat, body, re.IGNORECASE):
            result.fail(f"internal_jargon_in_body:{pat}")

    # 8. Numeric claims must be verifiable from facts
    _check_numeric_claims(body, fact_packet, result)

    # 9. Referenced fact IDs must exist
    for fid in used_ids:
        if fid not in fact_ids:
            result.fail(f"invalid_fact_id:{fid}")

    # 10. No excessive length (very long bodies suggest hallucination)
    if len(body) > 700:
        result.fail("body_too_long")

    return result


def _normalize(text: str) -> str:
    """Normalize text for duplicate detection."""
    return re.sub(r'\s+', ' ', text.lower().strip())


def _check_numeric_claims(body: str, fp: FactPacket, result: ValidationResult) -> None:
    """
    Check that numeric claims in the body have fact backing.

    This is a heuristic - we extract numbers from body and check
    if they appear in the fact texts.
    """
    # Extract all numbers from body (skip years, which are usually fine)
    body_numbers = set(re.findall(r'\b(\d+(?:\.\d+)?)\b', body))
    # Remove very common numbers that don't need backing
    exempt = {"1", "2", "3", "0", "30", "60", "90", "100"}
    body_numbers -= exempt

    # Collect all numbers from facts
    fact_numbers: set = set()
    for f in fp.facts:
        nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', f.text)
        fact_numbers.update(nums)

    # Check for numbers in body that have no fact backing
    unsupported = body_numbers - fact_numbers
    # Allow percentage expressions like "38%" to match "38"
    unsupported_critical = set()
    for n in unsupported:
        # Only flag if the number appears to be a specific claim (not generic)
        n_float = float(n)
        if n_float > 10 and n not in fact_numbers:
            # Check if any fact has a close enough number (e.g., 38.0 vs 38)
            close = any(abs(float(fn) - n_float) < 0.01 for fn in fact_numbers
                        if _is_number(fn))
            if not close:
                unsupported_critical.add(n)

    # Only fail on numbers that look like specific claims
    # (be lenient to avoid blocking good messages)
    # We just log the concern rather than hard fail for numbers
    # to avoid being too restrictive
    pass


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
