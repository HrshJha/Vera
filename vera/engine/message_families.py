"""
Message family dispatch and CTA configuration.

Each trigger kind maps to a message family with:
- objective
- preferred CTA
- tone hints
- customer/merchant framing
- urgency behavior
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MessageFamily:
    kind: str
    objective: str
    preferred_cta: str                    # "yes_no", "open_ended", "none"
    cta_hint: str                         # what the CTA should say
    tone_hints: List[str] = field(default_factory=list)
    framing: str = "merchant"             # "merchant" or "customer"
    allow_emoji: bool = True
    max_length_chars: int = 350
    # For customer-facing messages
    send_as: str = "vera"                 # "vera" or "merchant_on_behalf"

    def effective_send_as(self, is_customer_scope: bool) -> str:
        if is_customer_scope:
            return "merchant_on_behalf"
        return "vera"


_FAMILIES: Dict[str, MessageFamily] = {
    "research_digest": MessageFamily(
        kind="research_digest",
        objective="Share relevant research finding that could impact merchant's patients/customers",
        preferred_cta="open_ended",
        cta_hint="Offer to pull abstract or draft patient-ed content",
        tone_hints=["peer_clinical", "cite_source", "anchor_on_specific_cohort",
                    "low_hype", "offer_to_do_work"],
        allow_emoji=False,
        max_length_chars=320,
    ),
    "regulation_change": MessageFamily(
        kind="regulation_change",
        objective="Alert merchant to regulatory compliance requirement with deadline and propose immediate audit assistance",
        preferred_cta="yes_no",
        cta_hint="Offer to audit and document compliance (e.g. 'Reply YES to audit your setup and update SOPs')",
        tone_hints=["urgent_but_calm", "peer_clinical", "cite_source", "specific_deadline",
                    "actionable_step", "single_binary_cta"],
        allow_emoji=False,
        max_length_chars=300,
    ),
    "recall_due": MessageFamily(
        kind="recall_due",
        objective="Remind the merchant's customer that a service recall is due",
        preferred_cta="yes_no",
        cta_hint="Offer specific time slots to book",
        tone_hints=["warm_personal", "specific_slots", "gentle_urgency",
                    "match_customer_language"],
        framing="customer",
        send_as="merchant_on_behalf",
        allow_emoji=True,
        max_length_chars=280,
    ),
    "perf_dip": MessageFamily(
        kind="perf_dip",
        objective="Alert merchant to a significant performance drop and offer concrete help",
        preferred_cta="yes_no",
        cta_hint="Offer specific action to address the dip",
        tone_hints=["data_led", "loss_aversion", "specific_metric",
                    "concrete_next_step"],
        allow_emoji=False,
        max_length_chars=300,
    ),
    "perf_spike": MessageFamily(
        kind="perf_spike",
        objective="Celebrate performance spike with exact metrics and capitalize on momentum",
        preferred_cta="yes_no",
        cta_hint="Offer to extend or pin the trending offer (e.g. 'Reply YES to keep your top offer active')",
        tone_hints=["positive", "momentum_driven", "specific_metric", "actionable_next_step"],
        allow_emoji=True,
        max_length_chars=280,
    ),
    "renewal_due": MessageFamily(
        kind="renewal_due",
        objective="Prompt subscription renewal before expiry with business continuity reassurance",
        preferred_cta="yes_no",
        cta_hint="Offer 1-click renewal confirmation (e.g. 'Reply YES to renew and maintain visibility')",
        tone_hints=["value_reminder", "loss_aversion", "specific_days",
                    "business_continuity"],
        allow_emoji=False,
        max_length_chars=280,
    ),
    "festival_upcoming": MessageFamily(
        kind="festival_upcoming",
        objective="Capitalize on upcoming festival footfall by preparing an early-booking festive package/campaign tailored to the merchant's category",
        preferred_cta="yes_no",
        cta_hint="Offer a single concrete next step (e.g. 'Reply YES to preview an early festive package draft')",
        tone_hints=["festive_practical", "early_bird_advantage", "category_aligned",
                    "loss_aversion_footfall", "single_binary_cta"],
        allow_emoji=True,
        max_length_chars=300,
    ),
    "wedding_package_followup": MessageFamily(
        kind="wedding_package_followup",
        objective="Follow up on bridal/wedding package with specific next step",
        preferred_cta="yes_no",
        cta_hint="Offer to lock in the next preparation slot (e.g. 'Reply YES to confirm your slot')",
        tone_hints=["warm", "relationship_continuity", "wedding_countdown",
                    "specific_date_preference"],
        framing="customer",
        send_as="merchant_on_behalf",
        allow_emoji=True,
        max_length_chars=280,
    ),
    "curious_ask_due": MessageFamily(
        kind="curious_ask_due",
        objective="Ask the merchant an engaging, low-friction question about their weekly customer demand to tailor upcoming campaigns",
        preferred_cta="open_ended",
        cta_hint="Ask a specific, intriguing question about their weekly trending service or customer favorite",
        tone_hints=["conversational", "curious", "low_ask", "genuine_interest"],
        allow_emoji=True,
        max_length_chars=180,
    ),
    "winback_eligible": MessageFamily(
        kind="winback_eligible",
        objective="Win back an expired merchant by showing exact missed customer counts and proposing a 1-click restoration path",
        preferred_cta="yes_no",
        cta_hint="Offer seamless listing restoration (e.g. 'Reply YES to restore your active listing')",
        tone_hints=["value_reminder", "loss_aversion", "lapsed_customers",
                    "what_you_re_missing", "single_binary_cta"],
        allow_emoji=False,
        max_length_chars=320,
    ),
    "ipl_match_today": MessageFamily(
        kind="ipl_match_today",
        objective="Alert food/restaurant merchant to match-day footfall/delivery surge and launch an immediate match-hour special",
        preferred_cta="yes_no",
        cta_hint="Offer to post a match-day combo special on Google Profile (e.g. 'Reply YES to push the match combo offer')",
        tone_hints=["timely", "local_angle", "match_audience", "food_relevant", "single_binary_cta"],
        allow_emoji=True,
        max_length_chars=260,
    ),
    "review_theme_emerged": MessageFamily(
        kind="review_theme_emerged",
        objective="Alert merchant to a review pattern and offer to address it",
        preferred_cta="yes_no",
        cta_hint="Offer to help draft an operational response strategy (e.g. 'Reply YES to review the action plan')",
        tone_hints=["non_alarming", "data_led", "constructive", "actionable"],
        allow_emoji=False,
        max_length_chars=300,
    ),
    "milestone_reached": MessageFamily(
        kind="milestone_reached",
        objective="Celebrate approaching/reached milestone and suggest next step to maintain momentum",
        preferred_cta="yes_no",
        cta_hint="Offer to publish a milestone celebration post to Google Profile (e.g. 'Reply YES to post this update')",
        tone_hints=["celebratory_but_brief", "momentum", "concrete_action", "single_binary_cta"],
        allow_emoji=True,
        max_length_chars=240,
    ),
    "active_planning_intent": MessageFamily(
        kind="active_planning_intent",
        objective="Immediately advance active planning conversation to action",
        preferred_cta="yes_no",
        cta_hint="Provide the plan details and confirm to proceed",
        tone_hints=["action_mode", "concrete", "no_more_qualifying",
                    "merchant_already_said_yes"],
        allow_emoji=False,
        max_length_chars=360,
    ),
    "seasonal_perf_dip": MessageFamily(
        kind="seasonal_perf_dip",
        objective="Acknowledge seasonal dip and suggest a counter-strategy",
        preferred_cta="yes_no",
        cta_hint="Offer a seasonal campaign",
        tone_hints=["acknowledge_normal", "proactive", "category_seasonal"],
        allow_emoji=False,
        max_length_chars=300,
    ),
    "customer_lapsed_soft": MessageFamily(
        kind="customer_lapsed_soft",
        objective="Re-engage a lapsed customer before they churn",
        preferred_cta="yes_no",
        cta_hint="Offer a specific reason to return",
        tone_hints=["warm_reminder", "gentle_nudge", "personalized"],
        framing="customer",
        send_as="merchant_on_behalf",
        allow_emoji=True,
        max_length_chars=260,
    ),
    "customer_lapsed_hard": MessageFamily(
        kind="customer_lapsed_hard",
        objective="Last-attempt re-engagement for a long-lapsed customer",
        preferred_cta="yes_no",
        cta_hint="Offer a compelling reason to return",
        tone_hints=["we_miss_you", "value_offer", "simple_cta"],
        framing="customer",
        send_as="merchant_on_behalf",
        allow_emoji=True,
        max_length_chars=240,
    ),
    "appointment_tomorrow": MessageFamily(
        kind="appointment_tomorrow",
        objective="Remind merchant of upcoming appointments to reduce no-shows",
        preferred_cta="none",
        cta_hint="Informational only",
        tone_hints=["brief", "operational", "specific_count"],
        allow_emoji=False,
        max_length_chars=200,
    ),
    "dormant_with_vera": MessageFamily(
        kind="dormant_with_vera",
        objective="Re-engage a merchant who has been silent with Vera",
        preferred_cta="open_ended",
        cta_hint="Ask a low-friction question",
        tone_hints=["light_touch", "no_pressure", "genuine", "low_ask"],
        allow_emoji=True,
        max_length_chars=200,
    ),
    "gbp_unverified": MessageFamily(
        kind="gbp_unverified",
        objective="Prompt merchant to verify their Google Business Profile",
        preferred_cta="yes_no",
        cta_hint="Offer to guide through verification",
        tone_hints=["specific_benefit", "what_they_gain", "simple_steps"],
        allow_emoji=False,
        max_length_chars=280,
    ),
    "cde_opportunity": MessageFamily(
        kind="cde_opportunity",
        objective="Alert merchant to relevant CE/CDE opportunity",
        preferred_cta="open_ended",
        cta_hint="Offer more details about the event",
        tone_hints=["peer_tone", "professional_development", "credits_mentioned"],
        allow_emoji=False,
        max_length_chars=280,
    ),
    "competitor_opened": MessageFamily(
        kind="competitor_opened",
        objective="Alert merchant to new competition nearby",
        preferred_cta="yes_no",
        cta_hint="Offer to strengthen their positioning",
        tone_hints=["factual", "competitive_response", "no_alarmism"],
        allow_emoji=False,
        max_length_chars=280,
    ),
    "category_seasonal": MessageFamily(
        kind="category_seasonal",
        objective="Highlight a seasonal category trend and suggest action",
        preferred_cta="yes_no",
        cta_hint="Offer a seasonal offer or campaign",
        tone_hints=["seasonal_relevance", "local_angle", "data_led"],
        allow_emoji=True,
        max_length_chars=280,
    ),
}

# Default fallback family
_DEFAULT_FAMILY = MessageFamily(
    kind="default",
    objective="Engage the merchant with relevant information",
    preferred_cta="open_ended",
    cta_hint="Ask a simple open-ended question",
    tone_hints=["friendly", "brief", "practical"],
    allow_emoji=True,
    max_length_chars=300,
)


def get_family(kind: str) -> MessageFamily:
    """Return the message family for a trigger kind."""
    return _FAMILIES.get(kind, _DEFAULT_FAMILY)
