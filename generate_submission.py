#!/usr/bin/env python3
"""
Generate submission.jsonl for the 30 canonical test pairs in dataset/expanded/test_pairs.json.
Uses the Vera Orchestrator engine to compose, ground, and validate all 30 messages.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from vera.engine.context_store import ContextStore
from vera.engine.fact_builder import FactPacketBuilder
from vera.engine.llm_writer import LLMWriter
from vera.engine.message_families import get_family
from vera.engine.validator import validate

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_all_contexts(expanded_dir: Path, ctx: ContextStore) -> None:
    # 1. Categories
    cat_dir = expanded_dir / "categories"
    if cat_dir.exists():
        for p in cat_dir.glob("*.json"):
            with open(p) as f:
                data = json.load(f)
                slug = data.get("slug", p.stem)
                ctx.put("category", slug, 1, data)

    # 2. Merchants
    m_dir = expanded_dir / "merchants"
    if m_dir.exists():
        for p in m_dir.glob("*.json"):
            with open(p) as f:
                data = json.load(f)
                mid = data.get("merchant_id", p.stem)
                ctx.put("merchant", mid, 1, data)

    # 3. Customers
    c_dir = expanded_dir / "customers"
    if c_dir.exists():
        for p in c_dir.glob("*.json"):
            with open(p) as f:
                data = json.load(f)
                cid = data.get("customer_id", p.stem)
                ctx.put("customer", cid, 1, data)

    # 4. Triggers
    t_dir = expanded_dir / "triggers"
    if t_dir.exists():
        for p in t_dir.glob("*.json"):
            with open(p) as f:
                data = json.load(f)
                tid = data.get("id", p.stem)
                ctx.put("trigger", tid, 1, data)


def main():
    root = Path(__file__).parent
    expanded_dir = root / "dataset" / "expanded"
    test_pairs_file = expanded_dir / "test_pairs.json"
    output_file = root / "submission.jsonl"

    if not test_pairs_file.exists():
        logger.error(f"test_pairs.json not found at {test_pairs_file}. Run generate_dataset.py first.")
        return

    ctx = ContextStore()
    fact_builder = FactPacketBuilder()
    writer = LLMWriter(
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
    )

    logger.info("Loading expanded contexts into ContextStore...")
    load_all_contexts(expanded_dir, ctx)

    with open(test_pairs_file) as f:
        pairs_data = json.load(f)
    pairs = pairs_data.get("pairs", [])

    logger.info(f"Processing {len(pairs)} canonical test pairs concurrently...")
    
    def process_pair(item: dict) -> dict:
        test_id = item["test_id"]
        tid = item["trigger_id"]
        mid = item["merchant_id"]
        cid = item.get("customer_id")

        trigger = ctx.get_trigger(tid) or {}
        merchant = ctx.get_merchant(mid) or {}
        customer = ctx.get_customer(cid) if cid else None
        category = ctx.get_category_for_merchant(mid) or {}

        kind = trigger.get("kind", "unknown")
        family = get_family(kind)
        is_customer_scope = customer is not None
        send_as = family.effective_send_as(is_customer_scope)
        suppression_key = trigger.get("suppression_key", f"sup_{tid}")

        fp = fact_builder.build(trigger, merchant, category, customer)
        merchant_name = merchant.get("identity", {}).get("name", "")

        # Generate message with LLM
        output = writer.write(fp, family, merchant_name, is_customer_scope)
        vr = validate(output, fp)
        if not vr.passed:
            logger.warning(f"[{test_id}] Initial validation warning: {vr.failures}; using fallback")
            output = writer._template_fallback(fp, family, merchant_name, is_customer_scope)

        body = output.get("body", "").strip()
        cta = output.get("cta", family.preferred_cta)
        rationale = output.get("rationale", f"Grounded message for {kind}")

        entry = {
            "test_id": test_id,
            "body": body,
            "cta": cta,
            "send_as": send_as,
            "suppression_key": suppression_key,
            "rationale": rationale,
        }
        logger.info(f"Generated {test_id}: {body[:60]}...")
        return entry

    import concurrent.futures
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {executor.submit(process_pair, p): p for p in pairs}
        for future in concurrent.futures.as_completed(future_map):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                p = future_map[future]
                logger.error(f"Error processing {p.get('test_id')}: {e}")

    # Keep original order by test_id
    results.sort(key=lambda r: r["test_id"])

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Successfully wrote {len(results)} lines to {output_file}")


if __name__ == "__main__":
    main()
