"""
Vera Bot — FastAPI Application.

Implements all 5 required endpoints + /v1/teardown:
  GET  /v1/healthz
  GET  /v1/metadata
  POST /v1/context
  POST /v1/tick
  POST /v1/reply
  POST /v1/teardown
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from vera.engine.orchestrator import Orchestrator

# Load environment variables from .env if present
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Enable DEBUG for ranker to see gate rejections
logging.getLogger("vera.engine.ranker").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Vera orchestrator (single global instance - stateful by design)
orchestrator = Orchestrator()
START_TIME = time.time()

app = FastAPI(
    title="Vera Merchant Growth Message Engine",
    description="magicpin AI Challenge — Vera bot API",
    version="1.0.0",
)


# =============================================================================
# REQUEST MODELS
# =============================================================================

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


# =============================================================================
# ENDPOINTS (Supporting both GET and HEAD for cloud/uptime probes)
# =============================================================================

@app.api_route("/favicon.ico", methods=["GET", "HEAD"], include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/healthz", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok"}


@app.api_route("/v1/healthz", methods=["GET", "HEAD"])
async def healthz():
    """Liveness probe. Returns context counts per scope."""
    counts = orchestrator.ctx.count_by_scope()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    """Bot identity and approach."""
    return {
        "team_name": os.environ.get("TEAM_NAME", "Vera Team"),
        "team_members": os.environ.get("TEAM_MEMBERS", "Developer").split(","),
        "model": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        "approach": (
            "Deterministic signal ranker + LLM writer. "
            "Hard gates control when to send; "
            "Gemini Flash writes grounded messages from verified fact packets."
        ),
        "contact_email": os.environ.get("CONTACT_EMAIL", "team@example.com"),
        "version": "1.0.0",
        "submitted_at": "2026-08-25T00:00:00Z",
    }


@app.get("/v1/keypool")
async def keypool_status():
    """Inspect active LLM API key pool health and statistics."""
    from vera.engine.key_rotator import key_pool
    return {
        "total_keys": key_pool.total_keys,
        "available_keys": key_pool.available_keys_count,
        "keys": key_pool.get_status(),
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    """Receive and store a context push (category, merchant, customer, trigger)."""
    result = orchestrator.ctx.put(
        scope=body.scope,
        context_id=body.context_id,
        version=body.version,
        payload=body.payload,
    )
    if not result.get("accepted"):
        reason = result.get("reason", "unknown")
        if reason == "stale_version":
            return JSONResponse(status_code=409, content=result)
        elif reason in ("invalid_scope", "payload_too_large"):
            return JSONResponse(status_code=400, content=result)
    return result


@app.post("/v1/tick")
async def tick(body: TickBody):
    """
    Periodic wake-up. Bot inspects active triggers and generates proactive messages.
    Returns up to 20 actions.
    """
    try:
        actions = orchestrator.tick(
            now_iso=body.now,
            available_trigger_ids=body.available_triggers,
        )
        return {"actions": actions}
    except Exception as e:
        logger.error(f"[tick] Unhandled error: {e}", exc_info=True)
        # Return empty actions to avoid timeout penalty
        return {"actions": []}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """
    Receive a reply from a simulated merchant/customer.
    Returns the bot's next move: send, wait, or end.
    """
    try:
        result = orchestrator.reply(
            conv_id=body.conversation_id,
            merchant_id=body.merchant_id,
            customer_id=body.customer_id,
            from_role=body.from_role,
            message=body.message,
            turn_number=body.turn_number,
        )
        return result
    except Exception as e:
        logger.error(f"[reply] Unhandled error: {e}", exc_info=True)
        return {
            "action": "send",
            "body": "Thanks for your message! I'll look into this.",
            "cta": "none",
            "rationale": "Error fallback",
        }


@app.post("/v1/teardown")
async def teardown():
    """Wipe all state after test ends."""
    orchestrator.teardown()
    logger.info("[teardown] State wiped.")
    return {"status": "torn_down", "message": "All context and conversation state cleared."}


# =============================================================================
# ROOT
# =============================================================================

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "service": "Vera Merchant Growth Message Engine",
        "status": "running",
        "docs": "/docs",
    }
