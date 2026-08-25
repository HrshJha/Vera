# Vera — Merchant Growth Message Engine
**magicpin AI Challenge Submission**

Vera is an autonomous, context-aware Merchant Growth Assistant engineered for high-precision merchant engagement across hyper-local retail categories (Dentists, Gyms, Salons, Pharmacies, Restaurants).

---

## 🏛️ Core Architectural Tenet

> **"The deterministic system decides. The LLM writes."**

Vera strictly decouples decision intelligence from generative drafting:
- **Decision Engine (Deterministic)**: 7 Hard Gates, 10-factor priority formula, simulated time tracking, anti-repetition hashing, auto-reply detection, and category suppression. The LLM never decides *whether* to send, *which* trigger to prioritize, or *when* to suppress.
- **Message Composer (LLM)**: Gemini Flash models generate grounded WhatsApp messages using structured, sourced Fact Packets with provenance tags.
- **Output Validator (Grounding & Safety)**: Enforces single-binary CTAs, verifies clinical/business tone, blocks internal jargon, and ensures zero unverified factual claims.

---

## ⚡ Multi-Key Auto-Rotating Pool (`vera/engine/key_rotator.py`)

To guarantee zero downtime and eliminate rate limits (`429 Too Many Requests`), Vera features an automated API Key Pool:
- **Dynamic Pool Discovery**: Auto-detects all Gemini keys from `.env` (`GEMINI_API_KEY`, `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, etc.).
- **Automatic Cooldown & Failover**: When any key hits a quota or rate limit, it is automatically parked for its `retryDelay`, and traffic instantly switches to the next healthy key in the pool with **0ms delay**.
- **Multi-Model Cascade**: Cascades through `gemini-3.6-flash`, `gemini-3.5-flash-lite`, and OpenRouter backup models.
- **Live Telemetry**: Real-time pool status accessible at `GET /v1/keypool`.

---

## 📊 Evaluation & Verification

Vera achieves a **100% PASS** on the official judge test suite and an **EXCELLENT (84%)** rating on the quality rubric:

| Scenario / Metric | Status | Details |
|---|---|---|
| **Warmup & Healthz** | **PASS** | `<1ms` latency, versioned atomic context ingestion |
| **Auto-Reply Hell** | **PASS** | Turn 1 & 2 backoff 1800s, Turn 3 graceful END |
| **Intent Commitment** | **PASS** | Switches immediately to ACTION mode with zero qualifying questions |
| **Hostile Exit** | **PASS** | Graceful polite termination & permanent suppression |
| **Message Specificity** | **9/10** | Cites exact dates, numbers, regulatory bodies, and clinical metrics |
| **Category Fit** | **9/10** | Clinical salutations (`Dr.`), operator tone, zero taboo words |
| **Decision Quality** | **9/10** | Directly anchored on trigger payload facts |

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/HrshJha/Vera.git
cd Vera

# Create virtual environment & install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Configuration
Copy the example configuration:
```bash
cp .env.example .env
```
Edit `.env` and add your free Gemini API keys:
```env
GEMINI_API_KEY=your_primary_gemini_key
GEMINI_API_KEY_1=your_second_gemini_key  # Optional
GEMINI_MODEL=gemini-3.6-flash
```

### 3. Start the Server
```bash
bash start.sh
# or manually:
uvicorn app:app --host 0.0.0.0 --port 8080
```

### 4. Run the Judge Simulator
```bash
# Test all behavioral scenarios:
TEST_SCENARIO=all python -u judge_simulator.py

# Test message quality scoring:
TEST_SCENARIO=phase2_short python -u judge_simulator.py

# Inspect active key pool:
curl -s http://localhost:8080/v1/keypool | python3 -m json.tool
```

---

## 📁 Repository Structure

```
├── app.py                      # FastAPI server (7 endpoints: healthz, metadata, context, tick, reply, teardown, keypool)
├── requirements.txt            # Minimal, pinned dependencies
├── start.sh                    # Automated startup script
├── submission.jsonl            # 30 canonical evaluation test pairs
├── generate_submission.py      # Canonical submission generator
├── judge_simulator.py          # Official challenge test & quality evaluation suite
├── dataset/                    # Seed and expanded retail context dataset (50 merchants, 200 customers, 100 triggers)
└── vera/
    └── engine/
        ├── context_store.py    # Thread-safe versioned store (HTTP 409 conflict detection)
        ├── conversation.py     # Deterministic 10-state conversation machine
        ├── fact_builder.py     # Fact packet builder with provenance tagging
        ├── key_rotator.py      # Multi-key auto-rotating pool & failover manager
        ├── llm_writer.py       # Multi-model Gemini composer with JSON parser
        ├── message_families.py # 25+ trigger templates and single-binary CTAs
        ├── orchestrator.py     # Main engine pipeline orchestrator
        ├── ranker.py           # 7 Hard Gates & 10-factor priority formula
        ├── suppression.py      # Simulated time-based suppression & deduplication
        └── validator.py        # Grounding & category taboo word validator
```

---

## ⚖️ License & Attribution
Submitted for the **magicpin AI Challenge**. Built with ❤️ by the Vera Team.