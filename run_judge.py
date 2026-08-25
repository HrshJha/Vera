#!/usr/bin/env python3
"""
Wrapper to run judge_simulator.py with pre-configured Gemini settings.
Usage: python run_judge.py [scenario]
  scenario: warmup, phase2_short, auto_reply_hell, intent_transition, hostile, all
"""
import os
import sys

# Get API key from env or .env file
gemini_key = os.environ.get("GEMINI_API_KEY", "")
if not gemini_key:
    # Try to read from .env file
    try:
        with open(".env") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY="):
                    gemini_key = line.strip().split("=", 1)[1].strip('"\'')
                    break
    except FileNotFoundError:
        pass

if not gemini_key:
    print("[ERROR] GEMINI_API_KEY not set.")
    print("  Set it in .env file: GEMINI_API_KEY=your_key_here")
    print("  Or export it: export GEMINI_API_KEY=your_key_here")
    sys.exit(1)

# Patch the judge_simulator module config before importing
import judge_simulator as js

# Override config
js.BOT_URL = "http://localhost:8080"
js.LLM_PROVIDER = "gemini"
js.LLM_API_KEY = gemini_key
js.LLM_MODEL = "gemini-1.5-flash"
js.TEST_SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "warmup"

# Run
js.main()
