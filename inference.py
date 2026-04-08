# inference.py
# ===================================
# Inference Script — AI Incident Response Commander
# ===================================
#
# MANDATORY environment variables:
#   API_BASE_URL      The API endpoint for the LLM
#                     Default: https://api.groq.com/openai/v1
#   MODEL_NAME        The model identifier to use
#                     Default: llama-3.1-8b-instant
#   HF_TOKEN          Your Groq / HuggingFace API key
#
# STDOUT FORMAT — exactly three line types, in order:
#   [START] task=<task_name> env=<benchmark> model=<model_name>
#   [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
#   [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
#
# USAGE:
#   # Using Groq (recommended)
#   $env:HF_TOKEN    = "gsk_your-key-here"
#   python inference.py
#
#   # Override model or endpoint
#   $env:MODEL_NAME  = "llama-3.3-70b-versatile"
#   $env:API_BASE_URL = "https://api.groq.com/openai/v1"
#   python inference.py
#
# EXAMPLE OUTPUT:
#   [START] task=bad_deployment env=incident-response model=llama-3.1-8b-instant
#   [STEP] step=1 action=inspect_logs reward=-2.00 done=false error=null
#   [STEP] step=2 action=check_metrics reward=-2.00 done=false error=null
#   [STEP] step=3 action=rollback_deployment reward=148.00 done=true error=null
#   [END] success=true steps=3 rewards=-2.00,-2.00,148.00

import os
import sys
import json
import textwrap
from typing import List, Optional

# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from env.environment import Environment, VALID_ACTIONS

# ── Configuration — all overridable via environment variables ─────────────────

API_BASE_URL  = os.getenv("API_BASE_URL",  "https://api.groq.com/openai/v1")
API_KEY       = os.getenv("HF_TOKEN")      or os.getenv("GROQ_API_KEY")
MODEL_NAME    = os.getenv("MODEL_NAME",    "llama-3.1-8b-instant")
TASK_NAME     = os.getenv("TASK_NAME",     "")          # filled at runtime from scenario
BENCHMARK     = os.getenv("BENCHMARK",    "incident-response")
MAX_STEPS     = int(os.getenv("MAX_STEPS", "10"))
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS    = int(os.getenv("MAX_TOKENS", "200"))
FALLBACK_ACTION = "inspect_logs"             # safe default if LLM fails

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Site Reliability Engineer (SRE) responding to a production incident.

    At each step you receive the current observation (alerts, and logs/metrics if already retrieved).
    You must respond with exactly one JSON object and nothing else:
        {"action": "<action>", "reasoning": "<why>"}

    Valid actions:
        inspect_logs        — retrieve service logs
        check_metrics       — retrieve service metrics
        restart_service     — fix: restart the affected service
        rollback_deployment — fix: revert the last deployment
        escalate            — fix: escalate to network/infrastructure team

    Strategy:
        1. inspect_logs first to understand the error
        2. check_metrics to confirm your hypothesis
        3. Apply the correct fix based on evidence:
           - OutOfMemoryError + high memory  → restart_service
           - NullPointerException + recent deployment → rollback_deployment
           - SocketTimeoutException + packet loss     → escalate

    Do not include any text outside the JSON object.
    Do not use markdown code fences.
""").strip()

# ── Stdout logging — exact format from spec ───────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# ── Observation formatter — builds the LLM user message ──────────────────────

def build_user_message(step: int, observation: dict, history: List[str]) -> str:
    parts = [f"Step: {step}"]
    parts.append(f"\nALERTS:\n{observation['alerts']}")

    if observation.get("logs"):
        parts.append(f"\nLOGS:\n{observation['logs']}")
    else:
        parts.append("\nLOGS: [not yet retrieved — use inspect_logs]")

    if observation.get("metrics"):
        metrics_str = "\n".join(f"  {k}: {v}" for k, v in observation["metrics"].items())
        parts.append(f"\nMETRICS:\n{metrics_str}")
    else:
        parts.append("\nMETRICS: [not yet retrieved — use check_metrics]")

    if history:
        parts.append("\nPREVIOUS ACTIONS:\n" + "\n".join(history[-5:]))

    parts.append(f"\nValid actions: {', '.join(VALID_ACTIONS)}")
    parts.append('\nRespond with JSON only: {"action": "...", "reasoning": "..."}')
    return "\n".join(parts)

# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, conversation: List[dict]) -> str:
    """Call the LLM and return raw text. Returns empty string on failure."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation, # pyright: ignore[reportArgumentType]
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""

# ── Response parser ───────────────────────────────────────────────────────────

def parse_action(raw: str) -> tuple:
    """
    Parse LLM JSON response into (action, reasoning).
    Falls back to FALLBACK_ACTION on any parse failure.
    """
    try:
        cleaned = (raw.strip()
                   .removeprefix("```json")
                   .removeprefix("```")
                   .removesuffix("```")
                   .strip())
        data      = json.loads(cleaned)
        action    = str(data.get("action", "")).strip()
        reasoning = str(data.get("reasoning", ""))

        if action not in VALID_ACTIONS:
            return FALLBACK_ACTION, f"invalid action '{action}' — using fallback"

        return action, reasoning

    except Exception:
        return FALLBACK_ACTION, "parse error — using fallback"

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Validate API key before doing anything ────────────────────────────────
    if not API_KEY:
        print(
            "[ERROR] No API key found.\n"
            "  Set it with:\n"
            "    $env:HF_TOKEN = 'gsk_your-key-here'\n"
            "  Or:  $env:GROQ_API_KEY = 'gsk_your-key-here'",
            flush=True,
        )
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env    = Environment()

    # Episode state
    conversation: List[dict] = []   # full LLM chat history
    history:      List[str]  = []   # human-readable step history
    rewards:      List[float] = []
    steps_taken   = 0
    success       = False
    last_error: Optional[str] = None

    # Reset environment — picks a random scenario
    observation = env.reset()
    task_name   = env.scenario_key or "unknown"

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):

            # Build user message and add to conversation history
            user_msg = build_user_message(step, observation, history)
            conversation.append({"role": "user", "content": user_msg})

            # Ask LLM
            raw_reply = call_llm(client, conversation)

            # Add reply to history (even if empty — keeps turns balanced)
            conversation.append({
                "role":    "assistant",
                "content": raw_reply if raw_reply else '{"action": "' + FALLBACK_ACTION + '", "reasoning": "fallback"}',
            })

            # Parse action
            action, reasoning = parse_action(raw_reply)

            # Execute in environment
            try:
                observation, reward, done = env.step(action)
                last_error = None
            except ValueError as exc:
                # Invalid action somehow slipped through — log and fallback
                last_error = str(exc)
                observation, reward, done = env.step(FALLBACK_ACTION)

            rewards.append(float(reward))
            steps_taken = step

            # Emit [STEP] line — exact spec format
            log_step(
                step   = step,
                action = action,
                reward = float(reward),
                done   = done,
                error  = last_error,
            )

            # Update human-readable history for next LLM prompt
            history.append(
                f"Step {step}: {action} → reward {reward:+.2f}"
                + (f" | reasoning: {reasoning}" if reasoning else "")
            )

            if done:
                success = reward > 0
                break

        else:
            # Reached MAX_STEPS without done=true
            success = False

    except Exception as exc:
        last_error = str(exc)
        print(f"[DEBUG] Unexpected error: {exc}", flush=True)
        success = False

    finally:
        # [END] always emitted, even on exception — per spec
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()