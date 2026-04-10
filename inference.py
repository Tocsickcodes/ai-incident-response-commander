"""
Inference Script — AI Incident Response Commander

Environment variables:
    HF_TOKEN       Your Groq / HuggingFace API key  (required)
    API_BASE_URL   LLM endpoint  (default: https://api.groq.com/openai/v1)
    MODEL_NAME     Model identifier  (default: llama-3.1-8b-instant)

Stdout format:
    [START] task=<n> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

Usage:
    $env:HF_TOKEN = "gsk_..."
    python inference.py
"""

import os
import sys
import json
import textwrap
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import Environment
from env.models import VALID_ACTIONS

API_BASE_URL    = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY         = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY")
MODEL_NAME      = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
BENCHMARK       = os.getenv("BENCHMARK", "incident-response")
MAX_STEPS       = int(os.getenv("MAX_STEPS", "10"))
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "200"))
FALLBACK_ACTION = "inspect_logs"

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SRE responding to a production incident.
    Respond with exactly one JSON object and nothing else:
        {"action": "<action>", "reasoning": "<why>"}

    Valid actions:
        inspect_logs        — retrieve service logs
        check_metrics       — retrieve service metrics
        restart_service     — fix: restart the affected service
        rollback_deployment — fix: revert the last deployment
        escalate            — fix: escalate to network/infrastructure team

    Strategy:
        1. inspect_logs first
        2. check_metrics to confirm
        3. Pick fix based on evidence:
           OutOfMemoryError + high memory      → restart_service
           NullPointerException + recent deploy → rollback_deployment
           SocketTimeoutException + packet loss → escalate
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


def build_user_message(step: int, observation, history: List[str]) -> str:
    parts = [f"Step: {step}", f"\nALERTS:\n{observation.alerts}"]

    if observation.logs:
        parts.append(f"\nLOGS:\n{observation.logs}")
    else:
        parts.append("\nLOGS: [not retrieved — use inspect_logs]")

    if observation.metrics:
        parts.append("\nMETRICS:\n" + "\n".join(f"  {k}: {v}" for k, v in observation.metrics.items()))
    else:
        parts.append("\nMETRICS: [not retrieved — use check_metrics]")

    if history:
        parts.append("\nPREVIOUS ACTIONS:\n" + "\n".join(history[-5:]))

    parts.append(f"\nValid actions: {', '.join(VALID_ACTIONS)}")
    parts.append('\nRespond with JSON only: {"action": "...", "reasoning": "..."}')
    return "\n".join(parts)


def call_llm(client, conversation: List[dict]) -> str:
    try:
        response = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return ""


def parse_action(raw: str) -> tuple:
    try:
        cleaned   = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data      = json.loads(cleaned)
        action    = str(data.get("action", "")).strip()
        reasoning = str(data.get("reasoning", ""))
        if action not in VALID_ACTIONS:
            return FALLBACK_ACTION, f"invalid action '{action}'"
        return action, reasoning
    except Exception:
        return FALLBACK_ACTION, "parse error"


def main() -> None:
    rewards     : List[float] = []
    steps_taken = 0
    success     = False

    try:
        if not API_KEY:
            print("[ERROR] No API key found. Set HF_TOKEN or GROQ_API_KEY.", flush=True)
            return

        try:
            from openai import OpenAI
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except ImportError:
            print("[ERROR] openai package not installed. Run: pip install openai", flush=True)
            return

        env          = Environment()
        conversation : List[dict] = []
        history      : List[str]  = []
        last_error   : Optional[str] = None

        observation = env.reset()
        log_start(task=env.scenario_key or "unknown", env=BENCHMARK, model=MODEL_NAME)

        for step in range(1, MAX_STEPS + 1):
            user_msg = build_user_message(step, observation, history)
            conversation.append({"role": "user", "content": user_msg})

            raw_reply = call_llm(client, conversation)
            conversation.append({
                "role":    "assistant",
                "content": raw_reply or f'{{"action": "{FALLBACK_ACTION}", "reasoning": "fallback"}}',
            })

            action, reasoning = parse_action(raw_reply)

            try:
                result     = env.step(action)
                last_error = None
            except ValueError as exc:
                last_error = str(exc)
                result     = env.step(FALLBACK_ACTION)

            observation = result.observation
            reward      = result.reward
            done        = result.done

            rewards.append(float(reward))
            steps_taken = step

            log_step(step=step, action=action, reward=float(reward), done=done, error=last_error)
            history.append(f"Step {step}: {action} → reward {reward:+.2f}")

            if done:
                success = reward > 0
                break
        else:
            success = False

    except Exception as exc:
        print(f"[DEBUG] Unexpected error: {exc}", flush=True)
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()