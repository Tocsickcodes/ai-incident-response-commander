import os
import sys
import json
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import Environment
from env.models import VALID_ACTIONS

PROVIDER = "groq"

MODELS = {
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "groq":      "llama-3.1-8b-instant",
}

ENV_KEYS = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq":      "GROQ_API_KEY",
}

KEYS_URL = {
    "openai":    "platform.openai.com/api-keys",
    "anthropic": "console.anthropic.com/keys",
    "groq":      "console.groq.com/keys",
}

SYSTEM_PROMPT = """
You are an expert Site Reliability Engineer responding to a production incident.
Respond with exactly one JSON object and nothing else:
{"action": "<action>", "reasoning": "<why>"}

Valid actions:
    inspect_logs        - retrieve service logs
    check_metrics       - retrieve service metrics
    restart_service     - fix: restart the affected service
    rollback_deployment - fix: revert the last deployment
    escalate            - fix: escalate to network/infrastructure team

Strategy:
    1. inspect_logs first
    2. check_metrics to confirm
    3. Fix based on evidence:
       OutOfMemoryError + high memory       -> restart_service
       NullPointerException + recent deploy -> rollback_deployment
       SocketTimeoutException + packet loss -> escalate
""".strip()


def build_user_message(observation, step: int) -> str:
    logs    = observation.logs    if hasattr(observation, "logs")    else observation.get("logs")
    metrics = observation.metrics if hasattr(observation, "metrics") else observation.get("metrics")
    alerts  = observation.alerts  if hasattr(observation, "alerts")  else observation.get("alerts")

    parts = [f"Step: {step}", f"\nALERTS:\n{alerts}"]

    if logs:
        parts.append(f"\nLOGS:\n{logs}")
    else:
        parts.append("\nLOGS: [not retrieved — use inspect_logs]")

    if metrics:
        parts.append("\nMETRICS:\n" + "\n".join(f"  {k}: {v}" for k, v in metrics.items()))
    else:
        parts.append("\nMETRICS: [not retrieved — use check_metrics]")

    parts.append(f"\nValid actions: {', '.join(VALID_ACTIONS)}")
    parts.append('\nRespond with JSON only: {"action": "...", "reasoning": "..."}')
    return "\n".join(parts)


def parse_response(raw: str):
    try:
        cleaned   = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data      = json.loads(cleaned)
        action    = data.get("action", "").strip()
        reasoning = data.get("reasoning", "")
        if action not in VALID_ACTIONS:
            return "inspect_logs", f"invalid action '{action}'"
        return action, reasoning
    except Exception:
        return "inspect_logs", "parse error"


class OpenAIClient:
    def __init__(self, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model  = model

    def call(self, history: list[dict[str, Any]]) -> str:
        response = self.client.chat.completions.create(
            model       = self.model,
            messages    = [{"role": "system", "content": SYSTEM_PROMPT}] + history, # pyright: ignore[reportArgumentType]
            temperature = 0,
            max_tokens  = 200,
        )
        return response.choices[0].message.content or ""


class AnthropicClient:
    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model  = model

    def call(self, history: list[dict[str, Any]]) -> str:
        clean = [m for m in history if m["role"] != "system"]
        response = self.client.messages.create(
            model     = self.model,
            system    = SYSTEM_PROMPT,
            messages  = clean,
            max_tokens= 200,
        )
        return response.content[0].text or ""


class GroqClient:
    def __init__(self, model: str):
        try:
            from groq import Groq
        except ImportError:
            print("groq package not found. Run: pip install groq")
            sys.exit(1)
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("GROQ_API_KEY not set. Get a free key at console.groq.com/keys")
            sys.exit(1)
        self.client = Groq(api_key=api_key)
        self.model  = model

    def call(self, history: list[dict[str, Any]]) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model       = self.model,
                    messages    = messages, # type: ignore
                    temperature = 0,
                    max_tokens  = 200,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    wait = 10 * (attempt + 1)
                    print(f"Rate limit hit — retrying in {wait}s ({attempt+1}/3)...")
                    time.sleep(wait)
                else:
                    raise
        return ""


def make_client(provider: str, model: str):
    clients = {"openai": OpenAIClient, "anthropic": AnthropicClient, "groq": GroqClient}
    if provider not in clients:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(clients.keys())}")
    return clients[provider](model)


class LLMAgent:

    def __init__(self, provider: str = PROVIDER, model: str = ""):
        self.provider = provider
        self.model    = model or MODELS[provider]
        self.llm      = make_client(provider, self.model)
        self.history  : list[dict[str, Any]] = []
        self.step     : int  = 0

    def reset(self):
        self.history = []
        self.step    = 0

    def act(self, observation):
        self.step += 1
        self.history.append({"role": "user", "content": build_user_message(observation, self.step)})

        raw = self.llm.call(self.history)

        if not raw:
            return "inspect_logs", "empty response from LLM"

        self.history.append({"role": "assistant", "content": raw})  # type: ignore
        return parse_response(raw)


def run_llm_simulation(provider: str = PROVIDER, model: str = "", verbose: bool = True) -> dict:
    env   = Environment()
    agent = LLMAgent(provider=provider, model=model)
    obs   = env.reset()
    agent.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  AI INCIDENT COMMANDER  |  {provider.upper()}  |  {agent.model}")
        print(f"{'='*60}")
        print(f"  Scenario: {env.get_scenario_name()}")
        print(f"\n  ALERTS:\n{obs.alerts}")
        print(f"{'-'*60}")

    done = False
    while not done:
        action, reasoning = agent.act(obs)

        if verbose:
            print(f"\n  Step {env.step_count + 1}")
            print(f"  Reasoning : {reasoning}")
            print(f"  Action    : {action}")

        result = env.step(action)
        obs, reward, done = result.observation, result.reward, result.done

        if verbose:
            print(f"  Result    : {obs.message}")
            print(f"  Reward    : {reward:+.0f}")
            if obs.logs and action == "inspect_logs":
                print(f"\n  LOGS:\n{obs.logs}")
            if obs.metrics and action == "check_metrics":
                print(f"\n  METRICS")
                for k, v in obs.metrics.items():
                    print(f"    {k}: {v}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  DONE  |  Steps: {env.step_count}  |  Score: {env.total_reward:+.0f}")
        print(f"{'='*60}\n")

    return {"scenario": env.get_scenario_name(), "total_steps": env.step_count, "total_score": env.total_reward}


if __name__ == "__main__":
    key = ENV_KEYS[PROVIDER]
    if not os.environ.get(key):
        print(f"\n  {key} is not set.")
        print(f"  Run: $env:{key} = 'your-key-here'")
        print(f"  Get a key at: {KEYS_URL[PROVIDER]}\n")
        sys.exit(1)

    run_llm_simulation(provider=PROVIDER, verbose=True)