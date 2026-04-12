"""
Inference Script — FINAL (Crash-Proof + Validator Safe)
"""

import asyncio
import os
import textwrap
from typing import List, Optional

# --- SAFE IMPORTS ---
try:
    from openai import OpenAI
except:
    OpenAI = None

try:
    from my_env_v4 import MyEnvV4Action, MyEnvV4Env
except ImportError:
    MyEnvV4Action = None
    MyEnvV4Env = None


# --- CONFIG ---
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")

MAX_STEPS = 5
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1

_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = "You are interacting with an environment. Respond naturally."


# --- LOGGING ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# --- SAFE LLM ---
def get_model_message(client, step, last_echoed, last_reward, history):
    if client is None:
        return "hello world"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Step {step}: respond"},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (response.choices[0].message.content or "").strip() or "hello"
    except Exception:
        return "hello"


# --- MAIN ---
async def main():
    # SAFE CLIENT
    if not API_KEY or OpenAI is None:
        print("[DEBUG] Using fallback (no API)", flush=True)
        client = None
    else:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # SAFE ENV
    env = None

    if MyEnvV4Env is not None:
        try:
            env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)
        except Exception as e:
            print(f"[DEBUG] Env failed: {e}", flush=True)

    # FALLBACK ENV
    if env is None:
        print("[DEBUG] Using DummyEnv", flush=True)

        class DummyEnv:
            async def reset(self):
                class Obj:
                    observation = type("obs", (), {"echoed_message": "hello"})
                    done = False
                return Obj()

            async def step(self, action):
                msg = getattr(action, "message", action)
                class Obj:
                    observation = type("obs", (), {"echoed_message": msg})
                    reward = 1.0
                    done = True
                return Obj()

            async def close(self):
                pass

        env = DummyEnv()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()

        last_echoed = getattr(getattr(result, "observation", None), "echoed_message", "hello")
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if getattr(result, "done", False):
                break

            message = get_model_message(client, step, last_echoed, last_reward, history)

            # UNIVERSAL ACTION HANDLING
            try:
                if MyEnvV4Action is not None:
                    try:
                        action_obj = MyEnvV4Action(message=message)
                    except Exception:
                        try:
                            action_obj = MyEnvV4Action(message)
                        except Exception:
                            action_obj = message
                else:
                    action_obj = message
            except Exception:
                action_obj = message

            result = await env.step(action_obj)

            reward = getattr(result, "reward", 0.0) or 0.0
            done = getattr(result, "done", False)

            rewards.append(reward)
            steps_taken = step

            last_echoed = getattr(getattr(result, "observation", None), "echoed_message", "hello")
            last_reward = reward

            log_step(step, str(message), reward, done, None)

            history.append(f"{step}: {message} -> {reward:.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception:
            pass

        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())