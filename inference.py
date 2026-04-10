"""
Inference Script — AI Incident Response Commander
"""

import os
import json
import textwrap
from typing import List, Optional
from flask import Flask, request, jsonify

# --- SAFE IMPORT ---
try:
    from openai import OpenAI
except:
    OpenAI = None

# --- CONFIG ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
BENCHMARK    = "incident-response"

MAX_STEPS = 5
TEMPERATURE = 0
MAX_TOKENS = 100

VALID_ACTIONS = [
    "inspect_logs", "check_metrics", "restart_service",
    "rollback_deployment", "escalate"
]

FALLBACK_ACTION = "inspect_logs"

SYSTEM_PROMPT = textwrap.dedent("""
You are an SRE handling a production incident.
Return ONLY JSON:
{"action": "<action>", "reasoning": "<reason>"}
""").strip()

app = Flask(__name__)

# --- LOGGING ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

# --- SAFE LLM ---
def get_action(step: int):
    if not API_KEY or OpenAI is None:
        return VALID_ACTIONS[min(step - 1, len(VALID_ACTIONS) - 1)]

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Step {step}: choose best action for incident response"}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}   # Force JSON
        )

        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)

        action = data.get("action", FALLBACK_ACTION)
        return action if action in VALID_ACTIONS else FALLBACK_ACTION

    except Exception as e:
        print(f"[DEBUG] LLM failed: {e}", flush=True)
        return FALLBACK_ACTION

# ====================== FLASK ROUTES ======================

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health_check():
    """Healthcheck endpoint required by the platform"""
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "benchmark": BENCHMARK
    }), 200

@app.route('/predict', methods=['POST'])
@app.route('/', methods=['POST'])
def predict():
    """Main inference endpoint"""
    try:
        data = request.get_json() or {}
        # You can use input data if the benchmark sends any
        step = data.get("step", 1)

        action = get_action(step)

        # Simple deterministic reward
        reward = 1.0 if action in ["restart_service", "rollback_deployment"] else 0.5

        response = {
            "action": action,
            "reasoning": f"Selected {action} at step {step}",
            "reward": reward,
            "done": step >= MAX_STEPS
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ====================== MAIN ======================
if __name__ == "__main__":
    log_start(task="incident", env=BENCHMARK, model=MODEL_NAME)
    print("🚀 Starting Flask inference server on 0.0.0.0:5000", flush=True)
    
    # Run the server (required for the platform)
    app.run(host="0.0.0.0", port=5000, debug=False)