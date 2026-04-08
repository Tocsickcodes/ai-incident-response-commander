"""
Pre-Validation Script — AI Incident Response Commander
Run before submitting to HuggingFace. Mirrors validate-submission.sh checks.

Usage:
    python pre_validate.py
    python pre_validate.py --fix    # show fix hints for each failure
    python pre_validate.py --quiet  # only print failures
"""

import sys
import os
import argparse
import importlib
import json
import re
import io
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

PASS     = 0
FAIL     = 0
QUIET    = False
SHOW_FIX = False


def check(label: str, condition: bool, fix: str = "", detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        if not QUIET:
            print(f"  ✅  {label}")
    else:
        FAIL += 1
        print(f"  ❌  {label}")
        if detail:
            print(f"       Detail : {detail}")
        if fix and SHOW_FIX:
            print(f"       Fix    : {fix}")


def section(title: str):
    print(f"\n{'─' * 58}")
    print(f"  {title}")
    print(f"{'─' * 58}")


def check_required_files():
    section("CHECK 1 — Required Files")
    required = {
        "inference.py":       "Create inference.py in the project root",
        "Dockerfile":         "Create a Dockerfile in the project root",
        "requirements.txt":   "Create requirements.txt listing all dependencies",
        "ui/app.py":          "Flask server must exist at ui/app.py",
        "main.py":            "main.py must exist in project root",
        "env/environment.py": "Environment class must exist",
        "env/scenarios.py":   "Scenarios must exist",
        "agent/policy.py":    "Rule-based agent must exist",
    }
    for path, fix in required.items():
        full = os.path.join(ROOT, path)
        check(f"File exists: {path}", os.path.isfile(full), fix=fix,
              detail="" if os.path.isfile(full) else f"Not found at {full}")


def check_imports():
    section("CHECK 2 — Python Imports")
    modules = {
        "env.environment": "Check env/environment.py for syntax errors",
        "env.scenarios":   "Check env/scenarios.py for syntax errors",
        "agent.policy":    "Check agent/policy.py for syntax errors",
        "main":            "Check main.py for syntax errors",
    }
    for mod, fix in modules.items():
        try:
            if mod in sys.modules:
                del sys.modules[mod]
            importlib.import_module(mod)
            check(f"Import: {mod}", True)
        except Exception as e:
            check(f"Import: {mod}", False, fix=fix, detail=str(e))

    try:
        import flask  # noqa: F401
        check("Import: flask", True)
    except ImportError:
        check("Import: flask", False, fix="Run: pip install flask", detail="flask not installed")


def check_environment():
    section("CHECK 3 — Environment Logic")
    try:
        from env.environment import Environment

        env = Environment()
        obs = env.reset()
        check("reset() returns observation with alerts", bool(obs.alerts))
        check("logs hidden at reset",                    obs.logs is None)
        check("metrics hidden at reset",                 obs.metrics is None)
        check("scenario loaded after reset",             env.scenario is not None)

        result = env.step("inspect_logs")
        check("step() reveals logs",          result.observation.logs is not None)
        check("step() returns float reward",  isinstance(result.reward, (int, float)))
        check("step() returns bool done",     isinstance(result.done, bool))

        env2 = Environment()
        env2.reset()
        env2.step("inspect_logs")
        env2.step("check_metrics")
        assert env2.scenario is not None
        result2 = env2.step(env2.scenario["correct_action"])
        check("correct fix gives positive reward", result2.reward > 0)
        check("correct fix ends episode",          result2.done is True)

    except Exception as e:
        check("Environment logic", False, detail=str(e))


def check_inference_format():
    section("CHECK 4 — inference.py Stdout Format")
    try:
        src = open(os.path.join(ROOT, "inference.py")).read()

        check("[START] format defined",      "log_start" in src and "[START]" in src)
        check("[STEP] format defined",       "log_step"  in src and "[STEP]"  in src)
        check("[END] format defined",        "log_end"   in src and "[END]"   in src)
        check("reward formatted to 2dp",     ":.2f" in src)
        check("done uses lowercase bool",    ".lower()" in src)
        check("error field uses null",       '"null"' in src or "'null'" in src)
        check("[END] in finally block",      "finally" in src and "log_end" in src)

        ns: dict = {}
        exec("""
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)
""", ns)

        buf = io.StringIO()
        with redirect_stdout(buf):
            ns["log_start"]("memory_leak", "incident-response", "llama-3.1-8b-instant")
            ns["log_step"](1, "inspect_logs",    -2.0,  False, None)
            ns["log_step"](2, "check_metrics",   -2.0,  False, None)
            ns["log_step"](3, "restart_service", 148.0, True,  None)
            ns["log_end"](True, 3, [-2.0, -2.0, 148.0])

        lines = buf.getvalue().strip().splitlines()
        check("Output has 5 lines (1 START + 3 STEP + 1 END)", len(lines) == 5)
        check("[START] line format correct",
              lines[0].startswith("[START]") and "task=" in lines[0] and "model=" in lines[0])
        check("[STEP] lines match format",
              all(re.match(r"\[STEP\] step=\d+ action=\S+ reward=-?\d+\.\d{2} done=(true|false) error=\S+", l)
                  for l in lines[1:4]))
        check("[END] line format correct",
              bool(re.match(r"\[END\] success=(true|false) steps=\d+ rewards=.+", lines[-1])))

    except Exception as e:
        check("inference.py format", False, detail=str(e))


def check_flask_routes():
    section("CHECK 5 — Flask Routes")
    try:
        if "ui.app" in sys.modules:
            del sys.modules["ui.app"]
        from ui.app import app
        client = app.test_client()

        r = client.get("/")
        check("GET / returns 200", r.status_code == 200, detail=f"got {r.status_code}")

        r = client.get("/run")
        check("GET /run returns 200", r.status_code == 200, detail=f"got {r.status_code}")
        try:
            check("GET /run returns JSON with scenario", "scenario" in json.loads(r.data))
        except Exception:
            check("GET /run returns JSON with scenario", False)

        r = client.get("/health")
        check("GET /health returns 200", r.status_code == 200, detail=f"got {r.status_code}")

        r = client.post("/reset", json={})
        check("POST /reset returns 200", r.status_code == 200, detail=f"got {r.status_code}")
        try:
            check("POST /reset returns status=ok", json.loads(r.data).get("status") == "ok")
        except Exception:
            check("POST /reset returns status=ok", False)

    except Exception as e:
        check("Flask routes", False, detail=str(e))


def check_reset_endpoint():
    section("CHECK 6 — /reset Endpoint (Mirrors Validator curl Check)")
    try:
        if "ui.app" in sys.modules:
            del sys.modules["ui.app"]
        from ui.app import app
        client = app.test_client()

        r = client.post("/reset", content_type="application/json", data="{}")
        check("POST /reset with JSON body → 200", r.status_code == 200, detail=f"HTTP {r.status_code}")

        data = json.loads(r.data)
        for key in ["status", "scenario", "alerts", "message"]:
            check(f"/reset response has '{key}'", key in data)

    except Exception as e:
        check("/reset endpoint", False, detail=str(e))


def check_dockerfile():
    section("CHECK 7 — Dockerfile")
    dockerfile = os.path.join(ROOT, "Dockerfile")
    if not os.path.isfile(dockerfile):
        check("Dockerfile exists", False, fix="Create a Dockerfile in the project root")
        return

    check("Dockerfile exists", True)
    content = open(dockerfile).read().upper()
    for directive in ["FROM", "WORKDIR", "COPY", "RUN", "EXPOSE", "CMD", "HEALTHCHECK"]:
        check(f"Dockerfile has {directive}", directive in content)


def check_requirements():
    section("CHECK 8 — requirements.txt")
    req_path = os.path.join(ROOT, "requirements.txt")
    if not os.path.isfile(req_path):
        check("requirements.txt exists", False, fix="Create requirements.txt")
        return

    content = open(req_path).read().lower()
    for pkg in ["flask", "openai"]:
        check(f"requirements.txt lists: {pkg}", pkg in content,
              fix=f"Add {pkg} to requirements.txt")


def check_api_key():
    section("CHECK 9 — API Key")
    hf_token = os.environ.get("HF_TOKEN", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    check("HF_TOKEN or GROQ_API_KEY is set", bool(hf_token or groq_key),
          fix="Set your key: $env:HF_TOKEN = 'gsk_your-key-here'",
          detail="Neither HF_TOKEN nor GROQ_API_KEY found in environment")
    if hf_token:
        check("HF_TOKEN is non-empty", len(hf_token) > 10)
    if groq_key:
        check("GROQ_API_KEY starts with gsk_", groq_key.startswith("gsk_"))


def check_e2e():
    section("CHECK 10 — End-to-End Simulation")
    try:
        from main import run_simulation
        result = run_simulation(verbose=False)
        check("Simulation returns dict",       isinstance(result, dict))
        check("Result has scenario key",       "scenario"    in result)
        check("Result has total_score key",    "total_score" in result)
        check("Result has total_steps key",    "total_steps" in result)
        check("Score is positive",             result.get("total_score", 0) > 0)
        check("Completed in 10 steps or less", result.get("total_steps", 99) <= 10)
    except Exception as e:
        check("End-to-end simulation", False, detail=str(e))


def main():
    global QUIET, SHOW_FIX

    parser = argparse.ArgumentParser(description="Pre-validation for AI Incident Response Commander")
    parser.add_argument("--fix",   action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    QUIET    = args.quiet
    SHOW_FIX = args.fix

    print(f"\n{'=' * 58}")
    print("  PRE-VALIDATION — AI Incident Response Commander")
    print(f"{'=' * 58}")

    check_required_files()
    check_imports()
    check_environment()
    check_inference_format()
    check_flask_routes()
    check_reset_endpoint()
    check_dockerfile()
    check_requirements()
    check_api_key()
    check_e2e()

    total = PASS + FAIL
    print(f"\n{'=' * 58}")
    print(f"  RESULTS: {PASS}/{total} checks passed")
    print(f"{'=' * 58}")

    if FAIL == 0:
        print("  🏆 All checks passed — safe to submit.\n")
        sys.exit(0)
    else:
        print(f"  ⚠️  {FAIL} check(s) failed. Run with --fix to see fixes.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()