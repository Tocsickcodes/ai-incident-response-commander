---
title: AI Incident Response Commander
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
python_version: "3.10"
app_file: ui/app.py
pinned: false
---

# 🚨 AI Incident Response Commander

An OpenEnv-compliant environment where an AI agent responds to production incidents — investigating logs and metrics, identifying root causes, and applying fixes.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What It Does

Simulates real-world SRE (Site Reliability Engineering) work. When a production system breaks, an engineer must:

1. Read alert messages to understand what's failing
2. Dig through logs to find the error
3. Check metrics (CPU, memory, network) to confirm the cause
4. Apply the correct fix before users are impacted

This environment trains AI agents to do exactly that — automatically.

---

## Environment Description

### Partial Observability

The agent starts **blind** — only alerts are visible. It must actively investigate:

```
Reset                   →  alerts only
inspect_logs action     →  logs revealed
check_metrics action    →  metrics revealed
fix action              →  episode ends, reward calculated
```

### Action Space

| Action | Type | Description |
|--------|------|-------------|
| `inspect_logs` | Investigation | Retrieve service log output |
| `check_metrics` | Investigation | Retrieve CPU, memory, error rate, latency |
| `restart_service` | Remediation | Restart the affected service |
| `rollback_deployment` | Remediation | Revert the last deployment |
| `escalate` | Remediation | Escalate to network/infrastructure team |

### Observation Space

```python
@dataclass
class Observation:
    alerts:   str                       # Always visible — the alarm
    logs:     Optional[str]             # None until inspect_logs called
    metrics:  Optional[Dict[str, Any]]  # None until check_metrics called
    message:  str                       # Result of last action
    done:     bool                      # Episode over?
    reward:   float                     # Reward from last step
```

### State (Episode Metadata)

```python
@dataclass
class State:
    episode_id:        str        # Unique episode identifier
    step_count:        int        # Steps taken so far
    scenario_name:     str        # Which scenario is running
    logs_revealed:     bool       # Has inspect_logs been called?
    metrics_revealed:  bool       # Has check_metrics been called?
    total_reward:      float      # Cumulative reward
    action_history:    List[str]  # All actions taken in order
    done:              bool       # Episode complete?
    correct_action:    str        # The correct fix (for grading)
```

### Reward Function

| Event | Reward | Description |
|-------|--------|-------------|
| Correct fix | **+100** | Applied the right remediation |
| Root cause identified | **+50** | Investigated before fixing |
| Per step | **-2** | Time penalty — encourages efficiency |
| Unnecessary action | **-10** | Repeated an already-done investigation |
| Wrong fix | **-30** | Applied the wrong remediation |

**Score range:** -42 (worst) → +148 (best) → Optimal run = **+144**

---

## Tasks / Scenarios

### Easy — Memory Leak
- **Signals:** `OutOfMemoryError` in logs + `memory_usage_pct: 94` in metrics
- **Correct fix:** `restart_service`
- **Why easy:** Signal is unambiguous — OOM errors directly indicate the problem

### Medium — Bad Deployment
- **Signals:** `NullPointerException` in logs + recent deployment in metrics
- **Correct fix:** `rollback_deployment`
- **Why medium:** Must check BOTH sources to distinguish a bad deploy from a pre-existing bug

### Hard — Network Issue
- **Signals:** `SocketTimeoutException` in logs + `packet_loss_pct: 38` in metrics
- **Correct fix:** `escalate`
- **Why hard:** High error rates look like a service problem, but the service is healthy (41% memory, 29% CPU). The real cause is external packet loss — only visible in metrics.

---

## Agent Graders (0.0 – 1.0)

```
eval/
├── graders.py     ← EasyGrader, MediumGrader, HardGrader
└── baseline.py    ← Reproducible comparison script
```

| Grader | Scenario | Passing Score | Key Challenge |
|--------|----------|---------------|---------------|
| Easy | memory_leak | ≥ 0.70 | Pick correct fix on clear signal |
| Medium | bad_deployment | ≥ 0.75 | Check both logs AND metrics |
| Hard | network_issue | ≥ 0.80 | Avoid trap actions, identify external cause |

**Overall score** = Easy×20% + Medium×35% + Hard×45%

### Baseline Results (seed=42)

| Agent | Easy | Medium | Hard | Overall | Grade |
|-------|------|--------|------|---------|-------|
| Rule-based | 1.000 | 1.000 | 1.000 | **1.000** | S |
| Random | 0.900 | 0.000 | 0.800 | 0.540 | C |
| Always Escalate | 0.000 | 0.000 | 1.000 | 0.450 | C |
| Always Restart | 1.000 | 0.000 | 0.000 | 0.200 | F |

---

## Project Structure

```
ai-incident-agent/
├── env/
│   ├── environment.py    ← OpenEnv-compliant: reset(), step(), state()
│   ├── models.py         ← Typed dataclasses: Action, Observation, State, StepResult
│   └── scenarios.py      ← 3 scenario definitions
├── agent/
│   ├── policy.py         ← Rule-based agent (gold standard)
│   └── llm_agent.py      ← LLM agent (Groq/OpenAI/Anthropic)
├── eval/
│   ├── graders.py        ← Easy/Medium/Hard graders (0.0–1.0)
│   └── baseline.py       ← Reproducible baseline script
├── ui/
│   └── app.py            ← Flask server (/, /run, /reset, /state, /health)
├── tests/
│   └── test_environment.py  ← 85 tests
├── inference.py          ← OpenEnv-format inference script
├── pre_validate.py       ← Pre-submission validation (10 checks)
├── main.py               ← CLI runner
├── openenv.yaml          ← OpenEnv spec config
├── Dockerfile            ← Container definition
└── requirements.txt      ← Dependencies
```

---

## Setup

### Prerequisites
- Python 3.10+
- pip

### Install

```bash
git clone <your-repo-url>
cd ai-incident-agent
pip install -r requirements.txt
```

### Set API Key (for LLM agent)

```powershell
# Windows PowerShell
$env:HF_TOKEN = "gsk_your-groq-key-here"

# Mac/Linux
export HF_TOKEN="gsk_your-groq-key-here"
```

Get a free Groq key at: **console.groq.com/keys**

---

## Running

### CLI — quick test

```bash
python main.py
```

### Web UI

```bash
python ui/app.py
# Open http://localhost:5000
```

### Run tests

```bash
python tests/test_environment.py
# Expected: 85/85 passed
```

### Baseline evaluation

```bash
# All agents, comparison table
python eval/baseline.py --quiet

# Single agent, detailed report
python eval/baseline.py --agent rule

# LLM agent (requires HF_TOKEN)
python eval/baseline.py --agent groq

# Average over 3 runs
python eval/baseline.py --agent groq --runs 3
```

### Inference script

```bash
python inference.py
```

**Output format:**
```
[START] task=bad_deployment env=incident-response model=llama-3.1-8b-instant
[STEP] step=1 action=inspect_logs reward=-2.00 done=false error=null
[STEP] step=2 action=check_metrics reward=-2.00 done=false error=null
[STEP] step=3 action=rollback_deployment reward=148.00 done=true error=null
[END] success=true steps=3 rewards=-2.00,-2.00,148.00
```

### Pre-submission validation

```bash
python pre_validate.py        # run all 10 checks
python pre_validate.py --fix  # show fix hints for failures
```

### Docker

```bash
docker build -t ai-incident-agent .
docker run -p 5000:5000 -e HF_TOKEN=gsk_... ai-incident-agent
```

---

## HTTP API

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/run` | Run one episode, return JSON result |
| `POST` | `/reset` | Reset environment, return initial observation |
| `GET` | `/state` | Return current episode state metadata |
| `GET` | `/health` | Liveness check |

### Example — POST /reset

```bash
curl -X POST http://localhost:5000/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

```json
{
  "status": "ok",
  "scenario": "Bad Deployment",
  "alerts": "🚨 ALERT: Error rate spiked to 32%...",
  "message": "Environment reset. Ready for new episode."
}
```

### Example — GET /state

```bash
curl http://localhost:5000/state
```

```json
{
  "episode_id": "a3f2b1c4",
  "step_count": 2,
  "scenario_name": "Bad Deployment",
  "logs_revealed": true,
  "metrics_revealed": true,
  "total_reward": -4.0,
  "action_history": ["inspect_logs", "check_metrics"],
  "done": false,
  "correct_action": "rollback_deployment"
}
```

---

## Using the Typed API

```python
from env.environment import Environment
from env.models import Action, Observation, State

env = Environment()

# reset() → Observation
obs: Observation = env.reset()
print(obs.alerts)    # always visible
print(obs.logs)      # None — not yet revealed

# step() → StepResult
result = env.step("inspect_logs")        # string
result = env.step(Action("check_metrics"))  # or typed Action

obs: Observation = result.observation
print(obs.logs)      # now revealed
print(result.reward) # -2.0

# state() → State
s: State = env.state()
print(s.episode_id)       # "a3f2b1c4"
print(s.step_count)       # 1
print(s.logs_revealed)    # True
print(s.action_history)   # ["inspect_logs"]
```

---

## Adding Your Own Agent

Any class with an `act(observation)` method works:

```python
class MyAgent:
    def act(self, observation) -> str:
        # observation.alerts   — always available
        # observation.logs     — available after inspect_logs
        # observation.metrics  — available after check_metrics
        if observation.logs is None:
            return "inspect_logs"
        if observation.metrics is None:
            return "check_metrics"
        # your logic here
        return "escalate"
```

Register it in `eval/baseline.py` to grade it:

```python
BASELINE_AGENTS["my_agent"] = lambda seed: MyAgent()
```

Then run:

```bash
python eval/baseline.py --agent my_agent
```

---

## License

MIT — free to use, modify, and distribute.