import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import Environment
from env.models import VALID_ACTIONS, Observation
from env.scenarios import SCENARIOS
from agent.policy import Agent

PASS = 0
FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  ✅ PASS  {label}")
        PASS += 1
    else:
        print(f"  ❌ FAIL  {label}" + (f" → {detail}" if detail else ""))
        FAIL += 1

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


section("1. SCENARIO DATA INTEGRITY")

REQUIRED_KEYS = ["name", "alerts", "logs", "metrics", "root_cause", "correct_action"]

for key, scenario in SCENARIOS.items():
    for field in REQUIRED_KEYS:
        check(f"scenarios['{key}'] has field '{field}'", field in scenario)
    check(f"scenarios['{key}']['metrics'] is a dict", isinstance(scenario["metrics"], dict))
    check(f"scenarios['{key}']['correct_action'] is valid", scenario["correct_action"] in VALID_ACTIONS)

check("At least 3 scenarios defined", len(SCENARIOS) >= 3)


section("2. RESET BEHAVIOUR")

env = Environment()
obs = env.reset()

check("reset() returns an Observation",             isinstance(obs, Observation))
check("observation has alerts field",               hasattr(obs, "alerts"))
check("observation has logs field",                 hasattr(obs, "logs"))
check("observation has metrics field",              hasattr(obs, "metrics"))
check("alerts is a non-empty string",               isinstance(obs.alerts, str) and len(obs.alerts) > 0)
check("logs is None at reset (partial obs)",        obs.logs is None)
check("metrics is None at reset (partial obs)",     obs.metrics is None)
check("step_count is 0 after reset",                env.step_count == 0)
check("total_reward is 0 after reset",              env.total_reward == 0)
check("done is False after reset",                  env.done is False)
check("scenario is loaded after reset",             env.scenario is not None)
check("scenario_key is valid",                      env.scenario_key in SCENARIOS)


section("3. PARTIAL OBSERVABILITY")

env = Environment()
obs = env.reset()

check("logs hidden before inspect_logs",     obs.logs is None)
check("metrics hidden before check_metrics", obs.metrics is None)

r1 = env.step("inspect_logs")
check("logs revealed after inspect_logs",        r1.observation.logs is not None)
check("metrics still hidden after inspect_logs", r1.observation.metrics is None)

r2 = env.step("check_metrics")
check("metrics revealed after check_metrics",    r2.observation.metrics is not None)
check("logs still visible after check_metrics",  r2.observation.logs is not None)
check("alerts always visible",                   r2.observation.alerts is not None)


section("4. REWARD SYSTEM")

env = Environment()
env.reset()
check("inspect_logs gives -2 time penalty",
      env.step("inspect_logs").reward == -2, f"got {env.step('inspect_logs').reward}")

env = Environment()
env.reset()
env.step("inspect_logs")
check("repeat inspect_logs costs -12", env.step("inspect_logs").reward == -12)

env = Environment()
env.reset()
env.step("check_metrics")
check("repeat check_metrics costs -12", env.step("check_metrics").reward == -12)

env = Environment()
env.reset()
r = env.step("restart_service")
check("fix with no investigation costs -12",        r.reward == -12, f"got {r.reward}")
check("episode NOT ended after uninvestigated fix", r.done is False)

env = Environment()
env.reset()
env.step("inspect_logs")
env.step("check_metrics")
assert env.scenario is not None
correct = env.scenario["correct_action"]
r = env.step(correct)
check("correct fix rewards +148",      r.reward == 148, f"got {r.reward}")
check("episode ends after correct fix", r.done is True)

env = Environment()
env.reset()
env.step("inspect_logs")
assert env.scenario is not None
correct = env.scenario["correct_action"]
wrong = [a for a in ["restart_service", "rollback_deployment", "escalate"] if a != correct][0]
r = env.step(wrong)
check("wrong fix costs -32",           r.reward == -32, f"got {r.reward}")
check("episode ends after wrong fix",  r.done is True)


section("5. STEP COUNTER & HISTORY")

env = Environment()
env.reset()
check("step_count starts at 0", env.step_count == 0)

env.step("inspect_logs")
check("step_count is 1 after 1 action", env.step_count == 1)

env.step("check_metrics")
check("step_count is 2 after 2 actions", env.step_count == 2)
check("action_history tracks actions", env.action_history == ["inspect_logs", "check_metrics"])


section("6. GUARD RAILS")

env = Environment()
try:
    env.step("inspect_logs")
    check("step() before reset() raises RuntimeError", False)
except RuntimeError:
    check("step() before reset() raises RuntimeError", True)

env = Environment()
check("get_scenario_name() before reset() returns 'None'", env.get_scenario_name() == "None")

env = Environment()
env.reset()
try:
    env.step("nuke_everything")
    check("invalid action raises ValueError", False)
except ValueError:
    check("invalid action raises ValueError", True)

env = Environment()
env.reset()
env.step("inspect_logs")
env.step("check_metrics")
assert env.scenario is not None
env.step(env.scenario["correct_action"])
try:
    env.step("inspect_logs")
    check("step() after done raises RuntimeError", False)
except RuntimeError:
    check("step() after done raises RuntimeError", True)

env = Environment()
env.reset()
env.step("inspect_logs")
env.step("check_metrics")
env.reset()
check("reset() clears logs_revealed",    env.logs_revealed is False)
check("reset() clears metrics_revealed", env.metrics_revealed is False)
check("reset() clears step_count",       env.step_count == 0)
check("reset() clears total_reward",     env.total_reward == 0)
check("reset() clears action_history",   env.action_history == [])
check("reset() clears done flag",        env.done is False)


section("7. CUMULATIVE REWARD TRACKING")

env = Environment()
env.reset()
env.step("inspect_logs")
env.step("check_metrics")
assert env.scenario is not None
env.step(env.scenario["correct_action"])
check("total_reward sums correctly", env.total_reward == 144, f"got {env.total_reward}")


section("8. RULE-BASED AGENT CORRECTNESS")

agent = Agent()

check("agent picks inspect_logs when blind",
      agent.act(Observation(alerts="alert", logs=None, metrics=None, message="")) == "inspect_logs")

check("agent picks check_metrics after logs",
      agent.act(Observation(alerts="alert", logs="some log", metrics=None, message="")) == "check_metrics")

check("agent picks restart_service for memory leak",
      agent.act(Observation(
          alerts="memory alert",
          logs="java.lang.OutOfMemoryError: Java heap space",
          metrics={"memory_usage_pct": 94, "recent_deployments": "None in last 72h"},
          message="",
      )) == "restart_service")

check("agent picks rollback_deployment for bad deploy",
      agent.act(Observation(
          alerts="error spike",
          logs="NullPointerException at PaymentProcessor.java:142",
          metrics={"memory_usage_pct": 52, "recent_deployments": "v2.4.1 deployed 8 min ago"},
          message="",
      )) == "rollback_deployment")

check("agent picks escalate for network issue",
      agent.act(Observation(
          alerts="timeout alert",
          logs="java.net.SocketTimeoutException: Read timed out",
          metrics={"memory_usage_pct": 41, "packet_loss_pct": 38, "recent_deployments": "None"},
          message="",
      )) == "escalate")


section("9. ALL SCENARIOS SOLVABLE END-TO-END")

for scenario_key, scenario_data in SCENARIOS.items():
    env = Environment()
    agent = Agent()

    env.scenario_key     = scenario_key
    env.scenario         = scenario_data
    env.logs_revealed    = False
    env.metrics_revealed = False
    env.done             = False
    env.step_count       = 0
    env.total_reward     = 0
    env.action_history   = []

    obs = Observation(alerts=scenario_data["alerts"], logs=None, metrics=None, message="Incident detected.")

    done = False
    steps = 0
    while not done and steps < 10:
        action = agent.act(obs)
        result = env.step(action)
        obs, done = result.observation, result.done
        steps += 1

    check(f"'{scenario_key}' solved in ≤5 steps",    done and env.total_reward > 0 and steps <= 5,
          f"steps={steps}, score={env.total_reward}")
    check(f"'{scenario_key}' optimal score is +144", env.total_reward == 144, f"got {env.total_reward}")


section("10. MULTIPLE EPISODES — NO STATE LEAKAGE")

env = Environment()
agent = Agent()
scores = []

for episode in range(5):
    obs = env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        result = env.step(action)
        obs, done = result.observation, result.done
    scores.append(env.total_reward)
    check(f"Episode {episode + 1} score is +144", env.total_reward == 144, f"got {env.total_reward}")

check("All 5 episodes scored identically", len(set(scores)) == 1)


print(f"\n{'='*55}")
print(f"  TEST SUMMARY")
print(f"{'='*55}")
print(f"  Total:  {PASS + FAIL}")
print(f"  Passed: {PASS} ✅")
print(f"  Failed: {FAIL} ❌")
print(f"{'='*55}\n")

if FAIL == 0:
    print("  🏆 ALL TESTS PASSED — environment is solid!\n")
else:
    print(f"  ⚠️  {FAIL} test(s) failed — check output above.\n")