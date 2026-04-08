import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import Environment
from agent.policy import Agent


def run_simulation(verbose: bool = True) -> dict:
    env   = Environment()
    agent = Agent()
    obs   = env.reset()
    steps = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"  AI INCIDENT RESPONSE COMMANDER")
        print(f"{'='*60}")
        print(f"  Scenario : {env.get_scenario_name()}")
        print(f"\n  ALERTS\n{obs.alerts}")
        print(f"{'-'*60}")

    done = False
    while not done:
        action = agent.act(obs)

        if verbose:
            print(f"\n  Step {env.step_count + 1} — {action}")

        result = env.step(action)
        obs, reward, done = result.observation, result.reward, result.done

        step_record = {
            "step":             env.step_count,
            "action":           action,
            "reward":           reward,
            "message":          obs.message,
            "logs_revealed":    obs.logs is not None,
            "metrics_revealed": obs.metrics is not None,
            "done":             done,
        }
        if obs.logs:
            step_record["logs"] = obs.logs
        if obs.metrics:
            step_record["metrics"] = obs.metrics

        steps.append(step_record)

        if verbose:
            print(f"  {obs.message}")
            print(f"  Reward: {reward:+.0f}")
            if obs.logs and action == "inspect_logs":
                print(f"\n  LOGS\n{obs.logs}")
            if obs.metrics and action == "check_metrics":
                print(f"\n  METRICS")
                for k, v in obs.metrics.items():
                    print(f"    {k}: {v}")

    if verbose:
        s = env.state()
        print(f"\n{'='*60}")
        print(f"  DONE  |  Episode: {s.episode_id}  |  Steps: {s.step_count}  |  Score: {s.total_reward:+.0f}")
        print(f"{'='*60}\n")

    return {
        "scenario":    env.get_scenario_name(),
        "steps":       steps,
        "total_steps": env.step_count,
        "total_score": env.total_reward,
    }


if __name__ == "__main__":
    run_simulation(verbose=True)