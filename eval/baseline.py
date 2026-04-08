# eval/baseline.py
# Baseline inference script — runs every built-in agent against all graders
# and prints a reproducible, comparable score report.
#
# Usage:
#   python eval/baseline.py                  # run all baselines
#   python eval/baseline.py --agent rule     # rule-based only
#   python eval/baseline.py --agent random   # random agent only
#   python eval/baseline.py --seed 99        # custom random seed
#   python eval/baseline.py --runs 5         # repeat N times and average
#
# Adding your own agent:
#   1. Create a class with an act(observation) -> str method
#   2. Register it in BASELINE_AGENTS at the bottom of this file
#   3. Run this script

import sys
import os
import random
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.graders import EasyGrader, MediumGrader, HardGrader, FullGrader
from agent.policy import Agent as RuleBasedAgent
from env.environment import VALID_ACTIONS

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent.llm_agent import LLMAgent  # only for type checker, not runtime

def _make_groq_agent(provider: str = "groq"):
    """Factory that imports and creates LLMAgent at call time — avoids Pylance None issue."""
    from agent.llm_agent import LLMAgent as _LLMAgent
    return _LLMAgent(provider=provider)

try:
    from agent.llm_agent import LLMAgent as _check  # noqa: F401
    _llm_available = True
except Exception:
    _llm_available = False


# ── Baseline agent definitions ────────────────────────────────────────────────

class RandomAgent:
    """
    Picks a random valid action every step.
    Establishes the floor — any real agent should beat this.
    Uses a fixed seed for reproducibility.
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def act(self, observation: dict) -> str:
        return self.rng.choice(VALID_ACTIONS)

    def reset(self):
        pass  # stateless


class AlwaysInspectAgent:
    """
    Always calls inspect_logs, then check_metrics, then guesses restart_service.
    Tests whether a hardcoded strategy can accidentally score well.
    """
    def __init__(self):
        self.step = 0

    def act(self, observation: dict) -> str:
        self.step += 1
        if observation.get("logs") is None:
            return "inspect_logs"
        if observation.get("metrics") is None:
            return "check_metrics"
        return "restart_service"   # always guesses restart — wrong 2/3 of the time

    def reset(self):
        self.step = 0


class AlwaysEscalateAgent:
    """
    Investigates properly then always escalates regardless of evidence.
    Correct for network_issue, wrong for the other two.
    Useful sanity check: shows what "escalate-biased" looks like.
    """
    def act(self, observation: dict) -> str:
        if observation.get("logs") is None:
            return "inspect_logs"
        if observation.get("metrics") is None:
            return "check_metrics"
        return "escalate"

    def reset(self):
        pass


class OptimalRuleAgent:
    """
    Wrapper around our rule-based agent — this is the gold standard baseline.
    Should score 1.0 on easy and medium, and close to 1.0 on hard.
    """
    def __init__(self):
        self._agent = RuleBasedAgent()

    def act(self, observation: dict) -> str:
        return self._agent.act(observation)

    def reset(self):
        pass


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _grade_once(agent, grader_cls):
    """Grade one agent with one grader. Returns the result dict."""
    if hasattr(agent, "reset"):
        agent.reset()
    return grader_cls().grade(agent)


def _average_results(results: list) -> dict:
    """Average a list of grader result dicts into one."""
    if not results:
        return {}
    avg = results[0].copy()
    avg["score"] = round(sum(r["score"] for r in results) / len(results), 4)
    return avg


def run_baseline(agent, agent_name: str, runs: int = 1, seed: int = 42) -> dict:
    """
    Run all three graders against an agent, repeat `runs` times, return averages.
    Seeding the RandomAgent ensures reproducibility across identical runs.
    """
    grader_classes = [EasyGrader, MediumGrader, HardGrader]
    all_runs = {g.__name__: [] for g in grader_classes}

    for run_idx in range(runs):
        # Re-seed random agents so each run is reproducible
        if hasattr(agent, "rng"):
            agent.rng = random.Random(seed + run_idx)

        for grader_cls in grader_classes:
            if hasattr(agent, "reset"):
                agent.reset()
            result = grader_cls().grade(agent)
            all_runs[grader_cls.__name__].append(result)

    # Average per grader
    averaged = {}
    for grader_name, results in all_runs.items():
        averaged[grader_name] = _average_results(results)

    # Weighted overall (easy 20%, medium 35%, hard 45%)
    weights = {"EasyGrader": 0.20, "MediumGrader": 0.35, "HardGrader": 0.45}
    overall = sum(
        averaged[g]["score"] * weights[g]
        for g in weights
    )

    return {
        "agent":       agent_name,
        "runs":        runs,
        "seed":        seed,
        "overall":     round(overall, 4),
        "easy":        averaged["EasyGrader"],
        "medium":      averaged["MediumGrader"],
        "hard":        averaged["HardGrader"],
    }


# ── Pretty printer ────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 20) -> str:
    """Render a visual score bar like ████████░░░░ 0.80"""
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def _grade_label(score: float) -> str:
    if score >= 0.90: return "S  (Excellent)"
    if score >= 0.75: return "A  (Good)"
    if score >= 0.55: return "B  (Acceptable)"
    if score >= 0.35: return "C  (Weak)"
    return                   "F  (Failing)"


def print_agent_report(result: dict, verbose: bool = True):
    """Print a formatted report for one agent."""
    w = 62
    print("\n" + "═" * w)
    print(f"  Agent : {result['agent']}")
    print(f"  Runs  : {result['runs']}  |  Seed: {result['seed']}")
    print("─" * w)

    levels = [
        ("Easy   (memory_leak)",    result["easy"]),
        ("Medium (bad_deployment)", result["medium"]),
        ("Hard   (network_issue)",  result["hard"]),
    ]

    for label, r in levels:
        score = r["score"]
        fix   = "✅" if r["correct_fix"] else ("❌" if r["wrong_fix"] else "⬜")
        print(f"  {label:<26} {fix}  {_bar(score)}  {score:.3f}")

        if verbose:
            for note in r.get("notes", []):
                print(f"      ↳ {note}")
            print(f"      Actions: {' → '.join(r['action_history'])}")
            print()

    print("─" * w)
    overall = result["overall"]
    print(f"  Overall Score   {_bar(overall)}  {overall:.3f}")
    print(f"  Grade           {_grade_label(overall)}")
    print("═" * w)


def print_comparison_table(results: list):
    """Print a side-by-side comparison table of all agents."""
    w = 72
    print("\n\n" + "═" * w)
    print("  BASELINE COMPARISON TABLE")
    print("─" * w)
    print(f"  {'Agent':<26} {'Easy':>8} {'Medium':>8} {'Hard':>8} {'Overall':>10}  Grade")
    print("─" * w)

    # Sort by overall descending
    for r in sorted(results, key=lambda x: x["overall"], reverse=True):
        print(
            f"  {r['agent']:<26}"
            f"  {r['easy']['score']:>6.3f}"
            f"  {r['medium']['score']:>6.3f}"
            f"  {r['hard']['score']:>6.3f}"
            f"  {r['overall']:>8.3f}"
            f"  {_grade_label(r['overall'])}"
        )

    print("═" * w)
    print("  Weights: Easy=20%  Medium=35%  Hard=45%")
    print("═" * w + "\n")


# ── Registered baselines ──────────────────────────────────────────────────────

BASELINE_AGENTS = {
    "random":         lambda seed: RandomAgent(seed=seed),
    "always_restart":  lambda seed: AlwaysInspectAgent(),
    "always_escalate": lambda seed: AlwaysEscalateAgent(),
    "rule":           lambda seed: OptimalRuleAgent(),
}

# Register Groq LLM agent only if the package + key are available
if _llm_available and os.environ.get("GROQ_API_KEY"):
    BASELINE_AGENTS["groq"] = lambda seed: _make_groq_agent(provider="groq")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline inference script — grades agents on easy/medium/hard scenarios"
    )
    parser.add_argument(
        "--agent", default="all",
        choices=list(BASELINE_AGENTS.keys()) + ["all"],
        help="Which agent to run (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="How many times to repeat each grader and average (default: 1)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Show per-step notes and action history (default: True)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only show the comparison table, no per-agent detail"
    )
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    print("\n" + "=" * 62)
    print("  AI INCIDENT RESPONSE COMMANDER — BASELINE EVALUATION")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  seed={args.seed}  |  runs={args.runs}")
    print("=" * 62)

    agents_to_run = (
        list(BASELINE_AGENTS.keys())
        if args.agent == "all"
        else [args.agent]
    )

    all_results = []

    for agent_name in agents_to_run:
        print(f"\n  ⏳ Grading: {agent_name}...")
        t0 = time.time()
        agent  = BASELINE_AGENTS[agent_name](args.seed)
        result = run_baseline(agent, agent_name, runs=args.runs, seed=args.seed)
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 2)
        all_results.append(result)

        if verbose:
            print_agent_report(result, verbose=True)
        else:
            print(f"     Done in {elapsed:.2f}s — overall: {result['overall']:.3f}")

    if len(all_results) > 1:
        print_comparison_table(all_results)

    # Machine-readable summary line for CI/logging
    print("SCORES: " + " | ".join(
        f"{r['agent']}={r['overall']:.3f}" for r in all_results
    ))


if __name__ == "__main__":
    main()