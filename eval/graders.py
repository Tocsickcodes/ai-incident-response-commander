# eval/graders.py
# Agent grading system — Easy / Medium / Hard
# Each grader runs the agent on a fixed scenario and returns a score 0.0 – 1.0
#
# Score breakdown philosophy:
#   - Did it pick the correct fix?           (correctness)
#   - Did it investigate before acting?      (process quality)
#   - How efficiently did it get there?      (step economy)
#   - Did it avoid wasted / harmful actions? (discipline)
#
# Easy:   single clear signal, forgiving scoring
# Medium: requires BOTH logs + metrics before deciding, stricter efficiency
# Hard:   misleading signals, noisy alerts, penalises wrong fix heavily

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import Environment, VALID_ACTIONS
from env.scenarios import SCENARIOS


# ── Shared helpers ────────────────────────────────────────────────────────────

def _run_episode(agent, scenario_key: str) -> dict:
    """
    Run one episode on a fixed scenario (bypasses random selection).
    Returns a dict of everything we need to grade.
    """
    env = Environment()

    # Force a specific scenario instead of random
    env.scenario_key = scenario_key
    env.scenario     = SCENARIOS[scenario_key]
    env.logs_revealed    = False
    env.metrics_revealed = False
    env.done         = False
    env.step_count   = 0
    env.total_reward = 0
    env.action_history = []

    obs = {
        "alerts":  SCENARIOS[scenario_key]["alerts"],
        "logs":    None,
        "metrics": None,
        "message": "Incident detected. Start investigating.",
    }

    steps          = []
    wrong_fix      = False
    correct_fix    = False
    investigated   = False   # checked at least logs OR metrics before fixing

    done = False
    while not done and env.step_count < 15:   # safety cap
        result = agent.act(obs)
        # LLMAgent returns (action, reasoning) tuple; rule-based returns a plain string
        action = result[0] if isinstance(result, tuple) else result

        obs, reward, done = env.step(action)

        steps.append({
            "step":   env.step_count,
            "action": action,
            "reward": reward,
        })

        if action in ["restart_service", "rollback_deployment", "escalate"]:
            if action == SCENARIOS[scenario_key]["correct_action"]:
                correct_fix  = True
                investigated = env.logs_revealed or env.metrics_revealed
            else:
                wrong_fix = True

    return {
        "scenario_key":       scenario_key,
        "correct_fix":        correct_fix,
        "wrong_fix":          wrong_fix,
        "investigated":       investigated,
        "logs_checked":       env.logs_revealed,
        "metrics_checked":    env.metrics_revealed,
        "step_count":         env.step_count,
        "total_reward":       env.total_reward,
        "action_history":     env.action_history,
        "correct_action":     SCENARIOS[scenario_key]["correct_action"],
        "steps":              steps,
    }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, round(value, 3)))


# ── EASY GRADER ───────────────────────────────────────────────────────────────
#
# Scenario: memory_leak  (clearest signal — OOMError is unmistakable)
# Passing bar: chose the correct fix
# Scoring:
#   0.50  base for any correct fix
#   +0.20 investigated before fixing (not just guessing)
#   +0.20 checked BOTH logs and metrics
#   +0.10 finished in ≤ 4 steps (lean investigation)
#   −0.40 wrong fix applied (heavy penalty even on easy)

class EasyGrader:
    """
    Easy grader — memory_leak scenario.
    Signal is strong and unambiguous: OutOfMemoryError in logs + 94% memory in metrics.
    A correct agent should pass with score ≥ 0.70.
    """

    SCENARIO   = "memory_leak"
    DIFFICULTY = "easy"

    def grade(self, agent) -> dict:
        result = _run_episode(agent, self.SCENARIO)
        score  = 0.0
        notes  = []

        if result["wrong_fix"]:
            score -= 0.40
            notes.append("Wrong fix applied on an easy scenario (-0.40)")
        elif result["correct_fix"]:
            score += 0.50
            notes.append("Correct fix: restart_service (+0.50)")

            if result["investigated"]:
                score += 0.20
                notes.append("Investigated before fixing (+0.20)")
            else:
                notes.append("Fix applied without any investigation (0.00)")

            if result["logs_checked"] and result["metrics_checked"]:
                score += 0.20
                notes.append("Checked both logs and metrics (+0.20)")
            elif result["logs_checked"] or result["metrics_checked"]:
                score += 0.10
                notes.append("Checked only one data source (+0.10)")

            if result["step_count"] <= 4:
                score += 0.10
                notes.append(f"Efficient: finished in {result['step_count']} steps (+0.10)")
            else:
                notes.append(f"Slow: took {result['step_count']} steps (0.00)")
        else:
            notes.append("Did not reach a fix action within step limit (0.00)")

        return {
            "difficulty":     self.DIFFICULTY,
            "scenario":       self.SCENARIO,
            "score":          _clamp(score),
            "correct_fix":    result["correct_fix"],
            "wrong_fix":      result["wrong_fix"],
            "steps":          result["step_count"],
            "action_history": result["action_history"],
            "notes":          notes,
        }


# ── MEDIUM GRADER ─────────────────────────────────────────────────────────────
#
# Scenario: bad_deployment  (requires reading BOTH logs and metrics to be sure)
# Passing bar: correct fix + checked both sources
# Scoring:
#   0.40  correct fix
#   +0.25 checked logs (mandatory for this scenario)
#   +0.25 checked metrics (mandatory for this scenario)
#   +0.10 finished in ≤ 4 steps
#   −0.50 wrong fix (harder penalty — medium level should not guess)

class MediumGrader:
    """
    Medium grader — bad_deployment scenario.
    Both logs (NullPointerException) AND metrics (recent deployment) are needed
    to distinguish a bad deploy from a code bug that existed before.
    A correct agent should score ≥ 0.75 only if it checked both sources.
    """

    SCENARIO   = "bad_deployment"
    DIFFICULTY = "medium"

    def grade(self, agent) -> dict:
        result = _run_episode(agent, self.SCENARIO)
        score  = 0.0
        notes  = []

        # Process quality — must check both sources on medium
        if result["logs_checked"]:
            score += 0.25
            notes.append("Checked logs (+0.25)")
        else:
            notes.append("Did not check logs — critical gap (0.00)")

        if result["metrics_checked"]:
            score += 0.25
            notes.append("Checked metrics (+0.25)")
        else:
            notes.append("Did not check metrics — critical gap (0.00)")

        # Correctness
        if result["wrong_fix"]:
            score -= 0.50
            notes.append("Wrong fix on medium scenario (-0.50)")
        elif result["correct_fix"]:
            score += 0.40
            notes.append("Correct fix: rollback_deployment (+0.40)")

            if result["step_count"] <= 4:
                score += 0.10
                notes.append(f"Efficient: {result['step_count']} steps (+0.10)")
            else:
                notes.append(f"Took {result['step_count']} steps — no efficiency bonus (0.00)")
        else:
            notes.append("Did not reach a fix action (0.00)")

        return {
            "difficulty":     self.DIFFICULTY,
            "scenario":       self.SCENARIO,
            "score":          _clamp(score),
            "correct_fix":    result["correct_fix"],
            "wrong_fix":      result["wrong_fix"],
            "steps":          result["step_count"],
            "action_history": result["action_history"],
            "notes":          notes,
        }


# ── HARD GRADER ───────────────────────────────────────────────────────────────
#
# Scenario: network_issue  (most deceptive — logs show errors but service is healthy)
# The hard scenario tricks agents that jump to restart_service or rollback_deployment
# because error rates are high. The correct answer (escalate) requires recognising
# that the PROBLEM IS EXTERNAL — packet loss on the subnet, not the service itself.
#
# Scoring:
#   0.35  correct fix (escalate — hardest to reach)
#   +0.20 checked logs (required — must see SocketTimeoutException)
#   +0.20 checked metrics (required — must see packet_loss_pct)
#   +0.15 did NOT try restart or rollback first (avoided the trap)
#   +0.10 finished in ≤ 5 steps
#   −0.60 wrong fix (very heavy — hard scenario should not be guessed)

class HardGrader:
    """
    Hard grader — network_issue scenario.
    The trap: high error rate + timeouts LOOK like a service problem,
    but the service itself is fine (41% memory, 29% CPU).
    The real cause is packet loss on the subnet — needs escalation to network team.
    Agents that restart or rollback are penalised heavily.
    """

    SCENARIO   = "network_issue"
    DIFFICULTY = "hard"

    def grade(self, agent) -> dict:
        result = _run_episode(agent, self.SCENARIO)
        score  = 0.0
        notes  = []

        # Did the agent fall into the trap actions?
        trap_actions = {"restart_service", "rollback_deployment"}
        hit_trap     = any(a in trap_actions for a in result["action_history"])

        # Investigation quality
        if result["logs_checked"]:
            score += 0.20
            notes.append("Checked logs (+0.20)")
        else:
            notes.append("Did not check logs (0.00)")

        if result["metrics_checked"]:
            score += 0.20
            notes.append("Checked metrics — key to seeing packet_loss (+0.20)")
        else:
            notes.append("Did not check metrics — missed packet_loss_pct clue (0.00)")

        # Avoided traps
        if not hit_trap:
            score += 0.15
            notes.append("Avoided trap actions (restart/rollback) (+0.15)")
        else:
            notes.append("Fell into trap — tried restart or rollback on a network issue (0.00)")

        # Correctness
        if result["wrong_fix"]:
            score -= 0.60
            notes.append("Wrong fix on hard scenario (-0.60)")
        elif result["correct_fix"]:
            score += 0.35
            notes.append("Correct fix: escalate — identified external network cause (+0.35)")

            if result["step_count"] <= 5:
                score += 0.10
                notes.append(f"Efficient: {result['step_count']} steps (+0.10)")
            else:
                notes.append(f"Took {result['step_count']} steps (0.00)")
        else:
            notes.append("Did not reach a fix action (0.00)")

        return {
            "difficulty":     self.DIFFICULTY,
            "scenario":       self.SCENARIO,
            "score":          _clamp(score),
            "correct_fix":    result["correct_fix"],
            "wrong_fix":      result["wrong_fix"],
            "steps":          result["step_count"],
            "action_history": result["action_history"],
            "notes":          notes,
        }


# ── Composite grader — runs all three and returns overall score ───────────────

class FullGrader:
    """
    Runs Easy + Medium + Hard graders and computes a weighted overall score.
    Weights reflect difficulty: Easy=20%, Medium=35%, Hard=45%
    """

    WEIGHTS = {"easy": 0.20, "medium": 0.35, "hard": 0.45}

    def __init__(self):
        self.graders = [EasyGrader(), MediumGrader(), HardGrader()]

    def grade(self, agent) -> dict:
        results = [g.grade(agent) for g in self.graders]

        weighted_score = sum(
            r["score"] * self.WEIGHTS[r["difficulty"]]
            for r in results
        )

        return {
            "overall_score": _clamp(weighted_score),
            "weights":       self.WEIGHTS,
            "breakdown":     results,
        }