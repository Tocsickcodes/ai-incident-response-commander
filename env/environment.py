import random
import uuid
from typing import Optional

from .scenarios import SCENARIOS
from models import Action, Observation, State, StepResult, VALID_ACTIONS


class Environment:

    def __init__(self):
        self.scenario:          Optional[dict] = None
        self.scenario_key:      Optional[str]  = None
        self.logs_revealed:     bool  = False
        self.metrics_revealed:  bool  = False
        self.done:              bool  = False
        self.step_count:        int   = 0
        self.total_reward:      float = 0.0
        self.action_history:    list  = []
        self._episode_id:       str   = ""

    def reset(self) -> Observation:
        self.scenario_key     = random.choice(list(SCENARIOS.keys()))
        self.scenario         = SCENARIOS[self.scenario_key]
        self.logs_revealed    = False
        self.metrics_revealed = False
        self.done             = False
        self.step_count       = 0
        self.total_reward     = 0.0
        self.action_history   = []
        self._episode_id      = str(uuid.uuid4())[:8]

        return Observation(
            alerts  = self.scenario["alerts"],
            logs    = None,
            metrics = None,
            message = "Incident detected. Start investigating.",
        )

    def step(self, action) -> StepResult:
        action_str = action.action_str if isinstance(action, Action) else str(action)

        if self.scenario is None:
            raise RuntimeError("No scenario loaded. Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode is over. Call reset() to start a new one.")
        if action_str not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action_str}'. Choose from {VALID_ACTIONS}")

        self.step_count += 1
        self.action_history.append(action_str)

        reward  = -2.0
        message = ""

        alerts  = self.scenario["alerts"]
        logs    = self.scenario["logs"]    if self.logs_revealed    else None
        metrics = self.scenario["metrics"] if self.metrics_revealed else None

        if action_str == "inspect_logs":
            if not self.logs_revealed:
                self.logs_revealed = True
                logs    = self.scenario["logs"]
                message = "Logs retrieved successfully."
            else:
                reward -= 10.0
                message = "Logs already retrieved. Unnecessary action."

        elif action_str == "check_metrics":
            if not self.metrics_revealed:
                self.metrics_revealed = True
                metrics = self.scenario["metrics"]
                message = "Metrics retrieved successfully."
            else:
                reward -= 10.0
                message = "Metrics already retrieved. Unnecessary action."

        elif action_str in ["restart_service", "rollback_deployment", "escalate"]:
            if not self.logs_revealed and not self.metrics_revealed:
                reward -= 10.0
                message = "Attempted fix without any investigation."
            else:
                correct = self.scenario["correct_action"]
                if action_str == correct:
                    reward   += 150.0
                    message   = f"Correct fix! Root cause: {self.scenario['root_cause'].replace('_', ' ').title()}"
                    self.done = True
                else:
                    reward   -= 30.0
                    message   = f"Wrong fix. Correct action was: {correct.replace('_', ' ')}"
                    self.done = True

        self.total_reward += reward

        observation = Observation(
            alerts  = alerts,
            logs    = logs,
            metrics = metrics,
            message = message,
            done    = self.done,
            reward  = reward,
        )

        return StepResult(observation=observation, reward=reward, done=self.done)

    def state(self) -> State:
        if self.scenario is None:
            return State(
                episode_id="", step_count=0, scenario_name="none",
                scenario_key="none", logs_revealed=False, metrics_revealed=False,
                total_reward=0.0, action_history=[], done=False, correct_action="unknown",
            )

        return State(
            episode_id       = self._episode_id,
            step_count       = self.step_count,
            scenario_name    = self.scenario["name"],
            scenario_key     = self.scenario_key or "",
            logs_revealed    = self.logs_revealed,
            metrics_revealed = self.metrics_revealed,
            total_reward     = self.total_reward,
            action_history   = list(self.action_history),
            done             = self.done,
            correct_action   = self.scenario["correct_action"],
        )

    def get_scenario_name(self) -> str:
        return self.scenario["name"] if self.scenario else "None"