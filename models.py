from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


VALID_ACTIONS = [
    "inspect_logs",
    "check_metrics",
    "restart_service",
    "rollback_deployment",
    "escalate",
]


@dataclass
class Action:
    action_str: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.action_str not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{self.action_str}'. Must be one of: {VALID_ACTIONS}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    alerts:  str
    logs:    Optional[str]
    metrics: Optional[Dict[str, Any]]
    message: str
    done:    bool  = False
    reward:  float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            alerts  = data.get("alerts",  ""),
            logs    = data.get("logs"),
            metrics = data.get("metrics"),
            message = data.get("message", ""),
            done    = data.get("done",    False),
            reward  = float(data.get("reward", 0.0)),
        )


@dataclass
class State:
    episode_id:       str
    step_count:       int
    scenario_name:    str
    scenario_key:     str
    logs_revealed:    bool
    metrics_revealed: bool
    total_reward:     float
    action_history:   List[str]
    done:             bool
    correct_action:   str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepResult:
    observation: Observation
    reward:      float
    done:        bool

    def __iter__(self):
        yield self.observation
        yield self.reward
        yield self.done

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "reward":      self.reward,
            "done":        self.done,
        }