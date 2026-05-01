from dataclasses import dataclass
from math import exp


@dataclass(frozen=True)
class ExplorationRateSchedule:
    schedule: str
    start: float
    end: float
    decay_steps: int

    def __post_init__(self) -> None:
        if self.schedule not in {"constant", "linear", "exponential"}:
            msg = "exploration.schedule must be one of: constant, linear, exponential."
            raise ValueError(msg)
        if not 0.0 <= self.start <= 1.0:
            raise ValueError("exploration.start must be in [0, 1].")
        if not 0.0 <= self.end <= 1.0:
            raise ValueError("exploration.end must be in [0, 1].")
        if self.schedule != "constant" and self.decay_steps <= 0:
            raise ValueError("exploration.decay_steps must be positive.")

    def value(self, step_index: int) -> float:
        if step_index < 0:
            raise ValueError("step_index must be non-negative.")

        if self.schedule == "constant":
            return self.start

        if self.schedule == "linear":
            progress = min(step_index / self.decay_steps, 1.0)
            return self.start + progress * (self.end - self.start)

        return self.end + (self.start - self.end) * exp(-step_index / self.decay_steps)
