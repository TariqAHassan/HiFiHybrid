"""

    Training

"""
from __future__ import annotations
from typing import Any


def _none_double(value: Any) -> Any:
    if value is None:
        return None, None
    else:
        return value


class BestMetricTracker:
    def __init__(self, metrics: dict[str, str]) -> None:
        self.metrics = metrics

        for k, v in metrics.items():
            if v not in ("min", "max"):
                raise ValueError(f"Got invalid direction for '{k}'")

        self.best = dict.fromkeys(self.metrics)

    def __contains__(self, item: str) -> bool:
        return item in self.metrics

    def update(self, name: str, value: int | float, current_step: int) -> bool:
        last_value, last_step = _none_double(self.best[name])
        if last_step is not None and last_step == current_step:
            return False
        elif (
            last_value is None
            or (self.metrics[name] == "max" and value > last_value)
            or (self.metrics[name] == "min" and value < last_value)
        ):
            self.best[name] = (value, current_step)
            return True
        return False
