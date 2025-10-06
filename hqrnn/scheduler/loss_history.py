from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
from hqrnn.FFF_mode.types import Mode

# --- 4. Loss Tracker

@dataclass
class LossHistory:
    losses: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    epochs: List[int] = field(default_factory=list)
    modes: List[str] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    restarts: List[bool] = field(default_factory=list)

    def add(
        self,
        epoch: int,
        loss_dict: Dict[str, float],
        mode: Mode,
        lr: float,
        restart_happened: bool = False,
    ):
        self.epochs.append(epoch)
        self.modes.append(mode.value)
        self.learning_rates.append(float(lr))
        self.restarts.append(bool(restart_happened))

        for name, value in (loss_dict or {}).items():
            self.losses[name].append(float(value))