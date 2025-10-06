from enum import Enum
from dataclasses import dataclass
from collections import deque
from hqrnn.config.base import Config

# --- 2. 3-Mode (Find - Flee - Fight) System

class Mode(Enum):
    FIND = "find"  # Wide search for good candidates
    FIGHT = "fight"  # Intensify work on the chosen candidate
    FLEE = "flee"  # Shift to a fresh search area


@dataclass
class UnifiedModeState:
    config: Config
    mode: Mode = Mode.FIND  # Current mode

    best_loss: float = float('inf')
    last_loss: float = float('inf')
    ema_loss: float = float('inf')
    no_improve: int = 0  # Steps without improvement in fight
    cooldown: int = 0  # Steps to wait before another mode change

    flee_left: int = 0
    fight_count: int = 0
    flee_count: int = 0

    finished: bool = False

    fight_entry_lr: float = 0.0  # LR at fight entry
    volatility_strikes: int = 0  # Consecutive volatility triggers in find

    is_super_fight: bool = False  # Super-fight
    super_fight_entry_epoch: int = -1
    super_fight_duration: int = 0

    cycle_warmup_steps_hint: int = 0
    fight_start_epoch: int = -1
    fight_end_epoch: int = -1
    last_restart_warm_start: bool = False

    def __post_init__(self):
        self.recent_losses = deque(maxlen=self.config.mode_cfg.volatility_window)  # Rolling window for volatility
        self.fight_best_hits = deque(maxlen=self.config.mode_cfg.fight_in_fight_window)  # Improvements history in fight


@dataclass
class ModeUpdateResult:
    mode_changed: bool = False
    should_save_best: bool = False
    should_generate_flee_samples: bool = False
    triggered_by_volatility: bool = False
    request_warm_start_after_fight: bool = False
    flee_bound_to_warmup: bool = False