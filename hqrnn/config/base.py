# --- 1. Hyperparameters

import json
import hashlib
from dataclasses import dataclass, asdict, field
from typing import Optional

# ------ 1-1. Hyperparameter Settings

@dataclass
class ModelConfig:
    depth: int  # Ansatz depth
    n_D: int  # Number of qubits in the data register
    n_H: int  # Number of qubits in the hidden register
    seq_len: int  # Length of a single sequence (time step)

    @property
    def n_qubits(self) -> int:
        return self.n_D + self.n_H  # Total number of qubits


@dataclass
class DatasetConfig:
    csv_path: str
    # (Model 1: S&P500)
    total_days: Optional[int] = None
    map_exponent: Optional[int] = None  # Classify by grouping the 2**n_D normalized values into buckets of size 2**map_exponent
    # (Model 1: S&P500 & Model 2: Power)
    start_year: Optional[int] = None
    start_month: Optional[int] = None
    start_day: Optional[int] = None
    normalize_exponent: Optional[int] = None  # Defines the representable rate range
    # (Model 2: Power)
    target_year: Optional[int] = None
    target_month: Optional[int] = None
    target_day: Optional[int] = None
    target_weekday: Optional[int] = None  # 0 = Monday ~ 6 = Sunday
    # (Model 3: Digits)
    first_digit: Optional[int] = None
    second_digit: Optional[int] = None


@dataclass
class TrainingConfig:
    batch_size: int
    max_epochs: int
    learning_rate_init: float  # First peak of the learning rate
    learning_rate_min: float  # Lower bound of the learning rate
    plot_every_n_epochs: int  # Plotting interval (epochs) if not using the complex scheduler


@dataclass
class SchedulerConfig:
    restart_period_epochs: int  # Restart period (epochs) used in 'find' mode
    restart_period_multiplier: float  # Next period = previous period * multiplier
    restart_decay_rate: float  # In 'find' mode: next peak LR = previous peak LR * decay_rate
    warmup_step_ratio: float  # Warmup steps = fight-mode duration * ratio
    flee_warmup_bind: bool  # If True: warmup steps are bound to flee steps (equal)
    fight_end_use_warm_start: bool  # If True: after fight, LR linearly warms to the new peak LR


@dataclass
class LossConfig:
    mmd_sigma: float = 2.5  # RBF kernel sigma
    mmd_lambda: float = 0.75  # Weight for class-discrimination term (pure MMD = MMD between real class 0 and 1)
    mmd_k: float = 15.0  # Temperature for smooth-max when combining class-0 and class-1 losses


@dataclass
class ModeConfig:
    fight_blocks_restart: bool  # Prevent restarts while in fight mode
    fight_in_fight_threshold: float  # Threshold to enter super-fight from fight
    fight_in_fight_window: int  # Window size used to evaluate the super-fight trigger
    fight_decay_rate: float  # Exponential decay rate of LR while in fight mode
    super_fight_lr_boost: float  # Multiplier applied to LR in super-fight
    protection_proportion: float  # Early protection phase proportion of total training
    find_mode_sub_start: float  # Initial relative drop threshold to switch to fight
    find_mode_sub_end: float  # Final relative drop threshold (annealed over training)
    find_fight_rate: float  # LR ratio: find LR / fight LR
    patience_num: int  # Steps without improvement before fleeing from fight
    cooldown_steps: int  # Cooldown steps after a mode change
    stop_condition: int  # Maximum number of flee events before stopping
    ema_beta: float  # EMA smoothing factor for loss
    flee_steps: int  # Fixed flee steps if binding is disabled
    flee_noise_sigma: float  # Std of noise added during flee (if used)
    volatility_window: int  # Window size for volatility detection
    volatility_threshold_start: float  # Initial CV threshold for volatility trigger
    volatility_threshold_end: float  # Final CV threshold (annealed over training)
    volatility_patience: int  # Number of consecutive volatility strikes before triggering


@dataclass
class RegularizationConfig:
    weight_decay: float
    clip_norm: float


@dataclass
class CheckpointConfig:
    base_checkpoint_dir: str
    max_keep_checkpoints: int
    allow_hash_mismatch: bool


@dataclass
class ExperimentConfig:
    loss_function: str  # "nll" or "mmd"
    collapse_type: str  # "soft" or "hard"
    learning_mode: str  # "teacher_forcing" or "autoregressive"


@dataclass
class SchedulerToggleConfig:
    use_complex_scheduler: bool = False  # True: complex scheduler; False: cosine annealing without restarts


@dataclass
class Config:
    model: int = 1  # 1: S&P500, 2: Power Demand, 3: Digit Generation

    model_cfg: ModelConfig = field(init=False)
    dataset_cfg: DatasetConfig = field(init=False)
    training_cfg: TrainingConfig = field(init=False)
    scheduler_cfg: SchedulerConfig = field(init=False)
    scheduler_toggle_cfg: SchedulerToggleConfig = field(default_factory=SchedulerToggleConfig)
    mode_cfg: ModeConfig = field(init=False)
    regularization_cfg: RegularizationConfig = field(init=False)
    checkpoint_cfg: CheckpointConfig = field(init=False)
    loss_cfg: LossConfig = field(default_factory=LossConfig)
    exp_cfg: ExperimentConfig = field(init=False)

    def __post_init__(self):
        # Dynamically import and set parameters based on the model number
        if self.model == 1:  # S&P500
            from .model1 import set_model1_params
            set_model1_params(self)
        elif self.model == 2:  # Power Demand
            from .model2 import set_model2_params
            set_model2_params(self)
        elif self.model == 3:  # Digit Generation
            from .model3 import set_model3_params
            set_model3_params(self)
        else:
            raise ValueError(f"Invalid model: {self.model}. Model must be 1, 2, or 3.")

    # ------ 1-3. Other Settings (Hashing)

    def to_dict(self):
        return asdict(self)

    def get_hash(self) -> str:
        d = self.to_dict()
        d.pop("model", None)

        if "training_cfg" in d:
            d["training_cfg"].pop("max_epochs", None)

        if "checkpoint_cfg" in d:
            d["checkpoint_cfg"].pop("base_checkpoint_dir", None)
            d["checkpoint_cfg"].pop("max_keep_checkpoints", None)
            d["checkpoint_cfg"].pop("allow_hash_mismatch", None)

        if "mode_cfg" in d:
            d["mode_cfg"].pop("protection_proportion", None)

        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]