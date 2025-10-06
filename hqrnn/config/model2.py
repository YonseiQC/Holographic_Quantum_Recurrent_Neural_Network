from .base import (
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    SchedulerConfig,
    ModeConfig,
    RegularizationConfig,
    CheckpointConfig,
)

def set_model2_params(config):
    """Sets the parameters for Model 2: Power Demand"""
    config.model_cfg = ModelConfig(depth=5, n_D=6, n_H=6, seq_len=5)
    config.dataset_cfg = DatasetConfig(
        start_year=2024,
        start_month=1,
        start_day=3,
        target_year=2024,
        target_month=12,
        target_day=23,
        target_weekday=0,
        normalize_exponent=6,
        csv_path="https://raw.githubusercontent.com/w-jiwonchoi/HQRNN_dataset/main/model_2_power.csv",
    )
    config.training_cfg = TrainingConfig(
        batch_size=4,
        max_epochs=100,
        learning_rate_init=0.1,
        learning_rate_min=1e-6,
        plot_every_n_epochs=1000,
    )
    config.scheduler_cfg = SchedulerConfig(
        restart_period_epochs=120,
        restart_period_multiplier=2.0,
        restart_decay_rate=0.85,
        warmup_step_ratio=0.05,
        flee_warmup_bind=True,
        fight_end_use_warm_start=True,
    )
    config.mode_cfg = ModeConfig(
        fight_blocks_restart=True,
        fight_in_fight_threshold=0.2,
        fight_in_fight_window=60,
        fight_decay_rate=0.005,
        super_fight_lr_boost=0.5,
        protection_proportion=0.005,
        find_mode_sub_start=0.3,
        find_mode_sub_end=0.25,
        find_fight_rate=2.0,
        patience_num=150,
        cooldown_steps=40,
        stop_condition=15,
        ema_beta=0.95,
        flee_steps=10,
        flee_noise_sigma=0.01,
        volatility_window=100,
        volatility_threshold_start=0.12,
        volatility_threshold_end=0.08,
        volatility_patience=20,
    )
    config.regularization_cfg = RegularizationConfig(weight_decay=0.0, clip_norm=1e-6)
    config.checkpoint_cfg = CheckpointConfig(
        base_checkpoint_dir="runs/model/2",
        max_keep_checkpoints=5,
        allow_hash_mismatch=False,
    )