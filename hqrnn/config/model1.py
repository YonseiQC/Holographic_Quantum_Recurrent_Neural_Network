from .base import (
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    SchedulerConfig,
    ModeConfig,
    RegularizationConfig,
    CheckpointConfig,
)

def set_model1_params(config):
    """Sets the parameters for Model 1: S&P500"""
    config.model_cfg = ModelConfig(depth=6, n_D=6, n_H=6, seq_len=6)
    config.dataset_cfg = DatasetConfig(
        total_days=200,
        start_year=2024,
        start_month=1,
        start_day=5,
        normalize_exponent=6,
        csv_path="https://raw.githubusercontent.com/w-jiwonchoi/HQRNN_dataset/main/model_1_snp.csv",
    )
    config.training_cfg = TrainingConfig(
        batch_size=64,
        max_epochs=1000,
        learning_rate_init=0.01,
        learning_rate_min=1e-6,
        plot_every_n_epochs=500,
    )
    config.scheduler_cfg = SchedulerConfig(
        restart_period_epochs=20,
        restart_period_multiplier=2.0,
        restart_decay_rate=0.9,
        warmup_step_ratio=0.05,
        flee_warmup_bind=True,
        fight_end_use_warm_start=True,
    )
    config.mode_cfg = ModeConfig(
        fight_blocks_restart=True,
        fight_in_fight_threshold=0.2,
        fight_in_fight_window=40,
        fight_decay_rate=0.05,
        super_fight_lr_boost=0.5,
        protection_proportion=0.003,
        find_mode_sub_start=0.3,
        find_mode_sub_end=0.25,
        find_fight_rate=5.0,
        patience_num=200,
        cooldown_steps=10,
        stop_condition=10,
        ema_beta=0.90,
        flee_steps=0,
        flee_noise_sigma=0.0,
        volatility_window=50,
        volatility_threshold_start=0.30,
        volatility_threshold_end=0.20,
        volatility_patience=5,
    )
    config.regularization_cfg = RegularizationConfig(weight_decay=1e-6, clip_norm=1e-8)
    config.checkpoint_cfg = CheckpointConfig(
        base_checkpoint_dir="runs/model/1",
        max_keep_checkpoints=5,
        allow_hash_mismatch=False,
    )
