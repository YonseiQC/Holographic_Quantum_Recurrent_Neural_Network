from .base import (
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    SchedulerConfig,
    ModeConfig,
    RegularizationConfig,
    CheckpointConfig,
)

def set_model3_params(config):
    """Sets the parameters for Model 3: Digit Generation"""
    config.model_cfg = ModelConfig(depth=7, n_D=7, n_H=7, seq_len=7)
    config.dataset_cfg = DatasetConfig(
        first_digit=0,
        second_digit=1,
        csv_path="https://raw.githubusercontent.com/w-jiwonchoi/HQRNN_dataset/main/model_3_digit.csv",
    )
    config.training_cfg = TrainingConfig(
        batch_size=32,
        max_epochs=20000,
        learning_rate_init=0.1,
        learning_rate_min=1e-8,
        plot_every_n_epochs=5000,
    )
    config.scheduler_cfg = SchedulerConfig(
        restart_period_epochs=128,
        restart_period_multiplier=1.5,
        restart_decay_rate=0.9,
        warmup_step_ratio=0.1,
        flee_warmup_bind=True,
        fight_end_use_warm_start=True,
    )
    config.mode_cfg = ModeConfig(
        fight_blocks_restart=True,
        fight_in_fight_threshold=0.04,
        fight_in_fight_window=100,
        fight_decay_rate=0.001,
        super_fight_lr_boost=0.2,
        protection_proportion=0.15,
        find_mode_sub_start=0.5,
        find_mode_sub_end=0.25,
        find_fight_rate=5.0,
        patience_num=500,
        cooldown_steps=10,
        stop_condition=10,
        ema_beta=0.95,
        flee_steps=10,
        flee_noise_sigma=0.0,
        volatility_window=100,
        volatility_threshold_start=0.2,
        volatility_threshold_end=0.15,
        volatility_patience=5,
    )
    config.regularization_cfg = RegularizationConfig(weight_decay=0.001, clip_norm=0.0001)
    config.checkpoint_cfg = CheckpointConfig(
        base_checkpoint_dir="runs/model/3",
        max_keep_checkpoints=5,
        allow_hash_mismatch=False,
    )
