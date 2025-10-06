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
    config.model_cfg = ModelConfig(depth=8, n_D=7, n_H=7, seq_len=7)
    config.dataset_cfg = DatasetConfig(
        first_digit=0,
        second_digit=1,
        csv_path="https://raw.githubusercontent.com/w-jiwonchoi/HQRNN_dataset/main/model_3_digit.csv",
    )
    config.training_cfg = TrainingConfig(
        batch_size=32,
        max_epochs=10000,
        learning_rate_init=0.1,
        learning_rate_min=1e-8,
        plot_every_n_epochs=1000,
    )
    config.scheduler_cfg = SchedulerConfig(
        restart_period_epochs=100,
        restart_period_multiplier=2.0,
        restart_decay_rate=0.9,
        warmup_step_ratio=0.05,
        flee_warmup_bind=True,
        fight_end_use_warm_start=True,
    )
    config.mode_cfg = ModeConfig(
        fight_blocks_restart=True,
        fight_in_fight_threshold=0.2,
        fight_in_fight_window=50,
        fight_decay_rate=0.005,
        super_fight_lr_boost=0.5,
        protection_proportion=0.005,
        find_mode_sub_start=0.3,
        find_mode_sub_end=0.2,
        find_fight_rate=5.0,
        patience_num=200,
        cooldown_steps=10,
        stop_condition=10,
        ema_beta=0.95,
        flee_steps=10,
        flee_noise_sigma=0.0,
        volatility_window=50,
        volatility_threshold_start=0.25,
        volatility_threshold_end=0.20,
        volatility_patience=10,
    )
    config.regularization_cfg = RegularizationConfig(weight_decay=0.001, clip_norm=0.0001)
    config.checkpoint_cfg = CheckpointConfig(
        base_checkpoint_dir="runs/model/3",
        max_keep_checkpoints=5,
        allow_hash_mismatch=False,
    )