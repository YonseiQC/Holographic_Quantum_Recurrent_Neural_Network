from jax import random

from hqrnn.config.base import Config, ExperimentConfig
from hqrnn.quantum.model import QuantumModel
from hqrnn.data_handler.snp import SnpDataHandler
from hqrnn.data_handler.power import PowerDataHandler
from hqrnn.data_handler.digits import DigitDataHandler
from hqrnn.visualizer.snp import SnpVisualizer
from hqrnn.visualizer.power import PowerVisualizer
from hqrnn.visualizer.digits import DigitVisualizer
from hqrnn.utils.checkpoint import CheckpointManager
from hqrnn.trainers.autodiff import Trainer_Autodiff
from hqrnn.trainers.spsa import Trainer_SPSA
from hqrnn.utils.seed import get_key
from hqrnn.utils.plotting import save_loss_plot

# --- 12. Main

if __name__ == '__main__':
    #######################################################################
    # ⭐⭐⭐ Modify This
    #######################################################################
    EXPERIMENT_PARAMS = {
        "model": 2,  # 1: S&P500, 2: Power Demand, 3: Digit Generation
        "loss_function": "nll",  # "nll" or "mmd"
        "collapse_type": "soft", # "soft" or "hard"
        "learning_mode": "teacher_forcing" # "teacher_forcing" or "autoregressive"
    }
    #######################################################################

    print("1. Initializing Configurations...")
    config = Config(model=EXPERIMENT_PARAMS["model"])
    config.exp_cfg = ExperimentConfig(
        loss_function=EXPERIMENT_PARAMS["loss_function"],
        collapse_type=EXPERIMENT_PARAMS["collapse_type"],
        learning_mode=EXPERIMENT_PARAMS["learning_mode"]
    )
    print(f"    - Model: {config.model}")
    print(f"    - Experiment Config: {config.exp_cfg}")


    print("\n2. Preparing Model and Data Handler...")
    q_model = QuantumModel(config=config)

    if config.model == 1:
        data_handler = SnpDataHandler(config=config)
        visualizer = SnpVisualizer(config=config, q_model=q_model)
        visualizer.set_data_handler(data_handler)
    elif config.model == 2:
        data_handler = PowerDataHandler(config=config)
        visualizer = PowerVisualizer(config=config, q_model=q_model)
    elif config.model == 3:
        data_handler = DigitDataHandler(config=config)
        visualizer = DigitVisualizer(config=config, q_model=q_model)
    else:
        raise ValueError(f"Invalid model number: {config.model}")

    print("\n3. Starting Training...")
    key = get_key()

    if config.model == 2:
        all_trained_params = {}
        for hour in range(1, 25):
            print(f"\n{'='*20} Training for Hour {hour} {'='*20}")
            ckpt_manager = CheckpointManager(config, hour=hour)
            data_handler.set_active_hour(hour)
            if data_handler.X is None:
                print(f"No data for hour {hour}, skipping.")
                continue

            cfg = config.exp_cfg
            if cfg.loss_function == 'mmd' and cfg.collapse_type == 'hard':
                print(f"    - Using Trainer_SPSA for Hour {hour}")
                trainer = Trainer_SPSA(config, q_model, data_handler, visualizer, ckpt_manager)
            else:
                if cfg.loss_function == 'nll' and cfg.collapse_type == 'hard':
                    print(f"    - (Loss=Soft, State=Hard).")
                print(f"    - Using Trainer_Autodiff for Hour {hour}")
                trainer = Trainer_Autodiff(config, q_model, data_handler, visualizer, ckpt_manager)

            all_trained_params[hour], _, _ = trainer.run()

        print("\n" + "="*50)
        print("4. Generating 24-hour prediction visualization...")
        final_ckpt_manager = CheckpointManager(config)
        visualizer.visualize_24hour_prediction(all_trained_params, data_handler, final_ckpt_manager)

    else:
        ckpt_manager = CheckpointManager(config=config)

        cfg = config.exp_cfg
        if cfg.loss_function == 'mmd' and cfg.collapse_type == 'hard':
            print(f"   - Using Trainer_SPSA (gradient-free) for MMD + Hard combination.")
            trainer = Trainer_SPSA(config, q_model, data_handler, visualizer, ckpt_manager)
        else:
            if cfg.loss_function == 'nll' and cfg.collapse_type == 'hard':
                print(f"   - Warning: Using Trainer_Autodiff for NLL+Hard (Loss=Soft, State=Hard).")
            print(f"   - Using Trainer_Autodiff (gradient-based).")
            trainer = Trainer_Autodiff(config, q_model, data_handler, visualizer, ckpt_manager)

        final_params, final_state, loss_history = trainer.run()

        print("\n" + "="*50)
        print("4. Generating final visualizations...")
        key, final_key, best_key = random.split(key, 3)

        print("    - Generating plot for FINAL checkpoint...")
        visualizer.visualize_samples(
            final_params, final_key, "FINAL", ckpt_manager,
            final_state, save_to_disk=True, tag="final"
        )

        print("    - Loading and generating plot for BEST checkpoint...")
        try:
            loaded = ckpt_manager.load_checkpoint("best")
            if loaded is not None:
                best_params, _, best_state, best_epoch, _, _ = loaded
                visualizer.visualize_samples(
                    best_params, best_key, f"BEST_epoch_{best_epoch}", ckpt_manager,
                    best_state, save_to_disk=True, tag="best"
                )
                print(f"    - Successfully generated plot for BEST checkpoint (from epoch {best_epoch}).")
            else:
                print("    - Could not load BEST checkpoint for plotting.")
        except Exception as e:
            print(f"    - Error: {e}")

        print("    - Saving final loss plot...")
        final_epoch = loss_history.epochs[-1] if loss_history.epochs else 0
        save_loss_plot(ckpt_manager.plots_dir, config, loss_history, final_epoch, final_state)
    print("\nAll processes finished successfully.")