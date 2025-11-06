import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import traceback

from hqrnn.config.base import Config, ExperimentConfig
from hqrnn.quantum.model import QuantumModel
from hqrnn.visualizer.snp import SnpVisualizer
from hqrnn.data_handler.snp import SnpDataHandler
from hqrnn.trainers.autodiff import Trainer_Autodiff
from hqrnn.trainers.spsa import Trainer_SPSA
from hqrnn.utils.checkpoint import CheckpointManager
from hqrnn.utils.seed import set_seed, get_key
from hqrnn.FFF_mode.types import UnifiedModeState


def evaluate_checkpoint(params, config, data_handler, ckpt_manager, epoch_tag):
    q_model = QuantumModel(config=config)
    visualizer = SnpVisualizer(config=config, q_model=q_model)
    visualizer.set_data_handler(data_handler)
    
    key = get_key()
    
    mode_state = UnifiedModeState(config=config)
    
    dir_acc, mape = visualizer.visualize_samples(
        params, key, epoch_tag, ckpt_manager, mode_state, 
        save_to_disk=True, tag=f"eval_{epoch_tag}"
    )
    
    if dir_acc is not None and mape is not None:
        return float(mape), float(dir_acc)
    
    return None, None


def run_single_training(config: Config, seed: int, exp_name: str, param_str: str):
    print(f"\n--- Running: {exp_name} | Params: {param_str} | Seed: {seed} ---")
    print(f"  - Config hash: {config.get_hash()}")
    
    q_model = QuantumModel(config=config)
    data_handler = SnpDataHandler(config=config)
    visualizer = SnpVisualizer(config=config, q_model=q_model)
    visualizer.set_data_handler(data_handler)
    
    ckpt_manager = CheckpointManager(config=config)
    
    cfg = config.exp_cfg
    if cfg.loss_function == 'mmd' and cfg.collapse_type == 'hard':
        trainer = Trainer_SPSA(config, q_model, data_handler, visualizer, ckpt_manager)
    else:
        trainer = Trainer_Autodiff(config, q_model, data_handler, visualizer, ckpt_manager)
    
    _, final_state, _ = trainer.run()
    
    results = {}
    
    print("  - Loading best checkpoint for evaluation...")
    loaded_best = ckpt_manager.load_checkpoint("best")
    if loaded_best:
        best_params, _, _, _, _, _ = loaded_best
        mape_best, dir_best = evaluate_checkpoint(
            best_params, config, data_handler, ckpt_manager, "best"
        )
        results['mae_best'] = mape_best
        results['dir_acc_best'] = dir_best
    else:
        print("  - Warning: Could not load best checkpoint.")
        results['mae_best'] = None
        results['dir_acc_best'] = None
    
    is_fff_mode = config.scheduler_toggle_cfg.use_complex_scheduler
    last_tag = "final" if is_fff_mode else "last"
    
    print(f"  - Loading {last_tag} checkpoint for evaluation...")
    loaded_last = ckpt_manager.load_checkpoint(last_tag)
    if loaded_last:
        last_params, _, _, _, _, _ = loaded_last
        mape_last, dir_last = evaluate_checkpoint(
            last_params, config, data_handler, ckpt_manager, last_tag
        )
        results['mae_last'] = mape_last
        results['dir_acc_last'] = dir_last
    else:
        print(f"  - Warning: Could not load {last_tag} checkpoint.")
        results['mae_last'] = None
        results['dir_acc_last'] = None
    
    results['best_loss'] = final_state.best_loss
    
    return results


def plot_boxplot(df, x_var, y_var, title, file_path, ylabel=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    unique_vals = df[x_var].unique()
    if all(isinstance(v, (int, float, np.integer, np.floating)) for v in unique_vals):
        order = sorted(unique_vals)
    else:
        order = sorted(unique_vals, key=str)
    
    sns.stripplot(x=x_var, y=y_var, data=df, ax=ax, jitter=True, 
                  alpha=0.6, size=7, color='gray', order=order)
    
    sns.boxplot(x=x_var, y=y_var, data=df, ax=ax, order=order,
                boxprops=dict(alpha=0.7), width=0.6)
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(x_var.replace('_', ' ').title(), fontsize=14)
    ax.set_ylabel(ylabel if ylabel else y_var.replace('_', ' ').title(), fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Plot saved to {file_path}")


def run_experiment(exp_num, exp_name, params_to_test, seeds, base_config_fn):
    results = []
    base_results_dir = Path(f"runs/model/1/mmdexp3/exp{exp_num}_{exp_name}")
    summary_dir = base_results_dir / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    total_runs = len(params_to_test) * len(seeds)
    pbar = tqdm(total=total_runs, desc=f"Exp{exp_num}: {exp_name}")
    
    for params in params_to_test:
        for seed in seeds:
            try:
                set_seed(seed)
                
                config = base_config_fn(params)
                
                param_str = "_".join([f"{k}_{v}" for k, v in params.items()])
                run_name = f"{param_str}_seed_{seed}"
                unique_checkpoint_dir = base_results_dir / run_name
                config.checkpoint_cfg.base_checkpoint_dir = str(unique_checkpoint_dir)
                
                metrics = run_single_training(config, seed, exp_name, param_str)
                
                result_row = {'seed': seed}
                result_row.update(params)
                result_row.update(metrics)
                results.append(result_row)
                
            except Exception as e:
                print(f"\n!!! ERROR in exp{exp_num}, {params}, seed {seed} !!!")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {e}")
                print(traceback.format_exc())
                print("Skipping to next run...\n")
            
            pbar.update(1)
    
    pbar.close()
    
    if not results:
        print(f"\nWARNING: No successful runs for Exp{exp_num}")
        return None, summary_dir
    
    results_df = pd.DataFrame(results)
    csv_path = summary_dir / "results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults for Exp{exp_num} saved to {csv_path}")
    
    param_name = list(params_to_test[0].keys())[0]
    
    plot_boxplot(results_df, param_name, 'mae_best',
                f'MAE (Best Checkpoint) - {exp_name}',
                summary_dir / "plot_mae_best.png",
                ylabel='MAE (%)')
    
    plot_boxplot(results_df, param_name, 'mae_last',
                f'MAE (Last Checkpoint) - {exp_name}',
                summary_dir / "plot_mae_last.png",
                ylabel='MAE (%)')
    
    plot_boxplot(results_df, param_name, 'dir_acc_best',
                f'Direction Accuracy (Best Checkpoint) - {exp_name}',
                summary_dir / "plot_dir_acc_best.png",
                ylabel='Direction Accuracy (%)')
    
    plot_boxplot(results_df, param_name, 'dir_acc_last',
                f'Direction Accuracy (Last Checkpoint) - {exp_name}',
                summary_dir / "plot_dir_acc_last.png",
                ylabel='Direction Accuracy (%)')
    
    return results_df, summary_dir


def main():
    print("=" * 80)
    print("Starting Model1 Experiment Suite (12 Experiments) - REVERSE ORDER")
    print("=" * 80)
    
    seeds = [1, 2, 3, 4, 5, 6]
    
    print("\n" + "="*80)
    print("EXPERIMENT 12: Scheduler (Cosine vs FFF_mode, NLL)")
    print("="*80)
    try:
        def base_config_exp12(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function='nll',
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = params['use_complex_scheduler']
            return config
        
        run_experiment(
            12, "Scheduler_NLL",
            [{'use_complex_scheduler': sched} for sched in [False, True]],
            seeds,
            base_config_exp12
        )
    except Exception as e:
        print(f"Experiment 12 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 11: Scheduler (Cosine vs FFF_mode, MMD)")
    print("="*80)
    try:
        def base_config_exp11(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function='mmd',
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = params['use_complex_scheduler']
            return config
        
        run_experiment(
            11, "Scheduler_MMD",
            [{'use_complex_scheduler': sched} for sched in [False, True]],
            seeds,
            base_config_exp11
        )
    except Exception as e:
        print(f"Experiment 11 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 10: Batch Size (MMD + FFF_mode)")
    print("="*80)
    try:
        def base_config_exp10(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function='mmd',
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = True
            config.training_cfg.batch_size = params['batch_size']
            return config
        
        run_experiment(
            10, "Batch_MMD_FFF",
            [{'batch_size': bs} for bs in [1, 4, 16, 64, 256]],
            seeds,
            base_config_exp10
        )
    except Exception as e:
        print(f"Experiment 10 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 9: Batch Size (MMD + Cosine)")
    print("="*80)
    try:
        def base_config_exp9(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function='mmd',
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = False
            config.training_cfg.batch_size = params['batch_size']
            return config
        
        run_experiment(
            9, "Batch_MMD_Cosine",
            [{'batch_size': bs} for bs in [1, 4, 16, 64, 256]],
            seeds,
            base_config_exp9
        )
    except Exception as e:
        print(f"Experiment 9 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 8: Loss Function (MMD vs NLL, FFF_mode)")
    print("="*80)
    try:
        def base_config_exp8(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function=params['loss_function'],
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = True
            return config
        
        run_experiment(
            8, "Loss_FFF",
            [{'loss_function': loss} for loss in ['nll', 'mmd']],
            seeds,
            base_config_exp8
        )
    except Exception as e:
        print(f"Experiment 8 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 7: Loss Function (MMD vs NLL, Cosine)")
    print("="*80)
    try:
        def base_config_exp7(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function=params['loss_function'],
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = False
            return config
        
        run_experiment(
            7, "Loss_Cosine",
            [{'loss_function': loss} for loss in ['nll', 'mmd']],
            seeds,
            base_config_exp7
        )
    except Exception as e:
        print(f"Experiment 7 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 6: Depth (MMD + FFF_mode)")
    print("="*80)
    try:
        def base_config_exp6(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function='mmd',
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = True
            config.model_cfg.depth = params['depth']
            return config
        
        run_experiment(
            6, "Depth_MMD_FFF",
            [{'depth': d} for d in [1, 2, 4, 8, 16]],
            seeds,
            base_config_exp6
        )
    except Exception as e:
        print(f"Experiment 6 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 5: Depth (MMD + Cosine)")
    print("="*80)
    try:
        def base_config_exp5(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function='mmd',
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = False
            config.model_cfg.depth = params['depth']
            return config
        
        run_experiment(
            5, "Depth_MMD_Cosine",
            [{'depth': d} for d in [1, 2, 4, 8, 16]],
            seeds,
            base_config_exp5
        )
    except Exception as e:
        print(f"Experiment 5 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 4: Qubit Count (MMD + FFF_mode)")
    print("="*80)
    try:
        def base_config_exp4(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function='mmd',
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = True
            config.model_cfg.n_D = params['n_D']
            config.model_cfg.n_H = params['n_D']
            return config
        
        run_experiment(
            4, "Qubit_MMD_FFF",
            [{'n_D': n} for n in [3, 4, 5, 6, 7]],
            seeds,
            base_config_exp4
        )
    except Exception as e:
        print(f"Experiment 4 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 3: Qubit Count (MMD + Cosine)")
    print("="*80)
    try:
        def base_config_exp3(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function='mmd',
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = False
            config.model_cfg.n_D = params['n_D']
            config.model_cfg.n_H = params['n_D']
            return config
        
        run_experiment(
            3, "Qubit_MMD_Cosine",
            [{'n_D': n} for n in [3, 4, 5, 6, 7]],
            seeds,
            base_config_exp3
        )
    except Exception as e:
        print(f"Experiment 3 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 2: Learning Rate (MMD + FFF_mode)")
    print("="*80)
    try:
        def base_config_exp2(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function='mmd',
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = True
            config.training_cfg.learning_rate_init = params['learning_rate_init']
            return config
        
        run_experiment(
            2, "LR_MMD_FFF",
            [{'learning_rate_init': lr} for lr in [0.01, 0.05, 0.1, 0.5, 1.0]],
            seeds,
            base_config_exp2
        )
    except Exception as e:
        print(f"Experiment 2 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("EXPERIMENT 1: Learning Rate (MMD + Cosine)")
    print("="*80)
    try:
        def base_config_exp1(params):
            config = Config(model=1)
            config.exp_cfg = ExperimentConfig(
                loss_function='mmd',
                collapse_type='soft',
                learning_mode='teacher_forcing'
            )
            config.scheduler_toggle_cfg.use_complex_scheduler = False
            config.training_cfg.learning_rate_init = params['learning_rate_init']
            return config
        
        run_experiment(
            1, "LR_MMD_Cosine",
            [{'learning_rate_init': lr} for lr in [0.01, 0.05, 0.1, 0.5, 1.0]],
            seeds,
            base_config_exp1
        )
    except Exception as e:
        print(f"Experiment 1 failed: {e}\n{traceback.format_exc()}")
    
    print("\n" + "="*80)
    print("All 12 Experiments Completed! (Reverse Order)")
    print("="*80)


if __name__ == '__main__':
    main()
