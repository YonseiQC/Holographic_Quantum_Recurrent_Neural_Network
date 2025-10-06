from functools import partial
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from hqrnn.config.base import Config
from hqrnn.quantum.model import QuantumModel
from hqrnn.utils.checkpoint import CheckpointManager
from hqrnn.FFF_mode.types import UnifiedModeState
from hqrnn.utils.seed import get_key

# ------ 9-2. Model2 Visualizer

class PowerVisualizer:
    def __init__(self, config: Config, q_model: QuantumModel):
        self.config = config
        self.q_model = q_model

    def visualize_samples(self, params, key, epoch, ckpt_manager: CheckpointManager, mode_state: UnifiedModeState,
                          save_to_disk=True, tag=""):
        pass

    @partial(jax.jit, static_argnames=['self'])
    def _predict_next_demand_probs(self, params, x_sequence):
        mdl_cfg = self.config.model_cfg
        h_state = jnp.zeros(2**mdl_cfg.n_H, dtype=jnp.complex64).at[0].set(1.0)

        for t in range(mdl_cfg.seq_len):
            x_t = x_sequence[t].astype(jnp.float32)
            probs = self.q_model.circuit_for_probs(params, h_state, x_t, 0)
            full_state = self.q_model.circuit_for_state(params, h_state, x_t, 0)
            psi_matrix = jnp.reshape(full_state, (2**mdl_cfg.n_H, 2**mdl_cfg.n_D))
            norms = jnp.linalg.norm(psi_matrix, axis=0, keepdims=True) + 1e-12
            h_state = jnp.sum((psi_matrix / norms) * jnp.sqrt(probs)[None, :], axis=1)
            h_state /= (jnp.linalg.norm(h_state) + 1e-12)
        last_input = x_sequence[-1].astype(jnp.float32)
        return self.q_model.circuit_for_probs(params, h_state, last_input, 0)

    def visualize_24hour_prediction(self, all_trained_params, data_handler, ckpt_manager: CheckpointManager):
        ds_cfg = self.config.dataset_cfg
        mdl_cfg = self.config.model_cfg
        key = get_key()

        target_date = pd.to_datetime(f"{ds_cfg.target_year}-{ds_cfg.target_month}-{ds_cfg.target_day}")
        df_target = data_handler.df[data_handler.df['Date'].dt.date == target_date.date()]
        if df_target.empty:
            print(f"No data found for target date {target_date.date()}"); return

        actual_demands, predicted_demands, prediction_errors, hours_with_models = [], [], [], []

        for hour in tqdm(range(1, 25), desc="Predicting 24-hour demand"):  # Seperate training for 1, ..., 24 hour
            hour_col = str(hour)
            if hour not in all_trained_params or hour_col not in df_target.columns:
                continue

            hours_with_models.append(hour)
            params = all_trained_params[hour]
            actual_demand = df_target[hour_col].iloc[0]
            actual_demands.append(actual_demand)

            data_handler.set_active_hour(hour)
            if data_handler.X is not None and len(data_handler.X) > 0:
                last_sequence = data_handler.X[-1]
                pred_probs = self._predict_next_demand_probs(params, last_sequence)

                key, subkey = random.split(key)
                pred_label = random.choice(subkey, jnp.arange(2**mdl_cfg.n_D), p=pred_probs)

                pred_bits = jnp.array([(pred_label >> (mdl_cfg.n_D - 1 - i)) & 1 for i in range(mdl_cfg.n_D)])
                pred_rate = data_handler._denormalize(pred_bits.reshape(1, -1))[0]

                prev_week_date = target_date - pd.Timedelta(weeks=1)
                prev_week_data = data_handler.df[data_handler.df['Date'].dt.date == prev_week_date.date()]
                if not prev_week_data.empty and hour_col in prev_week_data.columns:
                    base_demand = prev_week_data[hour_col].iloc[0]
                    predicted_demand = base_demand * (1 + float(pred_rate))
                else:
                    predicted_demand = actual_demand
            else:
                predicted_demand = actual_demand
            predicted_demands.append(predicted_demand)
            prediction_errors.append(abs(predicted_demand - actual_demand))

        valid_indices = ~np.isnan(actual_demands) & ~np.isnan(predicted_demands)
        actual_valid = np.array(actual_demands)[valid_indices]
        predicted_valid = np.array(predicted_demands)[valid_indices]
        errors_valid = np.abs(actual_valid - predicted_valid)

        mae = np.mean(errors_valid) if len(errors_valid) > 0 else 0  # Mean absolute error
        rmse = np.sqrt(np.mean(errors_valid**2)) if len(errors_valid) > 0 else 0  # Root mean square error
        mape = np.mean(errors_valid / (actual_valid + 1e-9) * 100) if len(actual_valid) > 0 else 0  # Mean absolute percent error

        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        fig.suptitle(f'24-Hour Power Demand Prediction - {target_date.date()}', fontsize=16)
        ax.plot(hours_with_models, actual_demands, color='gray', linewidth=2.5, marker='o', label='Actual Demand')
        ax.plot(hours_with_models, predicted_demands, color='firebrick', linewidth=2, marker='s', linestyle='--', label='Predicted Demand')
        ax.set_xlabel('Hour of Day'); ax.set_ylabel('Power Demand'); ax.set_title('Actual vs. Predicted Demand')
        ax.legend(); ax.grid(True, alpha=0.5); ax.set_xticks(range(0, 25, 2))

        stats_text = (f"MAE: {mae:,.2f}\n" f"RMSE: {rmse:,.2f}\n" f"MAPE: {mape:.2f}%")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.8))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = f"power_prediction_summary_{target_date.date()}.png"
        img_path = ckpt_manager.plots_dir / filename
        plt.savefig(img_path, dpi=150); plt.close()
        print(f"Power demand prediction plot saved to {img_path}")

        results_df = pd.DataFrame({
            'Hour': hours_with_models,
            'Actual_Demand': actual_demands,
            'Predicted_Demand': predicted_demands,
            'Absolute_Error': prediction_errors
        })
        csv_path = ckpt_manager.csv_dir / f"demand_prediction_{target_date.date()}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Full prediction results saved to {csv_path}")
