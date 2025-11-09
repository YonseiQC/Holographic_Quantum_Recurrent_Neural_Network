from functools import partial
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random, lax
import matplotlib.pyplot as plt
from tqdm import tqdm

from hqrnn.config.base import Config
from hqrnn.quantum.model import QuantumModel
from hqrnn.utils.checkpoint import CheckpointManager
from hqrnn.FFF_mode.types import UnifiedModeState

# ------ 9-1. Model1 Visualizer

class SnpVisualizer:
    def __init__(self, config: Config, q_model: QuantumModel):
        self.config = config
        self.q_model = q_model
        self.data_handler = None

    def set_data_handler(self, data_handler):
        self.data_handler = data_handler

    @partial(jax.jit, static_argnames=['self'])
    def _predict_next(self, params, key, x_sequence):
        mdl_cfg = self.config.model_cfg
        n_D = mdl_cfg.n_D
        powers = 2 ** jnp.arange(n_D - 1, -1, -1)
        h_state = jnp.zeros(2**mdl_cfg.n_H, dtype=jnp.complex64).at[0].set(1.0)

        def _sample_from_probs(params, h_state, x_row, label_idx, key):
            sample_key, _ = random.split(key)
            probs = self.q_model.circuit_for_probs(params, h_state, x_row, label_idx)
            probs_clipped = jnp.clip(probs, 0.0, 1.0)
            probs_normalized = probs_clipped / (jnp.sum(probs_clipped) + 1e-9)
            measured_int = random.categorical(sample_key, jnp.log(probs_normalized))
            sample_bits = (jnp.floor_divide(measured_int, powers) % 2).astype(jnp.int32)
            return sample_bits, measured_int

        def body_fun(t, carry):
            h_state, key = carry
            key, subkey = random.split(key)
            x_t = x_sequence[t].astype(jnp.float32)
            sample_bits, measured_int = _sample_from_probs(params, h_state, x_t, 0, subkey)
            full_state = self.q_model.circuit_for_state(params, h_state, x_t, 0)
            psi_matrix = jnp.reshape(full_state, (2**mdl_cfg.n_H, 2**mdl_cfg.n_D))
            h_unnormalized = psi_matrix[:, measured_int]
            h_norm = jnp.linalg.norm(h_unnormalized)
            h_next = jnp.where(h_norm > 1e-9, h_unnormalized / h_norm, h_state)
            return (h_next, subkey)

        h_state, key = lax.fori_loop(0, mdl_cfg.seq_len, body_fun, (h_state, key))
        last_input = x_sequence[-1].astype(jnp.float32)
        final_sample_bits, _ = _sample_from_probs(params, h_state, last_input, 0, key)

        return final_sample_bits

    def visualize_samples(self, params, key, epoch, ckpt_manager: CheckpointManager, mode_state: UnifiedModeState,
                          save_to_disk=True, tag=""):
        if self.data_handler is None:
            print("Error: Data handler not set in Visualizer.")
            return

        print("Generating predictions...")
        mdl_cfg = self.config.model_cfg
        X_full = jnp.concatenate([self.data_handler.X_train, self.data_handler.X_test], axis=0)
        num_predictions = len(X_full)
        key, pred_key = random.split(key, 2)

        print(f"Predicting rates...")
        pred_keys = random.split(pred_key, num_predictions)
        pred_bits_list = []
        for i in tqdm(range(num_predictions), desc="Prediction"):
            bits_i = self._predict_next(params, pred_keys[i], X_full[i])
            pred_bits_list.append(bits_i)
        all_pred_bits = jnp.stack(pred_bits_list, axis=0)

        all_pred_bits_reshaped = all_pred_bits.reshape(num_predictions, 1, -1)
        pred_predicted_rates = jax.vmap(self.data_handler._denormalize)(all_pred_bits_reshaped).flatten()
        pred_predicted_rates = [float(r) for r in pred_predicted_rates]

        start_idx = mdl_cfg.seq_len
        start_value = self.data_handler.original_values[start_idx]
        actual_values = self.data_handler.original_values[start_idx: start_idx + num_predictions + 1]

        pred_pred_values = [start_value]
        for rate in pred_predicted_rates:
            pred_pred_values.append(pred_pred_values[-1] * (1 + rate))

        actual_values_full = list(self.data_handler.original_values[start_idx - mdl_cfg.seq_len: start_idx]) + list(actual_values)
        pred_values_full = list(self.data_handler.original_values[start_idx - mdl_cfg.seq_len: start_idx]) + pred_pred_values

        actual_seq_directions = []
        pred_seq_directions = []
        for i in range(1, num_predictions + 1):
            actual_start = actual_values_full[i]
            actual_end = actual_values_full[i + mdl_cfg.seq_len]
            actual_change = actual_end - actual_start
            actual_seq_directions.append(1 if actual_change > 0 else 0)

            pred_start = pred_values_full[i]
            pred_end = pred_values_full[i + mdl_cfg.seq_len]
            pred_change = pred_end - pred_start
            pred_seq_directions.append(1 if pred_change > 0 else 0)

        direction_accuracy = sum([1 for a, p in zip(actual_seq_directions, pred_seq_directions) if a == p]) / len(actual_seq_directions) * 100

        actual_daily_rates = [(actual_values[i+1] - actual_values[i]) / actual_values[i] for i in range(num_predictions)]
        rate_mae = sum([abs(p - a) for a, p in zip(actual_daily_rates, pred_predicted_rates)]) / num_predictions * 100
        value_mae = sum([abs(p - a) for a, p in zip(actual_values, pred_pred_values)]) / (num_predictions + 1)

        if save_to_disk:
            days = range(num_predictions + 1)
            train_days = self.data_handler.train_size

            fig1, ax1 = plt.subplots(1, 1, figsize=(16, 8))
            fig1.suptitle(f'S&P 500 Prediction - Epoch {epoch}', fontsize=16)
            ax1.plot(days, actual_values, color='gray', linewidth=2.5, label='Actual')
            ax1.plot(days, pred_pred_values, color='firebrick', linewidth=2, linestyle='-', label='Prediction')
            ax1.axvline(x=train_days, color='r', linestyle=':', linewidth=2, label='Train/Test Split')
            ax1.set_xlabel('Day'); ax1.set_ylabel('Open Value'); ax1.legend(); ax1.grid(True, alpha=0.5)

            textstr = f'Direction Accuracy: {direction_accuracy:.1f}%\nRate_MAE: {rate_mae:.2f}%\nValue_MAE: {value_mae:.1f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='left', bbox=props)

            plt.tight_layout(rect=[0, 0, 1, 0.96])

            img_path = ckpt_manager.plots_dir / f"SnP500_prediction_epoch_{epoch}{'_'+tag if tag else ''}.png"
            plt.savefig(img_path, dpi=150); plt.close(fig1)
            print(f"Prediction plot saved to {img_path}")
            print(f"Direction Accuracy: {direction_accuracy:.1f}%, Rate_MAE: {rate_mae:.2f}%, Value_MAE: {value_mae:.1f}")

            results_df = pd.DataFrame({
                'Day': range(num_predictions + 1),
                'Actual_Value': actual_values,
                'Predicted_Value': pred_pred_values,
            })
            csv_path = ckpt_manager.csv_dir / f"SnP500_prediction_epoch_{epoch}{'_'+tag if tag else ''}.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"Prediction comparison CSV saved to {csv_path}")
        return direction_accuracy, rate_mae, value_mae
