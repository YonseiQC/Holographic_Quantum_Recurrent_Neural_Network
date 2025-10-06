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
        h_state = jnp.zeros(2**mdl_cfg.n_H, dtype=jnp.complex64).at[0].set(1.0)  # Reset hidden qubits

        def _sample_from_probs(params, h_state, x_row, label_idx, key):
            sample_key, _ = random.split(key)
            probs = self.q_model.circuit_for_probs(params, h_state, x_row, label_idx)
            probs_clipped = jnp.clip(probs, 0.0, 1.0)
            probs_normalized = probs_clipped / (jnp.sum(probs_clipped) + 1e-9)
            measured_int = random.categorical(sample_key, jnp.log(probs_normalized))  # Sample class id
            sample_bits = (jnp.floor_divide(measured_int, powers) % 2).astype(jnp.int32)  # Class id -> D bits
            return sample_bits, measured_int  # Return both bit vector and integer

        def body_fun(t, carry):
            h_state, key = carry
            key, subkey = random.split(key)
            x_t = x_sequence[t].astype(jnp.float32)  # Current timestep bits
            sample_bits, measured_int = _sample_from_probs(params, h_state, x_t, 0, subkey)  # Collapse data register
            full_state = self.q_model.circuit_for_state(params, h_state, x_t, 0)  # State after circuit
            psi_matrix = jnp.reshape(full_state, (2**mdl_cfg.n_H, 2**mdl_cfg.n_D))  # Split to [H, D]
            h_unnormalized = psi_matrix[:, measured_int]  # Post-measure hidden amplitude column
            h_norm = jnp.linalg.norm(h_unnormalized)
            h_next = jnp.where(h_norm > 1e-9, h_unnormalized / h_norm, h_state)
            return (h_next, subkey)  # Carry next hidden state and RNG

        h_state, key = lax.fori_loop(0, mdl_cfg.seq_len, body_fun, (h_state, key))
        last_input = x_sequence[-1].astype(jnp.float32)  # Use last context as control input
        final_sample_bits, _ = _sample_from_probs(params, h_state, last_input, 0, key)  # Predict next-step bits

        return final_sample_bits

    def visualize_samples(self, params, key, epoch, ckpt_manager: CheckpointManager, mode_state: UnifiedModeState,
                          save_to_disk=True, tag=""):
        if self.data_handler is None:
            print("Error: Data handler not set in Visualizer.")
            return

        print("Generating predictions (One-step vs. Autoregressive)...")
        mdl_cfg = self.config.model_cfg
        X_full = jnp.concatenate([self.data_handler.X_train, self.data_handler.X_test], axis=0)  # All sequences
        num_predictions = len(X_full)  # Number of one-step targets

        key, one_step_key, ar_key = random.split(key, 3)

        # One-step
        print(f"1/2: Predicting one-step rates...")
        one_step_keys = random.split(one_step_key, num_predictions)
        one_step_bits_list = []
        for i in tqdm(range(num_predictions), desc="One-step"):
            bits_i = self._predict_next(params, one_step_keys[i], X_full[i])
            one_step_bits_list.append(bits_i)
        all_pred_bits = jnp.stack(one_step_bits_list, axis=0)  # [N, n_D]

        all_pred_bits_reshaped = all_pred_bits.reshape(num_predictions, 1, -1)  # Add time axis for _denormalize
        one_step_predicted_rates = jax.vmap(self.data_handler._denormalize)(all_pred_bits_reshaped).flatten()  # [N]
        one_step_predicted_rates = [float(r) for r in one_step_predicted_rates]  # Convert to Python floats

        # AR
        print(f"2/2: Predicting autoregressive rates...")
        N_train = self.data_handler.train_size
        N_test = len(self.data_handler.X_test)

        test_ar_predicted_rates = []
        if N_test > 0:
            current_sequence = self.data_handler.X_train[-1]
            ar_keys = random.split(ar_key, N_test)
            for i in tqdm(range(N_test), desc="AR loop (test)"):
                pred_bits = self._predict_next(params, ar_keys[i], current_sequence)  # Predict next bits
                pred_rate = self.data_handler._denormalize(pred_bits.reshape(1, -1))[0]  # Bits -> rate
                test_ar_predicted_rates.append(float(pred_rate))
                current_sequence = jnp.roll(current_sequence, -1, axis=0).at[-1].set(pred_bits)

        autoregressive_predicted_rates = one_step_predicted_rates[:N_train] + test_ar_predicted_rates

        start_idx = mdl_cfg.seq_len
        start_value = self.data_handler.original_values[start_idx]  # Anchor price
        actual_values = self.data_handler.original_values[start_idx: start_idx + num_predictions + 1]

        one_step_pred_values = [start_value]
        for rate in one_step_predicted_rates:
            one_step_pred_values.append(one_step_pred_values[-1] * (1 + rate))

        autoregressive_pred_values = [start_value]
        for rate in autoregressive_predicted_rates:
            autoregressive_pred_values.append(autoregressive_pred_values[-1] * (1 + rate))

        if save_to_disk:
            days = range(num_predictions + 1)
            train_days = self.data_handler.train_size

            fig1, ax1 = plt.subplots(1, 1, figsize=(16, 8))
            fig1.suptitle(f'S&P 500 Prediction (One-step) - Epoch {epoch}', fontsize=16)
            ax1.plot(days, actual_values, color='gray', linewidth=2.5, label='Actual')
            ax1.plot(days, one_step_pred_values, color='firebrick', linewidth=2, linestyle='-', label='Prediction (One-step)')
            ax1.axvline(x=train_days, color='r', linestyle=':', linewidth=2, label='Train/Test Split')
            ax1.set_xlabel('Day'); ax1.set_ylabel('Open Value'); ax1.legend(); ax1.grid(True, alpha=0.5)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            img_path1 = ckpt_manager.plots_dir / f"SnP500_onestep_epoch_{epoch}{'_'+tag if tag else ''}.png"
            plt.savefig(img_path1, dpi=150); plt.close(fig1)
            print(f"One-step prediction plot saved to {img_path1}")

            fig2, ax2 = plt.subplots(1, 1, figsize=(16, 8))
            fig2.suptitle(f'S&P 500 Prediction (Autoregressive) - Epoch {epoch}', fontsize=16)
            ax2.plot(days, actual_values, color='gray', linewidth=2.5, label='Actual')
            ax2.plot(days, autoregressive_pred_values, color='peru', linewidth=2, linestyle='--', label='Prediction (Autoregressive)')
            ax2.axvline(x=train_days, color='r', linestyle=':', linewidth=2, label='Train/Test Split')
            ax2.set_xlabel('Day'); ax2.set_ylabel('Open Value'); ax2.legend(); ax2.grid(True, alpha=0.5)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            img_path2 = ckpt_manager.plots_dir / f"SnP500_autoregressive_epoch_{epoch}{'_'+tag if tag else ''}.png"
            plt.savefig(img_path2, dpi=150); plt.close(fig2)
            print(f"Autoregressive prediction plot saved to {img_path2}")

        results_df = pd.DataFrame({
            'Day': range(num_predictions + 1),
            'Actual_Value': actual_values,
            'Predicted_OneStep': one_step_pred_values,
            'Predicted_Autoregressive': autoregressive_pred_values
        })
        csv_path = ckpt_manager.csv_dir / f"SnP500_prediction_epoch_{epoch}{'_'+tag if tag else ''}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Prediction comparison CSV saved to {csv_path}")