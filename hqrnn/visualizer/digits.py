from functools import partial
import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from hqrnn.config.base import Config
from hqrnn.quantum.model import QuantumModel
from hqrnn.utils.checkpoint import CheckpointManager
from hqrnn.FFF_mode.types import UnifiedModeState

# ------ 9-3. Model3 Visualizer

class DigitVisualizer:
    def __init__(self, config: Config, q_model: QuantumModel):
        self.config = config
        self.q_model = q_model

    @partial(jax.jit, static_argnames=['self', 'label_idx'])
    def _generate_digit(self, params, key, label_idx):
        mdl = self.config.model_cfg
        if (mdl.n_D, mdl.n_H, mdl.seq_len) != (7, 7, 7):
            raise ValueError(f"DigitVisualizer supports only (7,7,7), got {(mdl.n_D, mdl.n_H, mdl.seq_len)}")

        D, T = mdl.n_D, mdl.seq_len
        powers = 2 ** jnp.arange(D - 1, -1, -1)
        h = jnp.zeros(2**mdl.n_H, dtype=jnp.complex64).at[0].set(1.0)
        cur = jnp.zeros(D, dtype=jnp.float32)

        def _sample_from_probs(params, h_state, x_row, label_idx, key):
            sample_key, _ = random.split(key)
            probs = self.q_model.circuit_for_probs(params, h_state, x_row, label_idx)
            probs = jnp.clip(probs, 0.0, 1.0)
            probs /= (jnp.sum(probs) + 1e-9)
            measured_int = random.categorical(sample_key, jnp.log(probs))
            bits = (jnp.floor_divide(measured_int, powers) % 2).astype(jnp.int32)
            return bits, measured_int

        def body_fun(t, state):
            h_state, x_t, key, rows = state
            key, subkey = random.split(key)
            bits, idx = _sample_from_probs(params, h_state, x_t, label_idx, subkey)  # Collapse data register
            full_state = self.q_model.circuit_for_state(params, h_state, x_t, label_idx)  # Post-circuit state
            psi = jnp.reshape(full_state, (2**mdl.n_H, 2**D))  # [H, D]
            h_unnorm = psi[:, idx]; h_norm = jnp.linalg.norm(h_unnorm)  # Column for measured idx
            h_next = jnp.where(h_norm > 1e-9, h_unnorm / h_norm, h_state)
            return (h_next, bits.astype(jnp.float32), subkey, rows.at[t].set(bits))  # Accumulate rows

        init = (h, cur, key, jnp.zeros((T, D), dtype=jnp.int32))  # Accumulator init
        _, _, _, rows = lax.fori_loop(0, T, body_fun, init)  # Generate T rows of bits
        return rows  # Shape [T, D]

    def visualize_samples(self, params, key, epoch, ckpt_manager: CheckpointManager, mode_state: UnifiedModeState,
                          save_to_disk=False, n_samples_per_digit=3, tag=""):
        ds = self.config.dataset_cfg
        mdl = self.config.model_cfg
        if (mdl.n_D, mdl.n_H, mdl.seq_len) != (7, 7, 7):
            raise ValueError(f"DigitVisualizer supports only (7,7,7), got {(mdl.n_D, mdl.n_H, mdl.seq_len)}")

        fig, axes = plt.subplots(2, n_samples_per_digit, figsize=(3 * n_samples_per_digit, 6))  # 2 rows for two digits
        if n_samples_per_digit == 1:
            axes = axes.reshape(2, 1)

        for row, d in enumerate([ds.first_digit, ds.second_digit]):
            label_idx = 0 if d == ds.first_digit else 1  # Map digit to label embedding index
            for col in range(n_samples_per_digit):
                key, sub = random.split(key)
                rows = self._generate_digit(params, sub, label_idx)  # [7, 7] bits
                img = np.array(rows, dtype=np.int32)

                ax = axes[row, col]
                ax.imshow(img, cmap=ListedColormap(['midnightblue', 'gold']), vmin=0, vmax=1)
                ax.set_title(f'Generated Digit {d}')
                ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f'Digit Generation - Epoch {epoch} | Loss: {mode_state.last_loss:.4f}', fontsize=14, y=0.95)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        if save_to_disk:
            path = ckpt_manager.plots_dir / f"digit_samples_epoch_{epoch}{'_' + tag if tag else ''}.png"
            plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
            print(f"Digit samples saved to {path}")
        else:
            plt.show()
