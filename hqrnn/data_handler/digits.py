from typing import Optional
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import random
from hqrnn.config.base import Config, ModelConfig

# --- 8-4. Model3 Data Handler

class DigitDataHandler:
    def __init__(self, config: Config):
        self.config = config
        self.target_mmd_r0_r1: Optional[float] = None
        self.X0, self.Y0, self.X1, self.Y1 = self._load_and_prep_data()

    @staticmethod
    def _bits_to_int(bits_1d: np.ndarray) -> int:
        n = bits_1d.shape[0]
        pow2 = 2 ** np.arange(n - 1, -1, -1, dtype=np.int64)
        return int(np.sum(bits_1d.astype(np.int64) * pow2))

    def _prep_case_7x7(self, images: np.ndarray, labels: np.ndarray,
                         d0: int, d1: int, mdl_cfg: ModelConfig):
        n_samples = len(images)
        X = np.zeros((n_samples, mdl_cfg.seq_len, mdl_cfg.n_D), dtype=np.int32)
        Y = np.zeros((n_samples, mdl_cfg.seq_len), dtype=np.int32)

        for i in range(n_samples):
            for t in range(1, mdl_cfg.seq_len):
                X[i, t] = images[i, t - 1]
            for t in range(mdl_cfg.seq_len):
                Y[i, t] = self._bits_to_int(images[i, t])

        mask0 = (labels == d0); mask1 = (labels == d1)
        return jnp.array(X[mask0]), jnp.array(Y[mask0]), jnp.array(X[mask1]), jnp.array(Y[mask1])

    def _load_and_prep_data(self):
        ds_cfg = self.config.dataset_cfg
        mdl_cfg = self.config.model_cfg
        try:
            df = pd.read_csv(ds_cfg.csv_path)
        except Exception as e:
            print(f"Error reading CSV from {ds_cfg.csv_path}: {e}")
            raise

        if (mdl_cfg.n_D, mdl_cfg.n_H, mdl_cfg.seq_len) != (7, 7, 7):
            raise ValueError(
                f"Unsupported (n_D, n_H, seq_len)={mdl_cfg.n_D, mdl_cfg.n_H, mdl_cfg.seq_len}. "
                "Only (7,7,7) is supported for model3. "
            )

        images, labels = [], []
        digits = [ds_cfg.first_digit, ds_cfg.second_digit]
        for _, row in df.iterrows():
            label = int(row["label"])
            if label in digits:
                image = (row.iloc[1:50].values > 0.5).astype(np.int32).reshape(7, 7)
                images.append(image)
                labels.append(label)
        images, labels = np.array(images), np.array(labels)

        return self._prep_case_7x7(images, labels, ds_cfg.first_digit, ds_cfg.second_digit, mdl_cfg)

    def create_batch(self, key, batch_size):
        key0, key1 = random.split(key)
        b0 = batch_size // 2
        b1 = batch_size - b0

        idx0 = random.choice(key0, len(self.X0), (min(b0, len(self.X0)),), replace=False)
        idx1 = random.choice(key1, len(self.X1), (min(b1, len(self.X1)),), replace=False)

        Xb = jnp.concatenate([self.X0[idx0], self.X1[idx1]], 0)
        Yb = jnp.concatenate([self.Y0[idx0], self.Y1[idx1]], 0)
        Lb = jnp.concatenate([jnp.zeros(len(idx0), jnp.int32), jnp.ones(len(idx1), jnp.int32)], 0)  # Class labels {0,1}
        Cb = jnp.zeros(len(Lb), dtype=jnp.int32)

        return Xb.astype(jnp.float32), Yb, Lb, Cb

    def calculate_target_mmd(self, n_samples: int = 1024, sigma: float = 1.5) -> float:
        if self.target_mmd_r0_r1 is not None:
            return self.target_mmd_r0_r1

        print(f"Calculating target MMD between real 0s and 1s using up to {n_samples} samples...")

        X0, Y0, X1, Y1 = self.X0, self.Y0, self.X1, self.Y1

        n0 = int(min(n_samples // 2, Y0.shape[0]))
        n1 = int(min(n_samples // 2, Y1.shape[0]))
        if n0 == 0 or n1 == 0:
            self.target_mmd_r0_r1 = 0.0
            print("Not enough samples to compute target MMD.")
            return self.target_mmd_r0_r1

        Y0 = Y0[:n0]
        Y1 = Y1[:n1]

        D = self.config.model_cfg.n_D
        bit_table = jnp.array([[(j >> (D - 1 - i)) & 1 for i in range(D)] for j in range(2 ** D)], dtype=jnp.float32)
        R0_bits = bit_table[Y0]
        R1_bits = bit_table[Y1]

        R0 = R0_bits.reshape(n0, -1)
        R1 = R1_bits.reshape(n1, -1)

        gamma = 1.0 / (2.0 * sigma ** 2)

        def _rbf_kernel(A, B):
            diff = A[:, None, :] - B[None, :, :]
            dist_sq = jnp.sum(diff * diff, axis=-1)
            return jnp.exp(-gamma * dist_sq)

        def _mmd_from_sets(P, Q):
            if P.shape[0] == 0 or Q.shape[0] == 0:
                return 0.0
            Kpp = jnp.mean(_rbf_kernel(P, P))
            Kqq = jnp.mean(_rbf_kernel(Q, Q))
            Kpq = jnp.mean(_rbf_kernel(P, Q))
            return Kpp - 2.0 * Kpq + Kqq

        target_mmd = _mmd_from_sets(R0, R1)
        self.target_mmd_r0_r1 = float(target_mmd)
        print(f"Calculated Target MMD(R0 bits, R1 bits): {self.target_mmd_r0_r1:.6f}")
        return self.target_mmd_r0_r1
