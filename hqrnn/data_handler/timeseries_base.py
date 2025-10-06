import jax
import jax.numpy as jnp
from hqrnn.config.base import Config

# ------ 8-1. Model1 & 2 Data Handler

class TimeSeriesDataHandler:
    def __init__(self, config: Config):
        self.config = config
        self.n_D = config.model_cfg.n_D
        self.exp = config.dataset_cfg.normalize_exponent  # Exponent used in normalization
        self.M = 2 ** (self.n_D - 1)  # Half of the K=2^n_D classes
        self.S = jnp.sum(1 / 2 ** jnp.arange(self.exp, self.exp + self.n_D - 1))  # Max abs rate magnitude after mapping
        self.X = None
        self.Y = None

    def _clip_rate(self, x):
        return jnp.clip(x, -self.S + 1e-8, self.S - 1e-8)

    def _normalize(self, x):
        x_clipped = self._clip_rate(jnp.asarray(x))  # Ensure safe range before binning
        idx_bin = jnp.rint((x_clipped / self.S + 1) * (self.M - 0.5)).astype(jnp.int32)  # Map rate -> class index
        powers = 2 ** jnp.arange(self.n_D - 1, -1, -1)
        return (jnp.floor_divide(idx_bin, powers) % 2).astype(jnp.int32)  # Class index -> D-bit vector

    def _denormalize(self, bits):
        idx = jnp.sum(bits * (2 ** jnp.arange(self.n_D - 1, -1, -1)), axis=-1)  # Bits -> class index
        x = ((idx / (self.M - 0.5)) - 1) * self.S  # Class index -> rate
        return x.astype(jnp.float32)

    def _value_to_int(self, bits):
        powers = 2 ** jnp.arange(bits.shape[-1] - 1, -1, -1)
        return jnp.sum(bits * powers).astype(jnp.int32)  # Bits -> integer label