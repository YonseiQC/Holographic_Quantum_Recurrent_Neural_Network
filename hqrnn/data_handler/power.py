import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from hqrnn.config.base import Config
from .timeseries_base import TimeSeriesDataHandler

# ------ 8-3. Model2 Data Handler

class PowerDataHandler(TimeSeriesDataHandler):
    def __init__(self, config: Config):
        super().__init__(config)
        print("Loading Power Demand data...")
        self.df = pd.read_csv(self.config.dataset_cfg.csv_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.hourly_data = {}  # Cache per-hour (X, Y)
        self.active_hour = -1

    def _remove_outliers(self, values):
        if len(values) < 4:
            return values
        values = np.array(values, dtype=float)
        rates = np.diff(values) / values[:-1]  # Convert to rate-of-change series
        if len(rates) < 2:
            return values
        rate_jumps = np.abs(np.diff(rates))  # Magnitude of rate acceleration
        n_outliers = len(rate_jumps) // 10  # Heuristic: top 10% largest jumps
        if n_outliers == 0:
            return values
        outlier_indices = np.argsort(rate_jumps)[-n_outliers:]  # Indices of largest jumps
        print(f"Detected {n_outliers} potential outliers out of {len(values)} data points.")
        for rate_idx in sorted(outlier_indices, reverse=True):
            value_idx = rate_idx + 1
            if 0 < value_idx < len(values) - 1:
                values[value_idx] = (values[value_idx - 1] + values[value_idx + 1]) / 2  # Linear interpolate
        return values

    def _compute_rates(self, values):
        pd.options.mode.chained_assignment = None
        cleaned_values = self._remove_outliers(values)  # Outliers removed
        rates = (cleaned_values[1:] - cleaned_values[:-1]) / cleaned_values[:-1]  # Re-compute simple returns
        return np.nan_to_num(rates)

    def prepare_data_for_hour(self, hour: int):
        print(f"Preparing data for hour {hour}...")
        ds_cfg = self.config.dataset_cfg
        mdl_cfg = self.config.model_cfg

        target_date = pd.to_datetime(f"{ds_cfg.target_year}-{ds_cfg.target_month}-{ds_cfg.target_day}")
        start_date = pd.to_datetime(f"{ds_cfg.start_year}-{ds_cfg.start_month}-{ds_cfg.start_day}")

        df_weekday = self.df[self.df['Date'].dt.dayofweek == ds_cfg.target_weekday]  # Filter by weekday
        train_df = df_weekday[(df_weekday['Date'] >= start_date) & (df_weekday['Date'] < target_date)]  # Train range

        hour_col_str = str(hour)
        hour_col_padded = f"{hour:02d}"

        if hour_col_str in train_df.columns:
            hour_col = hour_col_str
        elif hour_col_padded in train_df.columns:
            hour_col = hour_col_padded
        else:
            print(f"Warning: Hour columns '{hour_col_str}' or '{hour_col_padded}' not found in data.")
            self.hourly_data[hour] = (None, None)
            return

        values = train_df[hour_col].values
        rates = self._compute_rates(values)

        X, Y = [], []
        for i in range(len(rates) - mdl_cfg.seq_len):
            x_seq_rates = rates[i: i + mdl_cfg.seq_len]
            y_rate = rates[i + mdl_cfg.seq_len]
            x_seq_bits = jax.vmap(self._normalize)(x_seq_rates)
            y_label = self._value_to_int(self._normalize(y_rate))
            X.append(x_seq_bits)
            Y.append(y_label)

        if not X:
            print(f"Warning: No data to train for hour {hour}.")
            self.hourly_data[hour] = (None, None)
        else:
            self.hourly_data[hour] = (jnp.stack(X), jnp.array(Y))
            print(f"Created {len(X)} sequences for hour {hour}.")

    def set_active_hour(self, hour: int):
        if hour not in self.hourly_data:
            self.prepare_data_for_hour(hour)
        self.active_hour = hour
        self.X, self.Y = self.hourly_data.get(hour, (None, None))

    def create_batch(self, key, batch_size):
        if self.X is None or len(self.X) == 0:
            return None, None, None
        num_samples = self.X.shape[0]
        indices = random.choice(key, num_samples, (min(batch_size, num_samples),), replace=False)
        X_batch = self.X[indices]
        Y_batch = self.Y[indices]
        Y_expanded = jnp.zeros((len(indices), self.config.model_cfg.seq_len), dtype=jnp.int32)
        Y_expanded = Y_expanded.at[:, -1].set(Y_batch)
        L_batch = jnp.zeros(len(indices), dtype=jnp.int32)
        return X_batch, Y_expanded, L_batch