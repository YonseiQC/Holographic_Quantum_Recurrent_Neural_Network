import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from hqrnn.config.base import Config
from .timeseries_base import TimeSeriesDataHandler

# ------ 8-2. Model1 Data Handler

class SnpDataHandler(TimeSeriesDataHandler):
    def __init__(self, config: Config):
        super().__init__(config)
        self.X_train, self.Y_train = None, None
        self.X_test, self.Y_test = None, None
        self.original_values = None
        self.train_size = 0
        self.dist_map = None  # P(y | context-bin) in training set
        self.context_bins = None  # Context bin boundaries
        self._load_and_prep_data()
        self._create_dist_map()

    def _load_and_prep_data(self):
        print("Loading and preparing S&P 500 data...")
        df = pd.read_csv(self.config.dataset_cfg.csv_path)

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

        ds_cfg = self.config.dataset_cfg
        start_date_str = f"{ds_cfg.start_year}-{ds_cfg.start_month:02d}-{ds_cfg.start_day:02d}"
        start_date = pd.to_datetime(start_date_str)

        df = df[df['Date'] >= start_date].reset_index(drop=True)
        print(f"Data filtered to start from: {start_date.date()}")

        if ds_cfg.total_days is not None:
            df = df.head(ds_cfg.total_days + self.config.model_cfg.seq_len + 1)
            print(f"Data limited to total {len(df)} records for sequence generation.")

        print(f"Data date range being used: {df['Date'].min()} to {df['Date'].max()}")

        self.original_values = df['Open'].values  # Raw price series
        rates = (self.original_values[1:] - self.original_values[:-1]) / self.original_values[:-1]  # Daily returns

        seq_len = self.config.model_cfg.seq_len
        X, Y = [], []
        for i in range(len(rates) - seq_len):
            x_seq_rates = rates[i: i + seq_len]  # Context window of returns
            y_rate = rates[i + seq_len]  # Next-step return as label target

            x_seq_bits = jax.vmap(self._normalize)(x_seq_rates)  # Vectorized rate -> bits
            y_label = self._value_to_int(self._normalize(y_rate))  # Target class id

            X.append(x_seq_bits)
            Y.append(y_label)

        X_full, Y_full = jnp.stack(X), jnp.array(Y)

        self.train_size = int(len(X_full) * 0.8)  # 80/20 split
        self.X_train, self.X_test = jnp.split(X_full, [self.train_size])
        self.Y_train, self.Y_test = jnp.split(Y_full, [self.train_size])

        print(f"Created {len(X_full)} total sequences.")
        print(f"Data split into {len(self.X_train)} training and {len(self.X_test)} testing samples.")

    def _create_dist_map(self):
        print("\nCreating distribution map for Model 1...")
        mdl_cfg = self.config.model_cfg
        ds_cfg = self.config.dataset_cfg
        n_D = mdl_cfg.n_D
        exp = ds_cfg.map_exponent
        K = 2 ** n_D
        seq_len = mdl_cfg.seq_len
        num_cases = 2 ** (n_D - exp)  # Number of context bins
        group_size = 2 ** exp  # Classes per bin

        all_bits = jnp.array([[(i >> (n_D - 1 - j)) & 1 for j in range(n_D)] for i in range(K)])  # Lookup table bits
        rate_lookup_table = self._denormalize(all_bits)  # Rate per class

        initial_boundaries = [rate_lookup_table[i * group_size] for i in range(num_cases)]
        initial_boundaries.append(rate_lookup_table[-1])
        initial_boundaries = jnp.array(initial_boundaries)

        min_rate_repre = initial_boundaries[0]
        max_rate_repre = initial_boundaries[-1]

        train_indices = jnp.arange(self.train_size)
        start_values = self.original_values[train_indices]
        end_values = self.original_values[train_indices + seq_len]
        train_contexts = (end_values / start_values) - 1.0 

        clipped_contexts = jnp.clip(train_contexts, min_rate_repre, max_rate_repre)

        initial_assignments = jnp.digitize(clipped_contexts, initial_boundaries) - 1
        initial_assignments = jnp.clip(initial_assignments, 0, num_cases - 1)

        case_counts = np.bincount(initial_assignments, minlength=num_cases)

        final_boundaries = list(initial_boundaries)

        for i in range(num_cases - 1, -1, -1):
            if case_counts[i] == 0:
                print(f"Case {i} is empty. Merging its range.")
                if i == num_cases - 1:
                    final_boundaries.pop(-2)  # Merge with left neighbor
                elif i == 0:
                    final_boundaries.pop(1)  # Merge with right neighbor
                else:
                    lower_b = final_boundaries[i]
                    upper_b = final_boundaries[i + 1]
                    mid_point = (lower_b + upper_b) / 2.0  # Re-center boundary
                    final_boundaries.pop(i + 1)
                    final_boundaries[i] = mid_point

        self.context_bins = jnp.array(final_boundaries)  # Finalized bin edges

        final_assignments = jnp.digitize(clipped_contexts, self.context_bins) - 1
        final_assignments = jnp.clip(final_assignments, 0, len(self.context_bins) - 2)

        num_final_cases = len(self.context_bins) - 1
        dist_map = {i: jnp.zeros(K, dtype=jnp.float32) for i in range(num_final_cases)}  # P(y | bin=i) counts
        final_case_counts = {i: 0 for i in range(num_final_cases)}

        for cid, label in zip(final_assignments, self.Y_train):
            cid = int(cid)
            dist_map[cid] = dist_map[cid].at[label].add(1)
            final_case_counts[cid] += 1

        for i in range(num_final_cases):
            if final_case_counts[i] > 0:
                dist_map[i] /= final_case_counts[i]
            else:
                dist_map[i] = jnp.full(K, 1.0 / K)

        self.dist_map = jnp.stack([dist_map[i] for i in range(num_final_cases)])
        print(f"Successfully created map with {num_final_cases} final cases.")

    def create_batch(self, key, batch_size):
        num_samples = self.X_train.shape[0]
        indices = random.choice(key, num_samples, (min(batch_size, num_samples),), replace=False)
        X_batch = self.X_train[indices]
        Y_batch = self.Y_train[indices]

        Y_expanded = jnp.zeros((len(indices), self.config.model_cfg.seq_len), dtype=jnp.int32)
        Y_expanded = Y_expanded.at[:, -1].set(Y_batch)  # Place label at final step
        L_batch = jnp.zeros(len(indices), dtype=jnp.int32)

        return X_batch, Y_expanded, L_batch