from functools import partial
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jax import random, lax
import optax
from tqdm import tqdm

from hqrnn.config.base import Config
from hqrnn.quantum.model import QuantumModel
from hqrnn.scheduler.scheduler import Scheduler
from hqrnn.FFF_mode.controller import ModeController
from hqrnn.scheduler.loss_history import LossHistory
from hqrnn.FFF_mode.types import Mode
from hqrnn.utils.seed import get_key
from hqrnn.utils.plotting import save_loss_plot

# --- 10. Trainer_Autodiff

class Trainer_Autodiff:
    def __init__(self, config: Config, q_model: QuantumModel, data_handler, visualizer, ckpt_manager):
        self.config = config
        self.q_model = q_model
        self.data_handler = data_handler
        self.visualizer = visualizer
        self.ckpt_manager = ckpt_manager
        self.exp_cfg = config.exp_cfg
        self.loss_cfg = config.loss_cfg
        self.target_mmd_r0_r1 = None
        if self.config.model == 1:
            self.dist_map = data_handler.dist_map
            self.context_bins = data_handler.context_bins

    def _bits_to_int(self, bits):
        n_D = self.config.model_cfg.n_D
        powers = 2 ** jnp.arange(n_D - 1, -1, -1)
        return jnp.sum(bits * powers).astype(jnp.int32)


    # ------ One-step transition

    def _step(self, carry, inputs, collapse_type):
        h_state, params, key = carry
        x_row, y_row, label_idx = inputs
        key, step_key, sample_key = random.split(key, 3)

        if collapse_type == 'hard':
            probs = self.q_model.circuit_for_probs(params, h_state, x_row, label_idx)
            full_state = self.q_model.circuit_for_state(params, h_state, x_row, label_idx)
            probs_clipped = jnp.clip(probs, 0.0, 1.0)
            probs_normalized = probs_clipped / (jnp.sum(probs_clipped) + 1e-9)  # Normalize for sampling
            measured_int = random.categorical(sample_key, jnp.log(probs_normalized))  # Sample y
            n_D = self.config.model_cfg.n_D
            powers = 2 ** jnp.arange(n_D - 1, -1, -1)
            sample_bits = (jnp.floor_divide(measured_int, powers) % 2).astype(jnp.int32)  # Class -> bits
            psi_matrix = jnp.reshape(full_state, (2**self.config.model_cfg.n_H, 2**self.config.model_cfg.n_D))  # [H, D]
            h_unnormalized = psi_matrix[:, measured_int]  # Take measured column
            h_norm = jnp.linalg.norm(h_unnormalized)
            h_next = jnp.where(h_norm > 1e-9, h_unnormalized / h_norm, h_state)
            h_next = jax.lax.stop_gradient(h_next)  # Hidden update in hard collapse
            return (h_next, params, key), (probs, sample_bits)

        elif collapse_type == 'soft':
            probs = self.q_model.circuit_for_probs(params, h_state, x_row, label_idx)
            full_state = self.q_model.circuit_for_state(params, h_state, x_row, label_idx)
            psi_matrix = jnp.reshape(full_state, (2**self.config.model_cfg.n_H, 2**self.config.model_cfg.n_D))  # [H, D]
            norms = jnp.linalg.norm(psi_matrix, axis=0, keepdims=True) + 1e-12  # Column norms
            h_next = jnp.sum((psi_matrix / norms) * jnp.sqrt(probs)[None, :], axis=1)  # Prob-weighted update
            h_next /= (jnp.linalg.norm(h_next) + 1e-12)  # Renormalize
            probs_clipped = jnp.clip(probs, 0.0, 1.0)
            probs_normalized = probs_clipped / (jnp.sum(probs_clipped) + 1e-9)  # Normalize for sampling
            measured_int = random.categorical(sample_key, jnp.log(probs_normalized))  # Sample y
            n_D = self.config.model_cfg.n_D
            powers = 2 ** jnp.arange(n_D - 1, -1, -1)
            sample_bits = (jnp.floor_divide(measured_int, powers) % 2).astype(jnp.int32)  # Class -> bits
            return (h_next, params, key), (probs, sample_bits)
        else:
            raise ValueError(f"Unknown collapse type: {collapse_type}")


    # ------ Sequence forward pass

    def _forward_sequence(self, params, X, Y, L, key, learning_mode, collapse_type):
        mdl_cfg = self.config.model_cfg
        h_initial = jnp.zeros(2**mdl_cfg.n_H, dtype=jnp.complex64).at[0].set(1.0)
        K = 2**mdl_cfg.n_D

        if learning_mode == 'teacher_forcing':
            inputs = (X, Y, jnp.full(mdl_cfg.seq_len, L))
            step_fn = partial(self._step, collapse_type=collapse_type)
            _, (probs_seq, samples_seq) = lax.scan(step_fn, (h_initial, params, key), inputs, length=mdl_cfg.seq_len)
            return probs_seq, samples_seq

        elif learning_mode == 'autoregressive':
            def ar_body_fun(t, carry):
                h_state, prev_output_bits, key, all_probs, all_samples = carry
                key, step_key = random.split(key)
                current_input = lax.cond(
                    t == 0,
                    lambda: X[0].astype(jnp.float32),  # First step uses given context
                    lambda: prev_output_bits  # Then feed previous prediction
                )
                inputs_t = (current_input, Y[t], L)
                (h_next, _, _), (probs, sample_bits) = self._step((h_state, params, step_key), inputs_t, collapse_type)
                all_probs = all_probs.at[t].set(probs)
                all_samples = all_samples.at[t].set(sample_bits)
                next_prev = sample_bits.astype(jnp.float32)
                return (h_next, next_prev, key, all_probs, all_samples)

            initial_carry = (
                h_initial,
                jnp.zeros((mdl_cfg.n_D,), dtype=jnp.float32),
                key,
                jnp.zeros((mdl_cfg.seq_len, K), dtype=jnp.float32),
                jnp.zeros((mdl_cfg.seq_len, mdl_cfg.n_D), dtype=jnp.int32),
            )
            _, _, _, probs_seq, samples_seq = lax.fori_loop(0, mdl_cfg.seq_len, ar_body_fun, initial_carry)
            return probs_seq, samples_seq

        else:
            raise ValueError(f"Unknown learning mode: {learning_mode}")


    # ------ NLL loss

    def _nll_loss_fn(self, params, X, Y, L, key, learning_mode, collapse_type):
        vmapped_forward = jax.vmap(self._forward_sequence, in_axes=(None, 0, 0, 0, 0, None, None))  # Batch over B
        probs_batch_seq, _ = vmapped_forward(params, X, Y, L, random.split(key, X.shape[0]), learning_mode, collapse_type)

        batch_indices = jnp.arange(Y.shape[0])
        if self.config.model in (1, 2):
            probs_last_step = probs_batch_seq[:, -1, :]  # Use final step only
            y_last_step = Y[:, -1]
            target_probs = probs_last_step[batch_indices, y_last_step]
            loss_per_sample = -jnp.log(jnp.clip(target_probs, 1e-12, 1.0))
        else:
            time_indices = jnp.arange(Y.shape[1])
            target_probs = probs_batch_seq[batch_indices[:, None], time_indices, Y]  # Per-timestep targets
            log_loss = -jnp.log(jnp.clip(target_probs, 1e-12, 1.0))
            loss_per_sample = jnp.mean(log_loss, axis=1)  # Mean over time

        return jnp.mean(loss_per_sample), loss_per_sample


    # ------ MMD loss

    def _mmd_loss_fn(self, params, X, Y, L, key, learning_mode, collapse_type):
        mdl_cfg = self.config.model_cfg
        loss_cfg = self.loss_cfg
        B, T, D = X.shape[0], mdl_cfg.seq_len, mdl_cfg.n_D

        vmapped_forward = jax.vmap(self._forward_sequence, in_axes=(None, 0, 0, 0, 0, None, None))
        probs_batch_seq, _ = vmapped_forward(params, X, Y, L, random.split(key, B), learning_mode, collapse_type)  # [B, T, 2^D]

        bit_table = jnp.array(
            [[(j >> (D - 1 - i)) & 1 for i in range(D)] for j in range(2**D)],
            dtype=jnp.float32
        )

        def _rbf(A, B, gamma):  # A: [N, d], B: [M, d]
            diff = A[:, None, :] - B[None, :, :]  # [N, M, d]
            dist2 = jnp.sum(diff * diff, axis=-1)  # [N, M]
            return jnp.exp(-gamma * dist2)


        # ----- MMD for Model 1
        if self.config.model == 1:
            sigma = self.loss_cfg.mmd_sigma
            gamma = 1.0 / (2 * sigma**2)

            v_bits_to_int = jax.vmap(self._bits_to_int)
            start_vals = v_bits_to_int(X[:, 0, :])  # Start index per sample
            end_vals = v_bits_to_int(X[:, -1, :])  # End index per sample
            context_values_batch = end_vals - start_vals

            context_ids_batch = jnp.digitize(context_values_batch, self.context_bins) - 1  # Bin to case id
            context_ids_batch = jnp.clip(context_ids_batch, 0, len(self.context_bins) - 2)

            Q = self.dist_map[context_ids_batch]  # Empirical prob
            P = probs_batch_seq[:, -1, :]  # Model prob

            def rbf_kernel_y(A, B, gamma):
                diff = A[:, None, :] - B[None, :, :]
                dist2 = jnp.sum(diff * diff, axis=-1)
                return jnp.exp(-gamma * dist2)

            Kpp = rbf_kernel_y(P, P, gamma).mean()
            Kqq = rbf_kernel_y(Q, Q, gamma).mean()
            Kpq = rbf_kernel_y(P, Q, gamma).mean()

            total_loss = jnp.maximum(0.0, Kpp - 2.0 * Kpq + Kqq)
            loss_per_sample = jnp.full((B,), total_loss)
            return total_loss, loss_per_sample


        # ----- MMD for Model 2
        if self.config.model == 2:
            sigma = jnp.asarray(loss_cfg.mmd_sigma, dtype=jnp.float32)
            gamma = 1.0 / (2.0 * sigma**2)

            probs_last = probs_batch_seq[:, -1, :]  # Use final step
            Eb_last = probs_last @ bit_table  # Predicted bit expectation
            y_bits_last = bit_table[Y[:, -1]]  # True bit vector

            Kpp = _rbf(Eb_last, Eb_last, gamma).mean()
            Kqq = _rbf(y_bits_last, y_bits_last, gamma).mean()
            Kpq = _rbf(Eb_last, y_bits_last, gamma).mean()

            total_loss = Kpp - 2.0 * Kpq + Kqq
            loss_per_sample = jnp.full((B,), total_loss, dtype=jnp.float32)
            return total_loss, loss_per_sample


        # ----- MMD for Model 3
        sigma = jnp.asarray(loss_cfg.mmd_sigma, dtype=jnp.float32)
        gamma = 1.0 / (2.0 * sigma**2)
        lambda_dist = jnp.asarray(loss_cfg.mmd_lambda, dtype=jnp.float32)

        Eb = (probs_batch_seq @ bit_table).reshape(B, T * D)  # Pred bit expectations over time
        y_bits = bit_table[Y].reshape(B, T * D)  # True bits over time

        def _mmd_masked(P, Q, mP, mQ):
            nP = jnp.sum(mP); nQ = jnp.sum(mQ)
            KP = _rbf(P, P, gamma); KQ = _rbf(Q, Q, gamma); KPQ = _rbf(P, Q, gamma)
            Mpp = mP[:, None] * mP[None, :]
            Mqq = mQ[:, None] * mQ[None, :]
            Mpq = mP[:, None] * mQ[None, :]
            term_pp = jnp.sum(KP * Mpp) / jnp.maximum(nP**2, 1e-9)
            term_qq = jnp.sum(KQ * Mqq) / jnp.maximum(nQ**2, 1e-9)
            term_pq = jnp.sum(KPQ * Mpq) / jnp.maximum(nP * nQ, 1e-9)
            ok = (nP > 0) & (nQ > 0)
            return jnp.where(ok, term_pp - 2.0 * term_pq + term_qq, 0.0)

        m0 = (L == 0).astype(jnp.float32)  # Mask for class 0
        m1 = (L == 1).astype(jnp.float32)  # Mask for class 1

        mmd_P0_R0 = _mmd_masked(Eb, y_bits, m0, m0)
        mmd_P1_R1 = _mmd_masked(Eb, y_bits, m1, m1)
        mmd_P0_R1 = _mmd_masked(Eb, y_bits, m0, m1)
        mmd_P1_R0 = _mmd_masked(Eb, y_bits, m1, m0)

        if self.target_mmd_r0_r1 is None:
            self.target_mmd_r0_r1 = float(getattr(self.data_handler, "calculate_target_mmd")(sigma=float(loss_cfg.mmd_sigma)))

        target = jnp.asarray(self.target_mmd_r0_r1, dtype=jnp.float32)

        L0 = mmd_P0_R0 + lambda_dist * (mmd_P0_R1 - target) ** 2  # Class-0 objective
        L1 = mmd_P1_R1 + lambda_dist * (mmd_P1_R0 - target) ** 2  # Class-1 objective

        k = jnp.asarray(loss_cfg.mmd_k, dtype=jnp.float32)  # Smooth-max temperature
        total_loss = jax.nn.logsumexp(k * jnp.stack([L0, L1])) / k  # Smooth aggregation

        loss_per_sample = jnp.where((L == 0), L0, L1)  # Per-sample selection
        return total_loss, loss_per_sample

    
    # ------ Training loop

    def run(self):
        cfg = self.config
        key = get_key()
        n_classes = 2 if cfg.model == 3 else 1  # Label embeddings for model 3
        params = self.q_model.initialize_params(key, n_classes=n_classes)

        opt_core = optax.chain(
            optax.clip_by_global_norm(cfg.regularization_cfg.clip_norm),
            optax.scale_by_adam(),
            optax.add_decayed_weights(cfg.regularization_cfg.weight_decay),
            optax.scale(-1.0)
        )
        opt_state = opt_core.init(params)

        if cfg.model == 3 and self.exp_cfg.loss_function == 'mmd':
            self.target_mmd_r0_r1 = float(self.data_handler.calculate_target_mmd(sigma=float(self.loss_cfg.mmd_sigma)))  # Precompute target

        @partial(jax.jit, static_argnames=['loss_function_name', 'learning_mode', 'collapse_type'])
        def train_step_jit(params, opt_state, X, Y, L, lr, key, loss_function_name, learning_mode, collapse_type):
            if loss_function_name == 'nll':
                loss_fn = self._nll_loss_fn
            elif loss_function_name == 'mmd':
                loss_fn = self._mmd_loss_fn
            else:
                raise ValueError(f"Unknown loss function: {loss_function_name}")

            (loss_val, selected_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, X, Y, L, key, learning_mode, collapse_type
            )  # Compute loss and grads
            updates, opt_state = opt_core.update(grads, opt_state, params)  # Optimizer step
            updates_scaled = jtu.tree_map(lambda u: lr * u, updates)  # LR scaling
            params = optax.apply_updates(params, updates_scaled)  # Apply updates
            return params, opt_state, loss_val, selected_loss

        scheduler = Scheduler(cfg)  # LR scheduler (complex or simple)
        mode_controller = ModeController(cfg)  # Find/Fight/Flee controller
        loss_history = LossHistory()  # For logging and plots
        start_epoch = 0

        pbar = tqdm(range(start_epoch, cfg.training_cfg.max_epochs), desc="Training")
        for epoch in pbar:
            key, batch_key, step_key = random.split(key, 3)
            Xb, Yb, Lb = self.data_handler.create_batch(batch_key, cfg.training_cfg.batch_size)
            if Xb is None:
                continue

            base_lr, _ = scheduler.get_lr(epoch, mode_controller.state)

            if cfg.scheduler_toggle_cfg.use_complex_scheduler:
                current_lr = max(base_lr * mode_controller.get_lr_multiplier(), cfg.training_cfg.learning_rate_min)
            else:
                current_lr = base_lr

            params, opt_state, loss_val_original, selected_loss = train_step_jit(
                params, opt_state, Xb, Yb, Lb, current_lr, step_key,
                self.exp_cfg.loss_function, self.exp_cfg.learning_mode, self.exp_cfg.collapse_type
            )

            loss_dict = {}
            if cfg.model == 3:
                n0 = int((Lb == 0).sum()); n1 = int((Lb == 1).sum())
                loss_dict['first_digit_loss'] = float(selected_loss[Lb == 0].mean()) if n0 > 0 else 0.0
                loss_dict['second_digit_loss'] = float(selected_loss[Lb == 1].mean()) if n1 > 0 else 0.0

            loss_val = loss_val_original
            loss_dict['total'] = float(loss_val)

            if cfg.scheduler_toggle_cfg.use_complex_scheduler:
                was_fight_mode = mode_controller.state.mode == Mode.FIGHT
                res = mode_controller.update(float(loss_val), base_lr, epoch) 
                is_fight_mode = mode_controller.state.mode == Mode.FIGHT

                if res.should_save_best:
                    self.ckpt_manager.save_checkpoint("best", epoch, params, opt_state, mode_controller.state, key, loss_history)

                if was_fight_mode and not is_fight_mode:
                    print(f"\n--- Fight mode ended at epoch {epoch}. Generating analysis plots... ---")
                    key, plot_key = random.split(key)
                    self.ckpt_manager.save_loss_plot(loss_history, epoch, mode_controller.state)  # Loss timeline
                    self.visualizer.visualize_samples(
                        params, plot_key, epoch, self.ckpt_manager,
                        mode_controller.state, save_to_disk=True, tag=f"fight_end_epoch_{epoch}"
                    )

                if mode_controller.state.mode == Mode.FLEE:
                    key, noise_key = random.split(key)
                    params = self._add_noise(params, noise_key, cfg.mode_cfg.flee_noise_sigma)

            else:
                if loss_val < mode_controller.state.best_loss:
                    mode_controller.state.best_loss = float(loss_val)
                    self.ckpt_manager.save_checkpoint("best", epoch, params, opt_state, mode_controller.state, key, loss_history)

                if (epoch + 1) % cfg.training_cfg.plot_every_n_epochs == 0 and epoch > 0:
                    print(f"\n--- Periodic plot at epoch {epoch + 1}. Generating analysis plots... ---")
                    key, plot_key = random.split(key)
                    self.ckpt_manager.save_loss_plot(loss_history, epoch + 1, mode_controller.state)  # Periodic plot
                    self.visualizer.visualize_samples(
                        params, plot_key, epoch + 1, self.ckpt_manager,
                        mode_controller.state, save_to_disk=True, tag=f"periodic_epoch_{epoch + 1}"
                    )

            loss_history.add(epoch, loss_dict, mode_controller.state.mode, current_lr)

            postfix = {"Loss": f"{float(loss_val):.4f}", "LR": f"{current_lr:.6f}", "Best": f"{mode_controller.state.best_loss:.4f}"}
            if cfg.scheduler_toggle_cfg.use_complex_scheduler:
                postfix["Mode"] = mode_controller.state.mode.value.capitalize()[:6]
            pbar.set_postfix(postfix)

        print("\nTraining completed.")
        return params, mode_controller.state, loss_history


    # ------ Noise adder

    def _add_noise(self, params, key, sigma):
        leaves, treedef = jtu.tree_flatten(params)
        noise_keys = random.split(key, len(leaves))
        noisy_leaves = [p + sigma * random.normal(k, p.shape) for p, k in zip(leaves, noise_keys)]
        return jtu.tree_unflatten(treedef, noisy_leaves)
