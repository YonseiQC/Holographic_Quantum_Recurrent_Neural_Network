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

# --- 11. Trainer_SPSA

class Trainer_SPSA:
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

    def _bits_to_int(self, bits: jnp.ndarray) -> jnp.int32:
        n_D = self.config.model_cfg.n_D
        pow2 = 2 ** jnp.arange(n_D - 1, -1, -1)
        return jnp.sum(bits * pow2, axis=-1).astype(jnp.int32)


    # ------ Forward pass with hard samples

    def _forward_sequence_hard_samples(self, params, X_seq, Y_seq, L, key, learning_mode):
        mdl = self.config.model_cfg
        h0 = jnp.zeros(2 ** mdl.n_H, dtype=jnp.complex64).at[0].set(1.0)

        def step_with_input(h, x_row, label_idx, key):
            probs = self.q_model.circuit_for_probs(params, h, x_row, label_idx, True)
            probs_clipped = jnp.clip(probs, 0.0, 1.0)
            probs_normalized = probs_clipped / (jnp.sum(probs_clipped) + 1e-9)

            d_idx = random.categorical(key, jnp.log(probs_normalized)).astype(jnp.int32)
            d_bits = (jnp.floor_divide(d_idx, 2 ** jnp.arange(mdl.n_D - 1, -1, -1)) % 2).astype(jnp.int32)

            full_state = self.q_model.circuit_for_state(params, h, x_row, label_idx, True)
            psi_matrix = jnp.reshape(full_state, (2 ** mdl.n_H, 2 ** mdl.n_D))
            h_unnorm = psi_matrix[:, d_idx]
            h_norm = jnp.linalg.norm(h_unnorm)
            h_next = jnp.where(h_norm > 1e-9, h_unnorm / h_norm, h)

            onehot = jax.nn.one_hot(d_idx, num_classes=2 ** mdl.n_D)
            return h_next, onehot, d_bits

        def scan_body_teacher(carry, t):
            h, key = carry
            key, subkey = random.split(key)
            x_t = X_seq[t].astype(jnp.float32)
            h_next, onehot, _ = step_with_input(h, x_t, L, subkey)
            return (h_next, key), onehot

        def scan_body_ar(carry, t):
            h, prev_x, key = carry
            key, subkey = random.split(key)
            x_t = lax.cond(t == 0, lambda: X_seq[0].astype(jnp.float32), lambda: prev_x)
            h_next, onehot, d_bits = step_with_input(h, x_t, L, subkey)
            return (h_next, d_bits.astype(jnp.float32), key), onehot

        if learning_mode == 'teacher_forcing':
            init_carry = (h0, key)
            _, onehots = lax.scan(scan_body_teacher, init_carry, jnp.arange(mdl.seq_len))

        elif learning_mode == 'autoregressive':
            init_carry = (h0, jnp.zeros_like(X_seq[0], dtype=jnp.float32), key)
            _, onehots = lax.scan(scan_body_ar, init_carry, jnp.arange(mdl.seq_len))

        else:
            raise ValueError(f"Unknown learning_mode: {learning_mode}")

        return onehots


    # ------ MMD objective

    def mmd_value_only(self, params, X, Y, L, key, learning_mode):
        mdl_cfg, loss_cfg = self.config.model_cfg, self.loss_cfg
        B, T, D = X.shape[0], mdl_cfg.seq_len, mdl_cfg.n_D
        keys = random.split(key, B)

        model_onehots_seq = jax.vmap(self._forward_sequence_hard_samples, in_axes=(None, 0, 0, 0, 0, None))(
            params, X, Y, L, keys, learning_mode
        )

        bit_table = jnp.array([[(j >> (D - 1 - i)) & 1 for i in range(D)] for j in range(2 ** D)], dtype=jnp.float32)

        def _rbf(A, B, gamma):
            diff = A[:, None, :] - B[None, :, :]
            dist2 = jnp.sum(diff * diff, axis=-1)
            return jnp.exp(-gamma * dist2)


        if self.config.model == 1:
            sigma = loss_cfg.mmd_sigma
            gamma = 1.0 / (2 * sigma**2)

            v_bits_to_int = jax.vmap(self._bits_to_int)
            start_vals = v_bits_to_int(X[:, 0, :])
            end_vals = v_bits_to_int(X[:, -1, :])
            context_values = end_vals - start_vals
            context_ids = jnp.digitize(context_values, self.context_bins) - 1
            context_ids = jnp.clip(context_ids, 0, len(self.context_bins) - 2)

            P = model_onehots_seq[:, -1, :]
            Q = self.dist_map[context_ids]

            Kpp = _rbf(P, P, gamma).mean()
            Kqq = _rbf(Q, Q, gamma).mean()
            Kpq = _rbf(P, Q, gamma).mean()
            return jnp.maximum(0.0, Kpp - 2.0 * Kpq + Kqq)
            

        if self.config.model == 2:
            sigma = jnp.asarray(loss_cfg.mmd_sigma, dtype=jnp.float32)
            gamma = 1.0 / (2.0 * sigma**2)

            P_bits_last = model_onehots_seq[:, -1, :] @ bit_table
            Q_bits_last = bit_table[Y[:, -1]]

            Kpp = _rbf(P_bits_last, P_bits_last, gamma).mean()
            Kqq = _rbf(Q_bits_last, Q_bits_last, gamma).mean()
            Kpq = _rbf(P_bits_last, Q_bits_last, gamma).mean()
            return Kpp - 2.0 * Kpq + Kqq


        sigma = jnp.asarray(loss_cfg.mmd_sigma, dtype=jnp.float32)
        gamma = 1.0 / (2.0 * sigma**2)
        lambda_dist = jnp.asarray(loss_cfg.mmd_lambda, dtype=jnp.float32)

        P_bits = (model_onehots_seq @ bit_table).reshape(B, T * D)
        Q_bits = bit_table[Y].reshape(B, T * D)

        def _mmd_masked(P, Q, mP, mQ):
            nP = jnp.sum(mP)
            nQ = jnp.sum(mQ)
            if nP == 0 or nQ == 0:
                return 0.0

            KP = _rbf(P, P, gamma)
            KQ = _rbf(Q, Q, gamma)
            KPQ = _rbf(P, Q, gamma)

            Mpp = mP[:, None] * mP[None, :]
            Mqq = mQ[:, None] * mQ[None, :]
            Mpq = mP[:, None] * mQ[None, :]

            term_pp = jnp.sum(KP * Mpp) / jnp.maximum(nP**2, 1e-9)
            term_qq = jnp.sum(KQ * Mqq) / jnp.maximum(nQ**2, 1e-9)
            term_pq = jnp.sum(KPQ * Mpq) / jnp.maximum(nP * nQ, 1e-9)
            return term_pp - 2.0 * term_pq + term_qq

        m0 = (L == 0).astype(jnp.float32)
        m1 = (L == 1).astype(jnp.float32)

        mmd_P0_R0 = _mmd_masked(P_bits, Q_bits, m0, m0)
        mmd_P1_R1 = _mmd_masked(P_bits, Q_bits, m1, m1)
        mmd_P0_R1 = _mmd_masked(P_bits, Q_bits, m0, m1)
        mmd_P1_R0 = _mmd_masked(P_bits, Q_bits, m1, m0)

        target = jnp.asarray(self.target_mmd_r0_r1, dtype=jnp.float32)

        L0 = mmd_P0_R0 + lambda_dist * (mmd_P0_R1 - target) ** 2
        L1 = mmd_P1_R1 + lambda_dist * (mmd_P1_R0 - target) ** 2

        k = jnp.asarray(loss_cfg.mmd_k, dtype=jnp.float32)
        return jax.nn.logsumexp(k * jnp.stack([L0, L1])) / k


    # ------ SPSA helpers

    def _sample_rademacher_like(self, key, pytree):
        leaves, treedef = jtu.tree_flatten(pytree)
        ks = random.split(key, len(leaves))
        deltas = [jnp.sign(random.normal(k, p.shape)).astype(p.dtype) for p, k in zip(leaves, ks)]
        return jtu.tree_unflatten(treedef, deltas)

    def spsa_step(self, params, opt_state, X, Y, L, key, lr, c):
        key_d, key1, key2 = random.split(key, 3)
        delta = self._sample_rademacher_like(key_d, params)
        params_plus = jtu.tree_map(lambda p, d: p + c * d, params, delta)
        params_minus = jtu.tree_map(lambda p, d: p - c * d, params, delta)

        loss_plus = self.mmd_value_only(params_plus, X, Y, L, key1, self.exp_cfg.learning_mode)
        loss_minus = self.mmd_value_only(params_minus, X, Y, L, key2, self.exp_cfg.learning_mode)

        scale = (loss_plus - loss_minus) / (2.0 * c + 1e-12)
        grads = jtu.tree_map(lambda d: scale * d, delta)

        updates, opt_state = self.opt_core.update(grads, opt_state, params)
        updates_scaled = jtu.tree_map(lambda u: -lr * u, updates)
        params = optax.apply_updates(params, updates_scaled)

        loss_avg = 0.5 * (loss_plus + loss_minus)
        return params, opt_state, float(loss_avg)


    # ------ Training loop

    def run(self):
        cfg = self.config
        key = get_key()

        self.opt_core = optax.chain(
            optax.clip_by_global_norm(cfg.regularization_cfg.clip_norm),
            optax.scale_by_adam(),
            optax.add_decayed_weights(cfg.regularization_cfg.weight_decay)
        )
        key, params_key = random.split(key)
        n_classes = 2 if cfg.model == 3 else 1
        params = self.q_model.initialize_params(params_key, n_classes=n_classes)
        opt_state = self.opt_core.init(params)

        if cfg.model == 3:
            self.target_mmd_r0_r1 = float(self.data_handler.calculate_target_mmd(sigma=float(self.loss_cfg.mmd_sigma)))

        scheduler = Scheduler(cfg)
        mode_controller = ModeController(cfg)
        loss_history = LossHistory()
        start_epoch = 0
        spsa_c = 1e-2

        pbar = tqdm(range(start_epoch, cfg.training_cfg.max_epochs), desc="Training (SPSA)")
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

            params, opt_state, loss_val = self.spsa_step(
                params, opt_state, Xb, Yb, Lb, step_key, current_lr, spsa_c
            )

            loss_dict = {'total': loss_val}
            if cfg.model == 3:
                loss_dict['first_digit_loss'] = loss_val
                loss_dict['second_digit_loss'] = loss_val

            if cfg.scheduler_toggle_cfg.use_complex_scheduler:
                was_fight_mode = mode_controller.state.mode == Mode.FIGHT
                res = mode_controller.update(loss_val, base_lr, epoch)
                if res.should_save_best:
                    self.ckpt_manager.save_checkpoint("best", epoch, params, opt_state, mode_controller.state, key, loss_history)
                if was_fight_mode and not (mode_controller.state.mode == Mode.FIGHT):
                    print(f"\n--- Fight mode ended at epoch {epoch}. Generating analysis plots... ---")
                    key, plot_key = random.split(key)
                    save_loss_plot(self.ckpt_manager.plots_dir, self.config, loss_history, epoch, mode_controller.state)
                    self.visualizer.visualize_samples(
                        params, plot_key, epoch, self.ckpt_manager, mode_controller.state,
                        save_to_disk=True, tag=f"fight_end_epoch_{epoch}"
                    )
                if mode_controller.state.mode == Mode.FLEE:
                    key, noise_key = random.split(key)
                    params = self._add_noise(params, noise_key, cfg.mode_cfg.flee_noise_sigma)
            else:
                if loss_val < mode_controller.state.best_loss:
                    mode_controller.state.best_loss = loss_val
                    self.ckpt_manager.save_checkpoint("best", epoch, params, opt_state, mode_controller.state, key, loss_history)
                if (epoch + 1) % cfg.training_cfg.plot_every_n_epochs == 0 and epoch > 0:
                    key, plot_key = random.split(key)
                    save_loss_plot(self.ckpt_manager.plots_dir, self.config, loss_history, epoch + 1, mode_controller.state)
                    self.visualizer.visualize_samples(
                        params, plot_key, epoch + 1, self.ckpt_manager, mode_controller.state,
                        save_to_disk=True, tag=f"periodic_epoch_{epoch + 1}"
                    )

            loss_history.add(epoch, loss_dict, mode_controller.state.mode, current_lr)
            postfix = {"Loss": f"{loss_val:.4f}", "LR": f"{current_lr:.6f}", "Best": f"{mode_controller.state.best_loss:.4f}"}
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