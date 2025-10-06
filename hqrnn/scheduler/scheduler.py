from typing import Tuple
import jax.numpy as jnp
import optax
from hqrnn.config.base import Config
from hqrnn.3F_mode.types import UnifiedModeState, Mode

# --- 3. Scheduler

class Scheduler:
    def __init__(self, config: Config):
        self.config = config

        if not self.config.scheduler_toggle_cfg.use_complex_scheduler:
            print("Scheduler Mode: Simple Cosine Annealing")
            self.simple_cosine_schedule = optax.cosine_decay_schedule(
                init_value=self.config.training_cfg.learning_rate_init,
                decay_steps=self.config.training_cfg.max_epochs,
                alpha=self.config.training_cfg.learning_rate_min / self.config.training_cfg.learning_rate_init
            )  # One-shot cosine decay without restarts
            self.cycles, self.current_cycle_idx, self.local_warmup_active = [], 0, False  # Placeholders for simple mode
            self.local_warmup_start, self.local_warmup_steps, self.local_warmup_target_peak = -1, 0, None
            self.fight_active, self.fight_start_step, self.fight_entry_lr = False, -1, 0.0
            return

        print("Scheduler Mode: Complex")
        self.cycles = self._build_cycle()
        self.current_cycle_idx = 0
        self.cycle_anchor_start = 0
        self.local_warmup_active = False
        self.local_warmup_start = -1
        self.local_warmup_steps = 0
        self.local_warmup_target_peak = None
        self.fight_active = False
        self.fight_start_step = -1
        self.fight_entry_lr = 0.0
        self.is_in_warm_start_decay = False

    def _build_cycle(self):
        cfg_s = self.config.scheduler_cfg
        total = self.config.training_cfg.max_epochs
        cycles, period, peak, made = [], max(1, int(cfg_s.restart_period_epochs)), float(self.config.training_cfg.learning_rate_init), 0
        while made < total:  # Makes another cycle if total epoch not ended
            steps_in_cycle = min(period, total - made)  # Trim last cycle to left total epoch
            cycles.append({"start_step": made,
                           "end_step": made + steps_in_cycle,
                           "steps": steps_in_cycle,
                           "peak_value": peak,
                           "is_final": (made + steps_in_cycle >= total)})
            made += steps_in_cycle
            if not cycles[-1]["is_final"]:
                period = max(1, int(period * cfg_s.restart_period_multiplier))  # Lengthen next cycle
                peak *= cfg_s.restart_decay_rate  # Decay next cycle peak
        return cycles


    @staticmethod
    def _cosine_w_cycle(step_in_cycle: int, cycle_info: dict, lr_min: float) -> float:
        total_steps = max(1, cycle_info["steps"])
        if step_in_cycle >= total_steps:
            return lr_min  # Drop to lr_min after the cycle ends
        progress = step_in_cycle / total_steps
        cosine_val = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        lr = lr_min + (cycle_info["peak_value"] - lr_min) * cosine_val
        return float(jnp.maximum(lr, lr_min))


    @staticmethod
    def _linear_warmup(t: int, T: int, lr_min: float, lr_peak: float) -> float:
        if T <= 0:
            return lr_peak
        s = max(0, min(t, T))
        return float(lr_min + (lr_peak - lr_min) * (s / T))


    def get_lr(self, step: int, mode_state: UnifiedModeState) -> Tuple[float, bool]:
        if not self.config.scheduler_toggle_cfg.use_complex_scheduler:
            lr = self.simple_cosine_schedule(step)
            return float(jnp.maximum(lr, self.config.training_cfg.learning_rate_min)), False  # Simple mode path

        lr_min = self.config.training_cfg.learning_rate_min
        mode_state.is_warming_up = False
        current = self.cycles[self.current_cycle_idx]

        if self.is_in_warm_start_decay and (step - self.cycle_anchor_start) >= current["steps"]:
            self.is_in_warm_start_decay = False  # Allow normal cycle warmup next cycle

        is_fight_or_super = (mode_state.mode == Mode.FIGHT) or mode_state.is_super_fight  # Fight covers super-fight

        if self.fight_active and not is_fight_or_super:
            self.fight_active = False  # Fight ended this step
            mode_state.fight_end_epoch = step  # For logging

            fight_len = max(1, step - (self.fight_start_step if self.fight_start_step >= 0 else step))  # Fight duration
            ratio = float(self.config.scheduler_cfg.warmup_step_ratio)  # Warmup length scale

            if self.config.scheduler_cfg.fight_end_use_warm_start:
                self.local_warmup_active = True
                self.local_warmup_start = step
                self.local_warmup_steps = max(1, int(round(fight_len * ratio)))  # Warmup tied to fight length

            prev_peak = float(current["peak_value"])
            total_epochs = max(1, self.config.training_cfg.max_epochs)
            decay = float(self.config.scheduler_cfg.restart_decay_rate)
            new_peak = prev_peak * decay * max(0.0, 1.0 - (fight_len / total_epochs))  # New peak formula
            new_peak = max(new_peak, lr_min)

            if self.current_cycle_idx < len(self.cycles) - 1:
                self.current_cycle_idx += 1  # Advance to next cycle after fight
            current = self.cycles[self.current_cycle_idx]
            current["peak_value"] = float(new_peak)  # Set new peak for the resumed cycle
            self.cycle_anchor_start = step  # Anchor cycle start at resume step
            self.local_warmup_target_peak = float(new_peak) if self.config.scheduler_cfg.fight_end_use_warm_start else None  # Warmup target
            mode_state.last_restart_warm_start = bool(self.config.scheduler_cfg.fight_end_use_warm_start)
            self.is_in_warm_start_decay = False

        if is_fight_or_super:
            if not self.fight_active:
                self.fight_active = True
                self.fight_start_step = step
                step_in_cycle = step - self.cycle_anchor_start
                baseline = self._cosine_w_cycle(step_in_cycle, current, lr_min) if not self.local_warmup_active else current["peak_value"]  # Entry LR baseline
                self.fight_entry_lr = max(float(baseline), float(mode_state.fight_entry_lr or 0.0))  # Preserve higher entry LR
                mode_state.fight_start_epoch = step
            progress = step - self.fight_start_step
            decay = self.config.mode_cfg.fight_decay_rate
            start_lr = max(self.fight_entry_lr, lr_min)
            lr = lr_min + (start_lr - lr_min) * jnp.exp(-decay * progress)  # Exponential decay during fight
            return float(jnp.maximum(lr, lr_min)), False

        if self.local_warmup_active:
            t = step - self.local_warmup_start
            if t < self.local_warmup_steps:
                mode_state.is_warming_up = True
                target_peak = float(self.local_warmup_target_peak or current["peak_value"])
                lr = self._linear_warmup(t=t, T=self.local_warmup_steps, lr_min=lr_min, lr_peak=target_peak)
                return float(jnp.maximum(lr, lr_min)), False
            else:
                lr = float(self.local_warmup_target_peak or current["peak_value"])
                self.cycle_anchor_start = step
                self.local_warmup_active = False
                self.local_warmup_start = -1
                self.local_warmup_steps = 0
                self.local_warmup_target_peak = None
                self.is_in_warm_start_decay = True
                return float(jnp.maximum(lr, lr_min)), False

        step_in_cycle = step - self.cycle_anchor_start
        restart_happened = (step_in_cycle == 0 and step > 0)
        warmup_steps_cycle = max(1, int(round(current["steps"] * float(self.config.scheduler_cfg.warmup_step_ratio))))  # Cycle warmup length
        mode_state.cycle_warmup_steps_hint = warmup_steps_cycle

        if step_in_cycle < warmup_steps_cycle and not self.is_in_warm_start_decay:
            mode_state.is_warming_up = True
            lr = self._linear_warmup(t=step_in_cycle, T=warmup_steps_cycle, lr_min=lr_min, lr_peak=current["peak_value"])  # Regular cycle warmup
            return float(jnp.maximum(lr, lr_min)), restart_happened

        if step_in_cycle >= current["steps"] and self.current_cycle_idx < len(self.cycles) - 1:
            self.current_cycle_idx += 1  # Move to next cycle
            self.cycle_anchor_start = step  # Reset anchor to this step
            current = self.cycles[self.current_cycle_idx]
            step_in_cycle = 0
            restart_happened = True  # Cosine restart occurred

        lr = self._cosine_w_cycle(step_in_cycle, current, lr_min)  # Cosine within cycle
        return float(jnp.maximum(lr, lr_min)), restart_happened