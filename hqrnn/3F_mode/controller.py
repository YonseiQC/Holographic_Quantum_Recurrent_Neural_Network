import numpy as np
from hqrnn.config.base import Config
from .types import Mode, UnifiedModeState, ModeUpdateResult

# --- 5. Mode Controller

class ModeController:
    def __init__(self, config: Config):
        self.cfg = config
        self.state = UnifiedModeState(config=self.cfg)  # Initialize runtime mode state


    # ------ enter fight mode

    def _enter_fight_mode(self, st: UnifiedModeState, current_lr: float, epoch: int, res: ModeUpdateResult):
        st.mode = Mode.FIGHT  # Switch to fight
        st.no_improve = 0  # Reset patience counter
        st.cooldown = self.cfg.mode_cfg.cooldown_steps
        st.fight_count += 1
        st.fight_entry_lr = current_lr  # Record LR at entry
        st.fight_best_hits.clear()
        st.is_super_fight = False
        st.super_fight_duration = 0
        st.volatility_strikes = 0
        st.recent_losses.clear()
        st.fight_start_epoch = epoch  # Record fight start epoch
        res.mode_changed = True  # Report mode change


    # ------ super-fight checker

    def _check_super_fight(self, st: UnifiedModeState, epoch: int) -> bool:
        if len(st.fight_best_hits) < self.cfg.mode_cfg.fight_in_fight_window:
            return False

        hit_rate = sum(st.fight_best_hits) / len(st.fight_best_hits)  # Hit rate formula

        if hit_rate >= self.cfg.mode_cfg.fight_in_fight_threshold and not st.is_super_fight and st.super_fight_duration == 0:
            st.is_super_fight = True  # Promote to super-fight
            st.super_fight_entry_epoch = epoch  # Record entry epoch
            st.super_fight_duration = 1  # Initialize duration
            print(f"Super-Fight activated! Hit rate: {hit_rate:.2f}")
            return True

        elif st.is_super_fight:
            st.super_fight_duration += 1
            if st.super_fight_duration > 100:  # Super-fight for 100epochs
                st.is_super_fight = False
                st.super_fight_duration = 0
                print(f"Super-Fight ended after 100 epochs")

        return False


    # ------ main mode update

    def update(self, total_loss: float, current_lr: float, epoch: int) -> ModeUpdateResult:
        st = self.state
        res = ModeUpdateResult()

        if st.finished:
            return res

        ema_prev = total_loss if st.ema_loss == float('inf') else st.ema_loss
        rel_drop = (ema_prev - total_loss) / (ema_prev + 1e-12)
        st.ema_loss = self.cfg.mode_cfg.ema_beta * ema_prev + (1.0 - self.cfg.mode_cfg.ema_beta) * total_loss  # EMA update

        if st.cooldown > 0:
            st.cooldown -= 1

        if epoch < self.cfg.training_cfg.max_epochs * self.cfg.mode_cfg.protection_proportion:
            st.last_loss = total_loss  # Protection window: no mode changes
            if total_loss < st.best_loss:
                st.best_loss = total_loss
                res.should_save_best = True
            return res

        improved_this_step = False  # Track new global best within this call
        if total_loss < st.best_loss:
            print(f"\nNew best total loss: {st.best_loss:.6f} -> {total_loss:.6f}")
            st.best_loss = total_loss
            improved_this_step = True
            res.should_save_best = True
            if st.mode == Mode.FIGHT:
                st.no_improve = 0  # Reset patience on improvement during fight
        elif st.mode == Mode.FIGHT:
            st.no_improve += 1  # Count non-improving steps during fight

        if st.mode == Mode.FIND:
            progress = epoch / self.cfg.training_cfg.max_epochs
            dyn_drop = self.cfg.mode_cfg.find_mode_sub_start - (self.cfg.mode_cfg.find_mode_sub_start - self.cfg.mode_cfg.find_mode_sub_end) * progress
            dyn_volt = self.cfg.mode_cfg.volatility_threshold_start - (self.cfg.mode_cfg.volatility_threshold_start - self.cfg.mode_cfg.volatility_threshold_end) * progress

            volatility_triggered = False
            st.recent_losses.append(total_loss)
            if len(st.recent_losses) == self.cfg.mode_cfg.volatility_window:
                losses_arr = np.array(st.recent_losses)
                mean_loss, std_loss = np.mean(losses_arr), np.std(losses_arr)
                cv = std_loss / (mean_loss + 1e-12)
                if cv > dyn_volt:
                    st.volatility_strikes += 1
                else:
                    st.volatility_strikes = 0
                if st.volatility_strikes >= self.cfg.mode_cfg.volatility_patience:
                    volatility_triggered = True

            if st.flee_count >= self.cfg.mode_cfg.stop_condition:
                st.finished = True

            elif st.cooldown == 0 and improved_this_step:
                print("Entering FIGHT by breaking All-Time Best!")
                self._enter_fight_mode(st, current_lr, epoch, res)  # Promote on new best

            elif st.cooldown == 0 and rel_drop >= dyn_drop:
                print(f"\nEntering FIGHT via Sharp Drop (drop: {rel_drop:.4f} > {dyn_drop:.4f})")
                self._enter_fight_mode(st, current_lr, epoch, res)  # Promote on sharp loss drop

            elif st.cooldown == 0 and volatility_triggered:
                print(f"\nEntering FIGHT via High Volatility (CV trigger)")
                self._enter_fight_mode(st, current_lr, epoch, res)  # Promote on volatility
                res.triggered_by_volatility = True

        elif st.mode == Mode.FIGHT:
            st.fight_best_hits.append(improved_this_step)

            if self._check_super_fight(st, epoch):
                res.mode_changed = True

            if st.no_improve >= self.cfg.mode_cfg.patience_num and st.cooldown == 0:
                st.mode = Mode.FLEE
                st.cooldown = self.cfg.mode_cfg.cooldown_steps
                st.flee_count += 1
                st.is_super_fight = False
                st.super_fight_duration = 0

                st.fight_end_epoch = epoch  # Record fight end
                fight_epochs = max(1, int(st.fight_end_epoch - (st.fight_start_epoch if st.fight_start_epoch >= 0 else epoch)))  # Fight length
                ratio = float(self.cfg.scheduler_cfg.warmup_step_ratio)
                flee_steps_from_fight = max(1, int(np.ceil(fight_epochs * ratio)))  # Tie flee to recent fight length
                st.flee_left = flee_steps_from_fight

                res.mode_changed = True
                res.should_generate_flee_samples = True
                print(f"\nEntering FLEE after {st.no_improve} steps without improvement "
                      f"(fight_len={fight_epochs}, flee_left={st.flee_left}, ratio={ratio:.4f})")

        elif st.mode == Mode.FLEE:
            st.flee_left -= 1
            if st.flee_left <= 0 and st.cooldown == 0:
                st.mode = Mode.FIND  # Return to find
                st.no_improve = 0
                st.cooldown = self.cfg.mode_cfg.cooldown_steps
                st.ema_loss = total_loss
                res.mode_changed = True
                print("\nReturning to FIND mode.")

        st.last_loss = total_loss  # Persist last loss
        return res


    # ------ learning-rate multiplier

    def get_lr_multiplier(self) -> float:
        if self.state.mode == Mode.FIGHT:
            base_multiplier = 1.0 / self.cfg.mode_cfg.find_fight_rate  # Reduce LR in fight
            if self.state.is_super_fight:
                multiplier = base_multiplier * self.cfg.mode_cfg.super_fight_lr_boost  #  Reduce LR in super-fight
                return max(multiplier, 0.1)
            return base_multiplier
        return 1.0