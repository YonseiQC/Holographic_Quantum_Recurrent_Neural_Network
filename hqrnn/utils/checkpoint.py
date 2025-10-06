import os
import json
import pickle
from pathlib import Path
from dataclasses import asdict
from collections import deque, defaultdict
from typing import Dict, Any, Optional

import numpy as np
import jax.numpy as jnp
from jax import tree_util as jtu

from hqrnn.config.base import Config, ExperimentConfig
from hqrnn.FFF_mode.types import Mode, UnifiedModeState
from hqrnn.scheduler.loss_history import LossHistory


# --- 6. Checkpoint Manager

class CheckpointManager:
    def __init__(self, config: Config, hour: Optional[int] = None):
        self.config = config
        base_dir = Path(config.checkpoint_cfg.base_checkpoint_dir) / f"Q{config.model_cfg.n_D}D{config.model_cfg.depth}_B{config.training_cfg.batch_size}_h{config.get_hash()}"  # Unique run folder by hyperparam hash

        if config.model == 2 and hour is not None:
            self.exp_dir = base_dir / f"hour_{hour}"  # Split directories by hour for model 2
        else:
            self.exp_dir = base_dir

        self.ckpt_dir = self.exp_dir / "ckpt"
        self.plots_dir = self.exp_dir / "plots"
        self.csv_dir = self.exp_dir / "csv"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)  # Ensure directories exist
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        print(f"Checkpoint directory: {self.exp_dir}")

        config_path = self.exp_dir / "config.json"
        if not config_path.exists():
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=4)  # Persist run config at start
            print(f"Configuration saved to {config_path}")

        self.meta_path = self.exp_dir / "meta.json"
        self.meta = self._load_meta()  # Load or initialize checkpoint index
        print(f"CSV outputs will be saved to: {self.csv_dir}")


    # ------ meta I/O

    def _load_meta(self) -> Dict[str, Any]:
        if self.meta_path.exists():
            try:
                with open(self.meta_path, "r") as f:
                    obj = json.load(f)
                obj.setdefault("checkpoints", [])
                obj.setdefault("best_epoch", None)
                obj.setdefault("best_loss", None)
                return obj
            except Exception:
                pass
        return {"checkpoints": [], "best_epoch": None, "best_loss": None}

    def _save_meta(self):
        tmp = self.meta_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.meta, f, indent=2)
        os.replace(tmp, self.meta_path)  # Atomic replace (to avoid partial writes)

    def _register_ckpt(self, filename: str, epoch: int):
        entry = {"file": filename, "epoch": int(epoch)}
        self.meta["checkpoints"].append(entry)
        seen, uniq = set(), []
        for e in reversed(self.meta["checkpoints"]):
            if e["file"] in seen:
                continue  # Keep only the most recent entry per file
            seen.add(e["file"])
            uniq.append(e)
        self.meta["checkpoints"] = list(reversed(uniq))

    def _prune_old_checkpoints(self):
        max_keep = int(self.config.checkpoint_cfg.max_keep_checkpoints)
        if max_keep <= 0:
            return
        all_files = sorted([p for p in self.ckpt_dir.glob("*.pkl") if p.name != "best.pkl"], key=lambda p: p.stat().st_mtime, reverse=True)  # Newest first, keep 'best' always
        keep = all_files[:max_keep]
        drop = all_files[max_keep:]
        for p in drop:
            try:
                p.unlink()
            except Exception:
                pass
        keep_names = {"best.pkl"} | {p.name for p in keep}
        self.meta["checkpoints"] = [e for e in self.meta["checkpoints"] if Path(e["file"]).name in keep_names]
        self._save_meta()

    def _atomic_pickle_dump(self, path: Path, obj: Dict[str, Any]):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)


    # ------ save/load checkpoints

    def save_checkpoint(
        self,
        tag: str,
        epoch: int,
        params: Dict,
        opt_state: Any,
        mode_state: UnifiedModeState,
        rng_key: Any,
        loss_history: LossHistory = None,
    ):
        mode_state_dict = asdict(mode_state)
        mode_state_dict.pop('config', None)
        mode_state_dict['recent_losses'] = list(mode_state.recent_losses)  # Convert deques to lists
        mode_state_dict['fight_best_hits'] = list(mode_state.fight_best_hits)

        path = (self.ckpt_dir if tag not in ("best", "last") else self.ckpt_dir) / f"{tag}.pkl"  # Same dir for all tags

        data = {
            "params": jtu.tree_map(np.array, params),
            "opt_state": jtu.tree_map(np.array, opt_state),
            "mode_state": mode_state_dict,
            "epoch": int(epoch),
            "rng_key": np.asarray(rng_key, dtype=np.uint32).tolist(),
            "config": self.config.to_dict(),
        }
        if loss_history is not None:
            data["loss_history"] = asdict(loss_history)

        self._atomic_pickle_dump(path, data)
        print(f"Saved '{tag}' checkpoint at epoch {epoch} -> {path}")

        self._register_ckpt(str(path), epoch)
        self._save_meta()
        self._prune_old_checkpoints()

    def save_epoch_checkpoint(self, epoch: int, *args, **kwargs):
        tag = f"e{int(epoch):06d}"
        return self.save_checkpoint(tag, epoch, *args, **kwargs)

    def save_last(self, epoch: int, *args, **kwargs):
        return self.save_checkpoint("last", epoch, *args, **kwargs)

    def save_best(self, epoch: int, best_loss: float, *args, **kwargs):
        self.meta["best_epoch"] = int(epoch)
        self.meta["best_loss"] = float(best_loss)
        self._save_meta()
        return self.save_checkpoint("best", epoch, *args, **kwargs)

    def load_checkpoint(self, tag: str = "best"):
        path = self.ckpt_dir / f"{tag}.pkl"
        if not path.exists():
            if self.meta.get("checkpoints"):
                cand = self.meta["checkpoints"][-1]["file"]  # Fallback to most recent
                path = Path(cand)
                if not path.exists():
                    return None
            else:
                return None

        with open(path, "rb") as f:
            data = pickle.load(f)

        saved_cfg_dict = data.get("config", {})
        saved_model_num = saved_cfg_dict.get('model')
        if saved_model_num is None:
            raise ValueError("Checkpoint config is missing 'model' key.")

        saved_cfg = Config(model=saved_model_num)
        exp_cfg_backup = saved_cfg_dict.pop('exp_cfg', None)
        saved_cfg_dict.pop('model', None)

        for key, value_dict in saved_cfg_dict.items():
            if hasattr(saved_cfg, key) and value_dict is not None and isinstance(value_dict, dict):
                try:
                    field_type = type(getattr(saved_cfg, key))
                    setattr(saved_cfg, key, field_type(**value_dict))
                except TypeError:
                    print(f"Warning: Could not reload nested config key '{key}'. Using default.")

        if exp_cfg_backup:
            saved_cfg.exp_cfg = ExperimentConfig(**exp_cfg_backup)

        if saved_cfg.get_hash() != self.config.get_hash() and not self.config.checkpoint_cfg.allow_hash_mismatch:
            raise ValueError(f"Config hash mismatch! Checkpoint: {saved_cfg.get_hash()}, Current: {self.config.get_hash()}")

        params = jtu.tree_map(jnp.array, data['params'])
        opt_state = jtu.tree_map(jnp.array, data['opt_state'])

        mode_data = data["mode_state"]
        mode_data['mode'] = Mode(mode_data['mode'])

        mode_state = UnifiedModeState(config=self.config)
        for k, v in mode_data.items():
            if hasattr(mode_state, k):
                setattr(mode_state, k, v)

        mode_state.recent_losses = deque(mode_data.get('recent_losses', []), maxlen=self.config.mode_cfg.volatility_window)
        mode_state.fight_best_hits = deque(mode_data.get('fight_best_hits', []), maxlen=self.config.mode_cfg.fight_in_fight_window)

        epoch = int(data["epoch"])
        rng_key = jnp.asarray(data["rng_key"], dtype=jnp.uint32)

        loss_history = None
        if "loss_history" in data:
            loss_history_data = data["loss_history"]
            loss_history_data['losses'] = defaultdict(list, loss_history_data.get('losses', {}))
            loss_history = LossHistory(**loss_history_data)

        print(f"Resumed from '{tag}' checkpoint at epoch {epoch}")
        return params, opt_state, mode_state, epoch, rng_key, loss_history