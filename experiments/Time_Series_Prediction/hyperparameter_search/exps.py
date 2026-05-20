# --- 0. imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

# --- 1. config
RESULTS_DIR = Path("results/experiment")

DATASET_CONFIGS = {
    "snp500": {
        "csv_path":           "https://raw.githubusercontent.com/w-jiwonchoi/HQRNN_dataset/main/model_1_snp.csv",
        "total_days":         200,
        "start_year":         2025,
        "start_month":        1,
        "start_day":          2,
        "normalize_exponent": 6,
    }
}

DEFAULT_DEPTH    = 1
DEFAULT_N_QUBITS = 5
DEFAULT_FFF      = True
DEFAULT_LOSS     = "mmd"

DEPTHS        = [1, 2, 4, 8, 10]
N_QUBITS_LIST = [3, 4, 5, 6, 7]
FFF_FLAGS     = [True, False]
LOSS_FNS      = ["nll", "mmd"]
SEEDS         = [2023, 2024, 2025, 2026, 2027]
SEQ_LEN       = 10
BATCH_SIZE    = 64
MAX_EPOCHS    = 1000

METRICS = ["return_pct", "mdd_pct", "rmse"]
METRIC_LABELS = {
    "return_pct":  "Return (%)",
    "mdd_pct":     "Max Drawdown (%)",
    "rmse":        "Rate RMSE",
}

EXP_TITLES = {
    1: "Circuit Depth",
    2: "Number of Qubits",
    3: "Feature Flag (fff)",
    4: "Loss Function",
}

BOX_COLOR   = "paleturquoise"
BOX_EDGE    = "grey"
MEDIAN_CLR  = "firebrick"
WHISKER_CLR = "grey"
WHISKER_LW  = 1.2
BOX_LW      = 1.2
BOX_WIDTH   = 0.35

METRICS_HIGHER = ["return_pct"]
METRICS_LOWER  = ["mdd_pct", "rmse"]

EXP_GROUP_VAR = {
    1: ("depth",    [str(d) for d in DEPTHS],        DEPTHS),
    2: ("n_qubits", [str(q) for q in N_QUBITS_LIST], N_QUBITS_LIST),
    3: ("fff",      ["True", "False"],               FFF_FLAGS),
    4: ("loss_fn",  LOSS_FNS,                        LOSS_FNS),
}

# --- 2. helper functions
def is_default(depth, n_qubits, fff, loss):
    return (depth == DEFAULT_DEPTH and n_qubits == DEFAULT_N_QUBITS
            and fff == DEFAULT_FFF and loss == DEFAULT_LOSS)

def build_conditions():
    rows = []
    for dataset in DATASET_CONFIGS:
        for seed in SEEDS:
            for d in DEPTHS:
                rows.append((1, dataset, d, DEFAULT_N_QUBITS, DEFAULT_FFF, DEFAULT_LOSS, seed))
            for q in N_QUBITS_LIST:
                if q != DEFAULT_N_QUBITS:
                    rows.append((2, dataset, DEFAULT_DEPTH, q, DEFAULT_FFF, DEFAULT_LOSS, seed))
            for fff in FFF_FLAGS:
                if not is_default(DEFAULT_DEPTH, DEFAULT_N_QUBITS, fff, DEFAULT_LOSS):
                    rows.append((3, dataset, DEFAULT_DEPTH, DEFAULT_N_QUBITS, fff, DEFAULT_LOSS, seed))
            for loss in LOSS_FNS:
                if not is_default(DEFAULT_DEPTH, DEFAULT_N_QUBITS, DEFAULT_FFF, loss):
                    rows.append((4, dataset, DEFAULT_DEPTH, DEFAULT_N_QUBITS, DEFAULT_FFF, loss, seed))
    return rows

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def condition_label(depth, n_qubits, fff, loss, seed):
    return f"d{depth}_q{n_qubits}_fff{int(fff)}_{loss}_s{seed}"

def condition_dir(dataset, depth, n_qubits, fff, loss, seed):
    return RESULTS_DIR / dataset / condition_label(depth, n_qubits, fff, loss, seed)

def _compute_metrics(pred_rates, actual_rates, pred_values, current_prices):
    signals = np.where(pred_values > current_prices, 1.0, 0.0)
    strat_r = signals * actual_rates
    cum = np.cumprod(1.0 + strat_r)
    total_ret = (cum[-1] - 1.0) * 100.0
    running_max = np.maximum.accumulate(cum)
    mdd = float(abs(np.min((cum - running_max) / (running_max + 1e-12))) * 100.0)
    rmse = float(np.sqrt(np.mean((pred_rates - actual_rates) ** 2)))
    return {
        "return_pct": float(total_ret),
        "mdd_pct": mdd,
        "rmse": rmse,
    }

def add_rank_sum(df: pd.DataFrame) -> pd.DataFrame:
    rdf = pd.DataFrame(index=df.index)
    for m in METRICS_HIGHER:
        if m in df.columns:
            rdf[m] = df[m].rank(ascending=False)
    for m in METRICS_LOWER:
        if m in df.columns:
            rdf[m] = df[m].rank(ascending=True)
    df = df.copy()
    df["rank_sum"] = rdf.sum(axis=1)
    return df

def expand_default_rows(all_rows: list, run_exp_ids: list) -> list:
    extra = []
    for row in all_rows:
        if row["exp_id"] != 1:
            continue
        if not is_default(row["depth"], row["n_qubits"], row["fff"], row["loss_fn"]):
            continue
        for eid in run_exp_ids:
            if eid in (2, 3, 4):
                extra.append({**row, "exp_id": eid})
    return all_rows + extra

def run_condition(exp_id, dataset_name, depth, n_qubits, fff, loss, seed, logger):
    import jax.numpy as jnp
    from jax import random

    from hqrnn.config.base import (
        Config, ExperimentConfig, ModelConfig, DatasetConfig,
        TrainingConfig, SchedulerConfig, ModeConfig,
        RegularizationConfig, CheckpointConfig, SchedulerToggleConfig,
    )
    from hqrnn.quantum.model     import QuantumModel
    from hqrnn.data_handler.snp  import SnpDataHandler
    from hqrnn.visualizer.snp    import SnpVisualizer
    from hqrnn.utils.checkpoint  import CheckpointManager
    from hqrnn.trainers.autodiff import Trainer_Autodiff
    from hqrnn.utils.seed        import set_seed

    cond_dir     = condition_dir(dataset_name, depth, n_qubits, fff, loss, seed)
    metrics_path = cond_dir / "metrics.csv"

    if metrics_path.exists():
        df_cached = pd.read_csv(metrics_path)
        df_cached["exp_id"] = exp_id
        return df_cached.to_dict("records")

    set_seed(seed)
    ds = DATASET_CONFIGS[dataset_name]

    cfg = Config(model=1)
    cfg.model_cfg   = ModelConfig(depth=depth, n_D=n_qubits, n_H=n_qubits, seq_len=SEQ_LEN)
    cfg.dataset_cfg = DatasetConfig(
        csv_path=ds["csv_path"], total_days=ds["total_days"],
        start_year=ds["start_year"], start_month=ds["start_month"],
        start_day=ds["start_day"], normalize_exponent=ds["normalize_exponent"],
    )
    cfg.training_cfg = TrainingConfig(
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
        learning_rate_init=0.01, learning_rate_min=1e-6,
        plot_every_n_epochs=99999,
    )
    cfg.scheduler_cfg = SchedulerConfig(
        restart_period_epochs=20, restart_period_multiplier=2.0,
        restart_decay_rate=0.9, warmup_step_ratio=0.05,
        flee_warmup_bind=True, fight_end_use_warm_start=True,
    )
    cfg.mode_cfg = ModeConfig(
        fight_blocks_restart=True, fight_in_fight_threshold=0.5,
        fight_in_fight_window=40, fight_decay_rate=0.01,
        super_fight_lr_boost=0.5, protection_proportion=0.5,
        find_mode_sub_start=0.50, find_mode_sub_end=0.30,
        find_fight_rate=2.0, patience_num=100, cooldown_steps=10,
        stop_condition=10, ema_beta=0.90, flee_steps=0,
        flee_noise_sigma=0.0, volatility_window=50,
        volatility_threshold_start=0.40, volatility_threshold_end=0.30,
        volatility_patience=5,
    )
    cfg.regularization_cfg   = RegularizationConfig(weight_decay=0.0, clip_norm=1e-6)
    cfg.scheduler_toggle_cfg = SchedulerToggleConfig(use_complex_scheduler=fff)
    cfg.checkpoint_cfg = CheckpointConfig(
        base_checkpoint_dir=f"runs/experiment/{dataset_name}/{condition_label(depth, n_qubits, fff, loss, seed)}",
        max_keep_checkpoints=99,
        allow_hash_mismatch=False,
    )
    cfg.exp_cfg = ExperimentConfig(
        loss_function=loss, collapse_type="soft",
        learning_mode="teacher_forcing",
    )

    data_handler = SnpDataHandler(config=cfg)
    q_model      = QuantumModel(config=cfg)
    ckpt_manager = CheckpointManager(config=cfg)

    if not (ckpt_manager.ckpt_dir / "best.pkl").exists() and not (ckpt_manager.ckpt_dir / "final.pkl").exists():
        visualizer = SnpVisualizer(config=cfg, q_model=q_model)
        visualizer.set_data_handler(data_handler)
        trainer = Trainer_Autodiff(
            config=cfg, q_model=q_model, data_handler=data_handler,
            visualizer=visualizer, ckpt_manager=ckpt_manager,
        )
        trainer.run()

    visualizer = SnpVisualizer(config=cfg, q_model=q_model)
    visualizer.set_data_handler(data_handler)

    X_full = jnp.concatenate([data_handler.X_train, data_handler.X_test], axis=0)
    n_all  = int(X_full.shape[0])
    keys   = random.split(random.PRNGKey(seed), n_all)

    orig      = data_handler.original_values
    prices    = orig[SEQ_LEN: SEQ_LEN + n_all + 1]

    act_rates = np.array(
        [(prices[i+1] - prices[i]) / prices[i] for i in range(n_all)],
        dtype=float,
    )
    current_prices = prices[:n_all]

    target_ckpts = []
    for f in ckpt_manager.ckpt_dir.glob("*.pkl"):
        name = f.stem
        if name.startswith("fightend") or name in ["best", "final"]:
            target_ckpts.append(name)

    cond_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for ckpt_name in target_ckpts:
        try:
            params = ckpt_manager.load_checkpoint(ckpt_name)[0]
        except Exception:
            continue

        bits_list = [
            visualizer._predict_next(params, keys[i], X_full[i])
            for i in range(n_all)
        ]
        all_bits   = jnp.stack(bits_list)
        pred_rates = np.array(data_handler._denormalize(all_bits), dtype=float)
        pred_values = prices[0] * np.cumprod(1.0 + pred_rates)

        metrics = _compute_metrics(pred_rates, act_rates, pred_values, current_prices)
        signals = np.where(pred_values > current_prices, 1.0, 0.0)

        pd.DataFrame({
            "step":         np.arange(n_all),
            "pred_rate":    pred_rates,
            "actual_rate":  act_rates,
            "actual_price": current_prices,
            "pred_value":   pred_values,
            "signal":       signals,
            "strategy_ret": signals * act_rates,
            "cum_strategy": np.cumprod(1.0 + signals * act_rates),
            "cum_bnh":      np.cumprod(1.0 + act_rates),
        }).to_csv(cond_dir / f"predictions_{ckpt_name}.csv", index=False)

        row = dict(exp_id=exp_id, dataset=dataset_name, depth=depth, n_qubits=n_qubits,
                   fff=fff, loss_fn=loss, seed=seed, checkpoint=ckpt_name, **metrics)
        rows.append(row)

    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    return rows

def compute_summary_stats(df: pd.DataFrame, exp_id: int, dataset_name: str) -> list:
    gvar, labels, vals = EXP_GROUP_VAR[exp_id]
    stats_rows = []
    for metric in METRICS:
        groups = [df[df[gvar] == v][metric].dropna().values for v in vals]
        for v, lbl, grp in zip(vals, labels, groups):
            stats_rows.append({
                "exp_id": exp_id, "dataset": dataset_name,
                "group_var": gvar, "group_val": str(lbl), "metric": metric,
                "n":      len(grp),
                "mean":   float(np.mean(grp))   if len(grp) else np.nan,
                "median": float(np.median(grp)) if len(grp) else np.nan,
                "std":    float(np.std(grp))    if len(grp) else np.nan,
                "q25":    float(np.percentile(grp, 25)) if len(grp) else np.nan,
                "q75":    float(np.percentile(grp, 75)) if len(grp) else np.nan,
                "min":    float(np.min(grp))    if len(grp) else np.nan,
                "max":    float(np.max(grp))    if len(grp) else np.nan,
            })
    return stats_rows

# --- 3. plotting functions
def make_bxp_stats(row):
    return dict(
        med    = row["median"],
        q1     = row["q25"],
        q3     = row["q75"],
        whislo = row["min"],
        whishi = row["max"],
        fliers = [],
        label  = str(row["group_val"]),
    )

# --- 4. main execution
def main(run_exp_ids: list):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger    = setup_logger(RESULTS_DIR / "experiment.log")
    plots_dir = RESULTS_DIR / "boxplots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_conditions = build_conditions()
    conditions = [c for c in all_conditions if c[0] in run_exp_ids]

    cached  = [c for c in conditions
               if (condition_dir(c[1], c[2], c[3], c[4], c[5], c[6]) / "metrics.csv").exists()]
    pending = [c for c in conditions
               if not (condition_dir(c[1], c[2], c[3], c[4], c[5], c[6]) / "metrics.csv").exists()]

    all_rows = []

    for c in tqdm(cached, desc="Loading cache", unit="cond"):
        exp_id, dataset, depth, n_qubits, fff, loss, seed = c
        mpath = condition_dir(dataset, depth, n_qubits, fff, loss, seed) / "metrics.csv"
        try:
            rows = pd.read_csv(mpath).to_dict("records")
            for r in rows:
                r["exp_id"] = exp_id
            all_rows.extend(rows)
        except Exception:
            pass

    for c in tqdm(pending, desc="Running experiments", unit="cond"):
        exp_id, dataset, depth, n_qubits, fff, loss, seed = c
        try:
            rows = run_condition(exp_id, dataset, depth, n_qubits, fff, loss, seed, logger)
            all_rows.extend(rows)
        except Exception:
            pass

    if not all_rows:
        return

    all_rows = expand_default_rows(all_rows, run_exp_ids)

    df_all = add_rank_sum(pd.DataFrame(all_rows))
    df_all.to_csv(RESULTS_DIR / "all_results.csv", index=False)

    all_stats = []
    for exp_id in run_exp_ids:
        for dataset_name in DATASET_CONFIGS:
            sub = df_all[(df_all["exp_id"] == exp_id) & (df_all["dataset"] == dataset_name)].copy()
            if sub.empty:
                continue
            sub = add_rank_sum(sub.reset_index(drop=True))
            out_path = RESULTS_DIR / f"exp{exp_id}_{dataset_name}_results.csv"
            sub.to_csv(out_path, index=False)
            all_stats.extend(compute_summary_stats(sub, exp_id, dataset_name))

    stats_csv_path = RESULTS_DIR / "boxplot_stats.csv"
    pd.DataFrame(all_stats).to_csv(stats_csv_path, index=False)

    df_plot = pd.read_csv(stats_csv_path)

    stat_cols = ["min", "max", "median", "q25", "q75"]
    df_plot.loc[df_plot["metric"] == "rmse", stat_cols] *= 100

    exclude_vals = ["B&H", "Buy-and-Hold", "Buy and Hold"]
    df_plot = df_plot[~df_plot["group_val"].astype(str).str.strip().isin(exclude_vals)]

    df_plot["group_val"] = df_plot["group_val"].replace({"Ours1": "Ours", "Ours2": "Ours"})

    for exp_id, exp_df in df_plot.groupby("exp_id"):
        for dataset, ds_df in exp_df.groupby("dataset"):

            group_var = ds_df["group_var"].iloc[0]

            raw_vals = ds_df["group_val"].unique()
            try:
                group_vals = sorted(raw_vals, key=float)
            except (ValueError, TypeError):
                group_vals = sorted(raw_vals, key=str)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(
                f"{EXP_TITLES[exp_id]}  -  {dataset.upper()}",
                fontsize=13, fontweight="bold", y=1.05,
            )

            for ax, metric in zip(axes.flat, METRICS):

                stats_list = []
                for gv in group_vals:
                    sel = ds_df[(ds_df["group_val"] == gv) & (ds_df["metric"] == metric)]
                    if sel.empty:
                        continue
                    stats_list.append(make_bxp_stats(sel.iloc[0]))

                if not stats_list:
                    ax.set_visible(False)
                    continue

                bp = ax.bxp(
                    stats_list,
                    showfliers  = False,
                    patch_artist= True,
                    widths      = BOX_WIDTH,
                    boxprops    = dict(facecolor=BOX_COLOR, color=BOX_EDGE, linewidth=BOX_LW),
                    medianprops = dict(color=MEDIAN_CLR, linewidth=2.0),
                    whiskerprops= dict(color=WHISKER_CLR, linewidth=WHISKER_LW),
                    capprops    = dict(color=WHISKER_CLR, linewidth=WHISKER_LW),
                )

                ax.set_title(METRIC_LABELS[metric], fontsize=11, pad=8)
                ax.set_xlabel(group_var, fontsize=9, labelpad=4)
                ax.tick_params(axis="both", labelsize=8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(0.8)
                ax.spines["bottom"].set_linewidth(0.8)
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))
                ax.set_facecolor("#FAFAFA")
                ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, color="gray")
                ax.set_axisbelow(True)

            plt.tight_layout()
            out = plots_dir / f"boxplot_exp{exp_id}_{dataset}.png"
            plt.savefig(out, dpi=180, bbox_inches="tight")
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HQRNN ablation study")
    parser.add_argument(
        "--exp", type=int, nargs="+", choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
    )
    args = parser.parse_args()
    main(run_exp_ids=args.exp)
