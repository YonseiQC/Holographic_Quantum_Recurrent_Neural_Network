#!/usr/bin/env python3
"""
hqrnn_hw_experiment.py
======================

HQRNN Model 1 (S&P 500 forecasting) and Model 3 (7x7 digit generation) trained
with SPSA and evaluated **in a single run**, on an IBM Quantum backend, on a
device-calibrated noisy simulator, or on the ideal Aer simulator.

What this script does
---------------------
* Trains both models from scratch with SPSA + MMD for `--epochs` epochs
  (default 1000).  On hardware both models advance one epoch per submitted job,
  so the two experiments share one Batch and one job stream.
* Model 1: every `--eval-every` epochs (default 10) it runs teacher-forced
  inference over **all 200 windows with 10 shots each** (= 2000 single-shot
  predictions), averages the 10 sampled rates per date into one prediction,
  writes a CSV + metrics row + a frame, and finally renders the evolution of
  the prediction curve as GIF/MP4.
* Model 3: every `--eval-every` epochs it generates digit-0 and digit-1 samples,
  builds KDE stacks (raw samples *and* CNN-filtered samples), and finally
  renders both as GIF/MP4.  1000 epochs / 10 = 100 frames.
* `--ablation` runs the whole ten-arm topology / error-suppression sweep back to
  back and writes one comparison table.

Hardware-efficiency design (see "TOPOLOGY" below)
-------------------------------------------------
The default ansatz topology is `hw_line`, not the paper's double-ring:
physical qubits are chosen as a **path** in the backend coupling map, data and
hidden qubits alternate along that path (D H D H ...), and RZZ gates act only on
physically adjacent pairs, split into an even/odd brickwork.  Consequences:

  * zero SWAP gates after transpilation (verified at runtime),
  * two-qubit depth per ansatz layer is exactly 2 RZZ layers,
  * the interaction graph is still connected, and combined with the time axis it
    forms the 2D lattice that the HQRNN paper's IQP hardness argument relies on,
  * `--topology cross_only` (rings removed, D_i-H_i matching only) and
    `--topology ring` (original paper ansatz) remain available for comparison.

Other hardware notes
--------------------
* Model 1 needs mid-circuit `reset` (teacher forcing feeds the *actual* rates).
  Intermediate rows are reset but not measured -- their outcomes are unused, and
  reset already performs the projection that drives the recurrence.
* Model 3 needs **no reset and no re-encoding**: after measuring a data qubit in
  the Z basis it is left in |b>, which is exactly RY(pi*b)|0>, i.e. the encoded
  input of the next timestep.  Autoregressive generation is therefore native.
* Circuits are transpiled **once** as parametric circuits; every SPSA
  perturbation, every window and every label is sent as a row of the PUB
  parameter array.  One epoch = one job for both models together.
* Fractional gates (native RZZ on Heron) are auto-detected.  When they are used,
  ZZ angles are re-parameterised as pi/4*(1+sin(w)) so every angle lands in
  [0, pi/2] without any angle-folding pass; RZZ(theta) on [0, pi/2] covers all
  RZZ gates up to local rotations, which the surrounding U3 layer supplies.

Noisy simulation
----------------
`--noisy` transpiles exactly as for hardware, contracts the circuit to the 12-14
physical qubits it actually touches, and builds an Aer noise model from that
device's T1/T2, gate errors, reset error and readout asymmetry on those qubits.
The calibration source is `--fake ibm_fez` (offline) or `--calibrate-from
ibm_pittsburgh` (live calibration, execution still local, zero QPU time).
Runtime dynamical decoupling and twirling are applied server-side, so `--dd` and
`--no-twirling` have no effect in local simulation.

Usage
-----
    python hqrnn_hw_experiment.py                       # ideal Aer
    python hqrnn_hw_experiment.py --fake ibm_fez --noisy
    python hqrnn_hw_experiment.py --calibrate-from ibm_pittsburgh --noisy
    python hqrnn_hw_experiment.py --hardware --backend ibm_pittsburgh
    python hqrnn_hw_experiment.py --ablation --fake ibm_fez --noisy --epochs 100
    python hqrnn_hw_experiment.py --resume              # continue an interrupted run

Requires the repo on sys.path (hqrnn.config, hqrnn.data_handler.snp).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
import warnings
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("HQRNN_CUDA_DEVICES", "2"))
os.environ.setdefault("JAX_PLATFORMS", os.environ.get("HQRNN_JAX_PLATFORMS", ""))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import linalg, ndimage
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import ParameterVector
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "qiskit is required: pip install qiskit qiskit-ibm-runtime qiskit-aer"
    ) from exc


# =============================================================================
# 1. Configuration
# =============================================================================

@dataclass
class Cfg:
    # --- execution target -----------------------------------------------------
    hardware: bool = False
    backend: str = ""                      # explicit backend name, else preference list
    backend_prefs: tuple = ("ibm_pittsburgh", "ibm_kingston", "ibm_marrakesh",
                            "ibm_fez", "ibm_boston", "ibm_phoenix", "ibm_miami")
    fake: str = ""                         # e.g. "ibm_fez": offline calibration source
    calibrate_from: str = ""               # live backend name; calibration only, runs local
    noisy: bool = False                    # device-calibrated noise on the local Aer run
    channel: str = ""                      # QiskitRuntimeService channel override
    instance: str = ""
    token: str = ""
    exec_mode: str = "batch"               # batch (billed per job) | job | session
    opt_level: int = 2
    fractional: str = "auto"               # auto | on | off
    zz_angle: str = "auto"                 # auto | free | folded
    dd: str = "runtime"                    # runtime | off
    dd_sequence: str = "XY4"
    twirling: bool = True
    max_retries: int = 3
    max_sim_qubits: int = 24               # statevector width cap for --noisy

    # --- what to run ----------------------------------------------------------
    only: int = 0                          # 0 = both, 1 = model1 only, 3 = model3 only
    ablation: bool = False                 # run the full ten-arm sweep

    # --- ansatz ---------------------------------------------------------------
    topology: str = "hw_line"              # hw_line | hw_ladder | cross_only | ring
    line_pattern: str = "alternating"      # alternating | blocked
    entangler: str = "zz"                  # zz (IsingZZ) | xx (IsingXX)
    final_1q: bool = False                 # extra single-qubit layer after the last RZZ
    label_gate: str = "rz"                 # rz (paper) | ry (acts on |0>, not absorbed)
    depth1: int = 1
    depth3: int = 1

    # --- SPSA schedule --------------------------------------------------------
    epochs: int = 1000
    eval_every: int = 10
    seed: int = 42
    spsa_avg: int = 1                      # SPSA directions averaged per job
    split_label_grad: bool = True          # drive lab[c] with L_c instead of the total

    # --- model 1 --------------------------------------------------------------
    m1_batch: int = 16
    m1_c: float = 0.05
    m1_lr: float = 0.01
    m1_shots: int = 128
    m1_pred_shots: int = 10                # "10 times per date"
    m1_aggregate: str = "mean"             # mean | sign_vote | up_ratio (all are recorded)
    m1_record_all_rows: bool = False

    # --- model 3 --------------------------------------------------------------
    m3_c: float = 0.1
    m3_lr: float = 0.1
    m3_shots: int = 256
    m3_n_real: int = 64
    m3_gen_shots: int = 2000               # per snapshot, per digit
    m3_final_gen: int = 5000               # final generation, per digit
    m3_select: int = 0                     # 0 -> 40% of generated samples
    m3_cnn: bool = True
    m3_cnn_epochs: int = 60

    # --- MMD ------------------------------------------------------------------
    mmd_sigma1: float = 3.0
    mmd_sigma3: float = 3.0
    mmd_lambda: float = 0.5
    mmd_k: float = 15.0

    # --- optimiser (the repo pairs these with the FFF scheduler) --------------
    clip_norm: float = 0.0                 # repo uses 1e-4; 0 disables
    weight_decay: float = 0.0              # repo uses 1e-3; 0 disables

    # --- FFF scheduler --------------------------------------------------------
    fff: bool = False                      # find / fight / flee + cosine restarts
    fff_restart_period: int = 100
    fff_restart_mult: float = 2.0
    fff_restart_decay: float = 0.9
    fff_warmup_ratio: float = 0.1
    fff_lr_min_ratio: float = 1e-3         # lr floor = lr_init * this
    fff_find_fight_rate: float = 2.0       # fight LR = find LR / this
    fff_fight_decay: float = 0.005
    fff_super_boost: float = 0.5
    fff_super_thresh: float = 0.2
    fff_super_window: int = 50
    fff_super_max: int = 100
    fff_patience: int = 500
    fff_cooldown: int = 10
    fff_protect: float = 0.20
    fff_ema_beta: float = 0.95
    fff_stop_after: int = 10
    fff_best_entry: bool = True            # enter fight on a new all-time best
    fff_drop_start: float = 0.25
    fff_drop_end: float = 0.20
    fff_vol_window: int = 50
    fff_vol_start: float = 0.25
    fff_vol_end: float = 0.15
    fff_vol_patience: int = 10

    # --- noise-adaptive fight entry ------------------------------------------
    fff_adapt: bool = True
    fff_noise_window: int = 100
    fff_drop_z: float = 3.0                # sharp-drop bar >= z * noise CV
    fff_vol_kappa: float = 2.0             # volatility bar >= kappa * noise CV
    fff_best_z: float = 2.0                # a "new best" must beat it by z*sigma
    fff_adapt_every: int = 100
    fff_target_fights: float = 5.0         # wanted fight entries per 1000 epochs
    fff_gain_up: float = 1.15
    fff_gain_down: float = 0.97
    fff_gain_min: float = 0.5
    fff_gain_max: float = 20.0

    # --- io -------------------------------------------------------------------
    out: str = "runs/hw_experiment"
    resume: bool = False
    rescore: bool = False                  # rebuild metrics from saved samples
    yes: bool = False
    mp4: bool = True
    fps: int = 6


def parse_args() -> Cfg:
    d = Cfg()
    p = argparse.ArgumentParser(
        description="HQRNN model 1 + model 3, SPSA on IBM Quantum, single run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hardware", action="store_true", help="run on a real IBM QPU")
    p.add_argument("--backend", default=d.backend, help="explicit backend name")
    p.add_argument("--fake", default=d.fake,
                   help="offline fake backend (e.g. ibm_fez) for layout and calibration")
    p.add_argument("--calibrate-from", default=d.calibrate_from,
                   help="live backend to pull calibration from; execution stays local")
    p.add_argument("--noisy", action="store_true",
                   help="simulate with a noise model built from that device's calibration")
    p.add_argument("--channel", default=d.channel)
    p.add_argument("--instance", default=d.instance)
    p.add_argument("--mode", choices=["batch", "job", "session"], default=d.exec_mode,
                   help="batch/job bill only real execution time; session bills wall clock")
    p.add_argument("--opt-level", type=int, default=d.opt_level)
    p.add_argument("--fractional", choices=["auto", "on", "off"], default=d.fractional)
    p.add_argument("--zz-angle", choices=["auto", "free", "folded"], default=d.zz_angle,
                   help="folded keeps every RZZ angle in [0, pi/2] for native fractional RZZ")
    p.add_argument("--dd", choices=["runtime", "off"], default=d.dd)
    p.add_argument("--no-twirling", action="store_true")
    p.add_argument("--max-sim-qubits", type=int, default=d.max_sim_qubits,
                   help="width cap for --noisy; routing SWAPs can push ring past it")
    p.add_argument("--only", type=int, choices=[0, 1, 3], default=d.only)
    p.add_argument("--ablation", action="store_true",
                   help="run all ten ablation arms back to back and compare them")
    p.add_argument("--topology", choices=["hw_line", "hw_ladder", "cross_only", "ring"],
                   default=d.topology)
    p.add_argument("--line-pattern", choices=["alternating", "blocked"], default=d.line_pattern)
    p.add_argument("--entangler", choices=["zz", "xx"], default=d.entangler,
                   help="two-qubit generator: IsingZZ (diagonal) or IsingXX")
    p.add_argument("--label-gate", choices=["rz", "ry"], default=d.label_gate,
                   help="rz is the paper's choice, but U3.RZ(a)=U3(t,p,m+a) absorbs it "
                        "and RZ on |0> is a global phase, so row 0 can never carry the "
                        "label; ry is neither absorbed nor inert on |0>")
    p.add_argument("--final-1q", action="store_true",
                   help="append a U3 layer after the last entangler; without it a "
                        "depth-1 ansatz ends on a diagonal gate and the hidden "
                        "register cancels out of the measurement distribution")
    p.add_argument("--mmd-lambda", type=float, default=d.mmd_lambda,
                   help="weight of the Gap-Matching class-separation term (0 disables it)")
    p.add_argument("--mmd-sigma3", type=float, default=d.mmd_sigma3)
    p.add_argument("--mmd-k", type=float, default=d.mmd_k)
    p.add_argument("--depth1", type=int, default=d.depth1)
    p.add_argument("--depth3", type=int, default=d.depth3)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--eval-every", type=int, default=d.eval_every)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--no-split-label-grad", action="store_true",
                   help="drive every parameter from the smooth-max total. By default "
                        "lab[0] is driven by L0 and lab[1] by L1, which costs no extra "
                        "circuits because both terms are already measured.")
    p.add_argument("--spsa-avg", type=int, default=d.spsa_avg,
                   help="SPSA directions averaged inside ONE job. The job round trip "
                        "is ~90%% of the wall clock, so raising this buys gradient "
                        "quality almost for free in wall time.")
    p.add_argument("--m1-batch", type=int, default=d.m1_batch)
    p.add_argument("--m1-shots", type=int, default=d.m1_shots)
    p.add_argument("--m1-c", type=float, default=d.m1_c)
    p.add_argument("--m1-lr", type=float, default=d.m1_lr)
    p.add_argument("--m1-pred-shots", type=int, default=d.m1_pred_shots)
    p.add_argument("--m1-aggregate", choices=["mean", "sign_vote", "up_ratio"],
                   default=d.m1_aggregate)
    p.add_argument("--m1-record-all-rows", action="store_true")
    p.add_argument("--m3-shots", type=int, default=d.m3_shots)
    p.add_argument("--m3-c", type=float, default=d.m3_c)
    p.add_argument("--m3-lr", type=float, default=d.m3_lr)
    p.add_argument("--m3-gen-shots", type=int, default=d.m3_gen_shots)
    p.add_argument("--m3-final-gen", type=int, default=d.m3_final_gen)
    p.add_argument("--no-cnn", action="store_true")
    p.add_argument("--m3-cnn-epochs", type=int, default=d.m3_cnn_epochs)
    p.add_argument("--m3-select", type=int, default=d.m3_select,
                   help="samples kept per digit by the CNN filter (0 = 40%% of generated)")
    p.add_argument("--m3-n-real", type=int, default=d.m3_n_real)
    g = p.add_argument_group(
        "FFF scheduler",
        "Port of hqrnn/FFF_mode + hqrnn/scheduler. The LR peak is taken from "
        "--m1-lr / --m3-lr, so enabling it does not change where training starts.")
    g.add_argument("--fff", action="store_true",
                   help="enable find/fight/flee mode control + cosine warm restarts")
    g.add_argument("--clip-norm", type=float, default=d.clip_norm,
                   help="global-norm gradient clip before Adam (repo: 1e-4, 0=off)")
    g.add_argument("--weight-decay", type=float, default=d.weight_decay,
                   help="decoupled weight decay (repo: 1e-3, 0=off)")
    g.add_argument("--fff-restart-period", type=int, default=d.fff_restart_period)
    g.add_argument("--fff-restart-mult", type=float, default=d.fff_restart_mult)
    g.add_argument("--fff-restart-decay", type=float, default=d.fff_restart_decay)
    g.add_argument("--fff-warmup-ratio", type=float, default=d.fff_warmup_ratio)
    g.add_argument("--fff-lr-min-ratio", type=float, default=d.fff_lr_min_ratio,
                   help="LR floor as a fraction of the peak; the repo's 1e-8/0.1 "
                        "lets the cosine reach zero, which stalls SPSA")
    g.add_argument("--fff-find-fight-rate", type=float, default=d.fff_find_fight_rate)
    g.add_argument("--fff-fight-decay", type=float, default=d.fff_fight_decay)
    g.add_argument("--fff-super-boost", type=float, default=d.fff_super_boost)
    g.add_argument("--fff-patience", type=int, default=d.fff_patience)
    g.add_argument("--fff-cooldown", type=int, default=d.fff_cooldown)
    g.add_argument("--fff-protect", type=float, default=d.fff_protect)
    g.add_argument("--fff-ema-beta", type=float, default=d.fff_ema_beta)
    g.add_argument("--fff-stop-after", type=int, default=d.fff_stop_after)
    g.add_argument("--fff-no-best-entry", action="store_true",
                   help="do not enter fight just because a new all-time best "
                        "appeared; with a noisy loss this path fires constantly")
    g.add_argument("--fff-drop-start", type=float, default=d.fff_drop_start)
    g.add_argument("--fff-drop-end", type=float, default=d.fff_drop_end)
    g.add_argument("--fff-vol-window", type=int, default=d.fff_vol_window)
    g.add_argument("--fff-vol-start", type=float, default=d.fff_vol_start)
    g.add_argument("--fff-vol-end", type=float, default=d.fff_vol_end)
    g.add_argument("--fff-vol-patience", type=int, default=d.fff_vol_patience)
    g.add_argument("--fff-no-adapt", action="store_true",
                   help="use the repo's fixed thresholds instead of calibrating "
                        "them against the measured SPSA noise floor")
    g.add_argument("--fff-noise-window", type=int, default=d.fff_noise_window)
    g.add_argument("--fff-drop-z", type=float, default=d.fff_drop_z,
                   help="sharp-drop bar in units of the loss noise CV")
    g.add_argument("--fff-vol-kappa", type=float, default=d.fff_vol_kappa,
                   help="volatility bar in units of the loss noise CV")
    g.add_argument("--fff-best-z", type=float, default=d.fff_best_z,
                   help="a new best counts as real only if it beats the old one "
                        "by this many noise sigmas")
    g.add_argument("--fff-adapt-every", type=int, default=d.fff_adapt_every)
    g.add_argument("--fff-target-fights", type=float, default=d.fff_target_fights,
                   help="wanted fight entries per 1000 epochs; the gain on both "
                        "bars is driven towards this rate")
    p.add_argument("--out", default=d.out)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rescore", action="store_true",
                   help="recompute model 3 metrics from the .npy samples already "
                        "in --out and exit; no simulator, no QPU. Use this when "
                        "snapshots failed inside a metric but samples were saved.")
    p.add_argument("--yes", "-y", action="store_true", help="skip the cost confirmation")
    p.add_argument("--no-mp4", action="store_true")
    p.add_argument("--fps", type=int, default=d.fps)
    a = p.parse_args()

    return Cfg(
        hardware=a.hardware, backend=a.backend, fake=a.fake,
        calibrate_from=a.calibrate_from, noisy=a.noisy, channel=a.channel,
        instance=a.instance, exec_mode=a.mode, opt_level=a.opt_level,
        fractional=a.fractional, zz_angle=a.zz_angle, dd=a.dd,
        twirling=not a.no_twirling, max_sim_qubits=a.max_sim_qubits,
        only=a.only, ablation=a.ablation,
        topology=a.topology, entangler=a.entangler, final_1q=a.final_1q,
        label_gate=a.label_gate,
        mmd_lambda=a.mmd_lambda, mmd_sigma3=a.mmd_sigma3, mmd_k=a.mmd_k,
        line_pattern=a.line_pattern, depth1=a.depth1, depth3=a.depth3, epochs=a.epochs,
        eval_every=a.eval_every, seed=a.seed, spsa_avg=max(1, a.spsa_avg),
        split_label_grad=not a.no_split_label_grad,
        m1_batch=a.m1_batch, m1_c=a.m1_c,
        m1_lr=a.m1_lr, m1_shots=a.m1_shots, m1_pred_shots=a.m1_pred_shots,
        m1_aggregate=a.m1_aggregate, m1_record_all_rows=a.m1_record_all_rows,
        m3_c=a.m3_c, m3_lr=a.m3_lr, m3_shots=a.m3_shots, m3_gen_shots=a.m3_gen_shots,
        m3_final_gen=a.m3_final_gen, m3_cnn=not a.no_cnn,
        m3_cnn_epochs=a.m3_cnn_epochs, m3_select=a.m3_select, m3_n_real=a.m3_n_real,
        clip_norm=a.clip_norm, weight_decay=a.weight_decay,
        fff=a.fff, fff_restart_period=a.fff_restart_period,
        fff_restart_mult=a.fff_restart_mult, fff_restart_decay=a.fff_restart_decay,
        fff_warmup_ratio=a.fff_warmup_ratio, fff_lr_min_ratio=a.fff_lr_min_ratio,
        fff_find_fight_rate=a.fff_find_fight_rate, fff_fight_decay=a.fff_fight_decay,
        fff_super_boost=a.fff_super_boost, fff_patience=a.fff_patience,
        fff_cooldown=a.fff_cooldown, fff_protect=a.fff_protect,
        fff_ema_beta=a.fff_ema_beta, fff_stop_after=a.fff_stop_after,
        fff_best_entry=not a.fff_no_best_entry,
        fff_drop_start=a.fff_drop_start, fff_drop_end=a.fff_drop_end,
        fff_vol_window=a.fff_vol_window, fff_vol_start=a.fff_vol_start,
        fff_vol_end=a.fff_vol_end, fff_vol_patience=a.fff_vol_patience,
        fff_adapt=not a.fff_no_adapt, fff_noise_window=a.fff_noise_window,
        fff_drop_z=a.fff_drop_z, fff_vol_kappa=a.fff_vol_kappa,
        fff_best_z=a.fff_best_z, fff_adapt_every=a.fff_adapt_every,
        fff_target_fights=a.fff_target_fights,
        out=a.out, resume=a.resume, rescore=a.rescore,
        yes=a.yes, mp4=not a.no_mp4, fps=a.fps,
    )


def log(msg: str) -> None:
    tqdm.write(f"[{time.strftime('%H:%M:%S')}] {msg}")


# =============================================================================
# 2. Ansatz topology
# =============================================================================
#
# A Topology fixes (a) where each logical qubit sits along the chain of virtual
# qubits that will be pinned to physical qubits, and (b) which pairs carry an
# RZZ, grouped into layers of mutually disjoint pairs so that each layer costs
# exactly one two-qubit gate depth.
#
#   hw_line      D H D H ...  -- RZZ on every adjacent pair, even/odd brickwork.
#                Fits any coupling map that contains a path of n_D+n_H qubits,
#                so routing inserts zero SWAPs.
#   cross_only   D H D H ...  -- RZZ on the D_i-H_i matching only (rings removed
#                as requested).  Shallowest possible, but the 12/14-qubit system
#                factorises into n_D independent 2-qubit models.
#   hw_ladder    D on one rail, H on the other, every D_i-H_i coupling on a rung
#                and both intra-register chains on the rails: the paper's double
#                ring minus its two closing edges, in 3 RZZ layers.  Needs a
#                2 x n ladder in the coupling map, i.e. a square-lattice device.
#                Heavy-hex maps have no 4-cycles, so there it is unavailable.
#   ring         paper ansatz: intra-D ring + intra-H ring + cross coupling.
#                Not a subgraph of a heavy-hex map, so the router adds SWAPs.

@dataclass
class Topology:
    kind: str
    n_D: int
    n_H: int
    pos_D: list                 # logical data index  -> chain position
    pos_H: list                 # logical hidden index-> chain position
    layers: list                # [[(pos_a, pos_b, edge_id), ...], ...]
    n_edges: int

    @property
    def n_total(self) -> int:
        return self.n_D + self.n_H

    def describe(self) -> str:
        roles = ["?"] * self.n_total
        for i, p in enumerate(self.pos_D):
            roles[p] = f"D{i}"
        for i, p in enumerate(self.pos_H):
            roles[p] = f"H{i}"
        return " ".join(roles)


def _greedy_edge_layers(edges: list) -> list:
    """Pack (a, b, id) edges into layers of pairwise disjoint edges."""
    layers: list = []
    for a, b, eid in edges:
        for layer in layers:
            if all(a not in (x, y) and b not in (x, y) for x, y, _ in layer):
                layer.append((a, b, eid))
                break
        else:
            layers.append([(a, b, eid)])
    return layers


def make_topology(kind: str, n_D: int, n_H: int, line_pattern: str = "alternating") -> Topology:
    n_total = n_D + n_H

    if kind in ("hw_line", "cross_only"):
        pos_D, pos_H = [], []
        if line_pattern == "blocked":
            pos_D = list(range(n_D))
            pos_H = list(range(n_D, n_total))
        else:
            # interleave D and H as evenly as possible (identical registers -> D H D H ...)
            d = h = 0
            for p in range(n_total):
                take_d = (d * n_H <= h * n_D) if h < n_H else True
                if d >= n_D:
                    take_d = False
                if take_d:
                    pos_D.append(p)
                    d += 1
                else:
                    pos_H.append(p)
                    h += 1

        if kind == "hw_line":
            edges = [(p, p + 1, p) for p in range(n_total - 1)]
            layers = [
                [e for e in edges if e[0] % 2 == 0],
                [e for e in edges if e[0] % 2 == 1],
            ]
            layers = [ly for ly in layers if ly]
            n_edges = len(edges)
        else:  # cross_only
            edges = [(pos_D[i], pos_H[i], i) for i in range(min(n_D, n_H))]
            layers = _greedy_edge_layers(edges)
            n_edges = len(edges)
        return Topology(kind, n_D, n_H, pos_D, pos_H, layers, n_edges)

    if kind == "hw_ladder":
        if n_D != n_H:
            raise ValueError("hw_ladder needs n_D == n_H")
        pos_D = list(range(n_D))
        pos_H = list(range(n_D, n_total))
        eid = 0
        rungs, rail_a, rail_b = [], [], []
        for i in range(n_D):
            rungs.append((pos_D[i], pos_H[i], eid)); eid += 1
        for i in range(n_D - 1):
            rail_a.append((pos_D[i], pos_D[i + 1], eid)); eid += 1
        for i in range(n_H - 1):
            rail_b.append((pos_H[i], pos_H[i + 1], eid)); eid += 1
        layers = [
            rungs,
            [e for e in rail_a if e[0] % 2 == 0] + [e for e in rail_b if (e[0] - n_D) % 2 == 0],
            [e for e in rail_a if e[0] % 2 == 1] + [e for e in rail_b if (e[0] - n_D) % 2 == 1],
        ]
        return Topology(kind, n_D, n_H, pos_D, pos_H, [ly for ly in layers if ly], eid)

    if kind == "ring":
        pos_D = list(range(n_D))
        pos_H = list(range(n_D, n_total))
        edges, eid = [], 0
        for i in range(n_D):
            edges.append((pos_D[i], pos_D[(i + 1) % n_D], eid)); eid += 1
        for i in range(n_H):
            edges.append((pos_H[i], pos_H[(i + 1) % n_H], eid)); eid += 1
        for i in range(min(n_D, n_H)):
            edges.append((pos_D[i], pos_H[i], eid)); eid += 1
        return Topology(kind, n_D, n_H, pos_D, pos_H, _greedy_edge_layers(edges), eid)

    raise ValueError(f"unknown topology '{kind}'")


# =============================================================================
# 3. Parameters
# =============================================================================
#
# params = {"t1q": (depth, n_total, 3), "zz": (depth, n_edges), "lab": (n_classes, n_H)}
# The chain position, not the logical index, indexes t1q -- the ansatz is defined
# on the chain, and the D/H roles only decide encoding and measurement.

def init_params(depth: int, n_total: int, n_edges: int, n_H: int, n_classes: int,
                rng: np.random.Generator, final_1q: bool = False) -> dict:
    p = {
        "t1q": rng.standard_normal((depth, n_total, 3)) * np.sqrt(2.0 / (n_total * 3)),
        "zz": rng.standard_normal((depth, n_edges)) * np.sqrt(2.0 / depth),
        "lab": rng.standard_normal((n_classes, n_H)) * 0.1,
    }
    if final_1q:
        p["t1qf"] = rng.standard_normal((n_total, 3)) * np.sqrt(2.0 / (n_total * 3))
    return p


def n_trainable(params: dict) -> int:
    return int(sum(np.asarray(v).size for v in params.values()))


def zz_angle(w: np.ndarray, mode: str) -> np.ndarray:
    """Map free SPSA parameters to RZZ angles.

    "free"   -> identity (any angle, RZZ decomposed into 2 CZ/ECR).
    "folded" -> pi/4*(1+sin w) in [0, pi/2], the range native fractional RZZ
                accepts.  Every RZZ is locally equivalent to one with an angle in
                [0, pi/2], and the surrounding U3 layer supplies the local part,
                so this costs no expressivity while removing the angle-folding
                pass (which cannot run on unbound parameters).
    """
    if mode == "folded":
        return (np.pi / 4.0) * (1.0 + np.sin(w))
    return w


# =============================================================================
# 4. Parametric circuits
# =============================================================================

def build_circuit(topo: Topology, depth: int, seq_len: int, *,
                  use_reset: bool, use_encoding: bool, record_rows: list,
                  entangler: str = "zz", final_1q: bool = False,
                  label_gate: str = "rz"):
    """One unrolled HQRNN circuit with every angle left symbolic.

    use_reset / use_encoding = True   -> model 1 (teacher forcing on actual rates)
    use_reset / use_encoding = False  -> model 3 (autoregressive; the measured
                                         data qubit is already in |b>, which is
                                         exactly RY(pi*b)|0>, the next input)

    entangler "zz" is the paper's IsingZZ, diagonal in the measurement basis.
    "xx" is IsingXX, which on IBM hardware becomes the same RZZ or CZ pair
    conjugated by Hadamards, so it costs four extra single-qubit gates per pair.
    The IQP circuit the paper maps onto for its hardness argument is written with
    XX couplings, so this arm tests that correspondence directly.
    """
    n_D, n_H, n_total = topo.n_D, topo.n_H, topo.n_total
    record_rows = sorted(record_rows)

    pv = {
        "t1q": ParameterVector("t1q", depth * n_total * 3),
        "zz": ParameterVector("zz", depth * topo.n_edges),
        "lab": ParameterVector("lab", n_H),
        "enc": ParameterVector("enc", seq_len * n_D) if use_encoding else None,
        "t1qf": ParameterVector("t1qf", n_total * 3) if final_1q else None,
    }

    qr = QuantumRegister(n_total, "q")          # index == chain position
    cregs = {t: ClassicalRegister(n_D, f"row{t}") for t in record_rows}
    qc = QuantumCircuit(qr, *[cregs[t] for t in record_rows])

    for t in range(seq_len):
        if use_reset and t > 0:
            for i in range(n_D):
                qc.reset(qr[topo.pos_D[i]])
        lab_gate = qc.ry if label_gate == "ry" else qc.rz
        for i in range(n_H):
            lab_gate(pv["lab"][i], qr[topo.pos_H[i]])
        if use_encoding:
            for i in range(n_D):
                qc.ry(pv["enc"][t * n_D + i], qr[topo.pos_D[i]])
        for d in range(depth):
            for p in range(n_total):
                b = (d * n_total + p) * 3
                qc.u(pv["t1q"][b], pv["t1q"][b + 1], pv["t1q"][b + 2], qr[p])
            gate = qc.rxx if entangler == "xx" else qc.rzz
            for layer in topo.layers:
                for pa, pb, eid in layer:
                    gate(pv["zz"][d * topo.n_edges + eid], qr[pa], qr[pb])
        if final_1q:
            # Without this the layer ends on a diagonal RZZ.  A diagonal gate does
            # not change amplitude magnitudes, so summing the hidden register out
            # of P(data) factorises it away completely and the hidden register --
            # and with it the label -- cannot influence the measurement at all.
            for p_ in range(n_total):
                qc.u(pv["t1qf"][3 * p_], pv["t1qf"][3 * p_ + 1],
                     pv["t1qf"][3 * p_ + 2], qr[p_])
        if t in cregs:
            for i in range(n_D):
                qc.measure(qr[topo.pos_D[i]], cregs[t][i])

    return qc, pv


class Binder:
    """Turns parameter dicts into the (N, n_params) array a SamplerV2 PUB takes.

    Binding client-side is avoided entirely: the transpiled circuit is uploaded
    once and every SPSA perturbation / window / label is one row of this array.
    """

    def __init__(self, tqc: QuantumCircuit, pv: dict, zz_mode: str):
        order = list(tqc.parameters)
        col = {p: i for i, p in enumerate(order)}
        missing = [k for k, vec in pv.items()
                   if vec is not None and any(p not in col for p in vec)]
        if missing:
            raise RuntimeError(
                f"transpilation dropped parameters from {missing}; "
                f"retry with --opt-level 0"
            )
        self.n_params = len(order)
        self.idx = {k: np.array([col[p] for p in vec], dtype=np.int64)
                    for k, vec in pv.items() if vec is not None}
        self.zz_mode = zz_mode

    def values(self, params_list: list, labels: list, encodings=None) -> np.ndarray:
        """params_list[i], labels[i], encodings[i] -> row i of the PUB array."""
        n = len(params_list)
        out = np.zeros((n, self.n_params), dtype=np.float64)
        for i, prm in enumerate(params_list):
            out[i, self.idx["t1q"]] = np.asarray(prm["t1q"], dtype=np.float64).ravel()
            out[i, self.idx["zz"]] = zz_angle(
                np.asarray(prm["zz"], dtype=np.float64), self.zz_mode).ravel()
            out[i, self.idx["lab"]] = np.asarray(prm["lab"], dtype=np.float64)[labels[i]]
            if "t1qf" in self.idx:
                out[i, self.idx["t1qf"]] = np.asarray(prm["t1qf"], dtype=np.float64).ravel()
            if encodings is not None and "enc" in self.idx:
                out[i, self.idx["enc"]] = np.pi * np.asarray(
                    encodings[i], dtype=np.float64).ravel()
        return out


# =============================================================================
# 5. Backend, physical layout, execution
# =============================================================================
#
# BILLING NOTE -- why this script never opens a Session by default
# ----------------------------------------------------------------
# In Session mode the QPU is locked for you, so the reported "usage" is the wall
# clock from the first job until the session closes -- every second your Python
# process spends transpiling, binding, computing MMD or plotting is billed.  That
# is why an SPSA loop shows up as 50m39s or 78m47s.
# In Batch mode usage is the sum of the *execution* time of the jobs only; the
# classical gaps in between cost nothing, which is why the same work shows up as
# 32s or 1m20s.  This script therefore defaults to --mode batch, transpiles once
# up front, and never does classical work that a job is waiting on.

EXEC_MODES = ("batch", "job", "session")


def get_service(cfg: Cfg):
    from qiskit_ibm_runtime import QiskitRuntimeService
    kwargs = {}
    if cfg.channel:
        kwargs["channel"] = cfg.channel
    if cfg.instance:
        kwargs["instance"] = cfg.instance
    if cfg.token:
        kwargs["token"] = cfg.token
    return QiskitRuntimeService(**kwargs)


def open_backend(cfg: Cfg, min_qubits: int):
    """Returns (backend, kind) with kind in {"hardware", "calib", "aer"}.

    "calib" means the backend object is used only for its coupling map and
    calibration; execution stays on the local simulator.
    """
    if cfg.hardware:
        service = get_service(cfg)
        names = [cfg.backend] if cfg.backend else list(cfg.backend_prefs)
        want_frac = cfg.fractional in ("auto", "on")
        for name in names:
            for frac in ([True, False] if want_frac else [False]):
                try:
                    be = service.backend(name, use_fractional_gates=frac)
                except TypeError:
                    be = service.backend(name)
                except Exception:
                    continue
                try:
                    if not be.status().operational:
                        continue
                except Exception:
                    pass
                log(f"backend: {be.name} (fractional_gates={frac})")
                return be, "hardware"
        cands = service.backends(simulator=False, operational=True,
                                 min_num_qubits=min_qubits)
        if not cands:
            raise RuntimeError("no operational backend with enough qubits")
        be = min(cands, key=lambda b: b.status().pending_jobs)
        log(f"backend (fallback): {be.name}")
        return be, "hardware"

    if cfg.calibrate_from:
        service = get_service(cfg)
        be = service.backend(cfg.calibrate_from,
                             use_fractional_gates=cfg.fractional in ("auto", "on"))
        log(f"calibration source: live {be.name} (execution stays local)")
        return be, "calib"

    if cfg.fake:
        from qiskit_ibm_runtime import fake_provider
        cls_name = "Fake" + "".join(
            w.capitalize() for w in cfg.fake.replace("ibm_", "").split("_"))
        cls = getattr(fake_provider, cls_name, None)
        if cls is None:
            raise RuntimeError(
                f"no fake backend {cls_name} in qiskit_ibm_runtime.fake_provider")
        log(f"calibration source: {cls_name} (execution stays local)")
        return cls(), "calib"

    return None, "aer"


def coupling_adjacency(backend) -> dict:
    adj = defaultdict(set)
    cm = backend.coupling_map
    edges = cm.get_edges() if cm is not None else []
    for a, b in edges:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return dict(adj)


def _target_error(backend, names, key, default):
    try:
        tgt = backend.target
    except Exception:
        return default
    for name in names:
        try:
            if name not in tgt.operation_names:
                continue
            props = tgt[name]
        except Exception:
            continue
        for k in (key, tuple(reversed(key))):
            p = props.get(k)
            if p is not None and getattr(p, "error", None) is not None:
                return float(p.error)
    return default


def edge_cost(backend, a, b):
    return _target_error(backend, ("cz", "ecr", "cx", "rzz"), (a, b), 0.02)


def node_cost(backend, q):
    return _target_error(backend, ("measure",), (q,), 0.02)


def find_best_path(adj: dict, ecost, ncost, length: int, budget: int = 400000):
    """Cheapest simple path of `length` qubits in the coupling graph (DFS + pruning)."""
    best = [float("inf"), None]
    seen = [0]

    def dfs(path, visited, cost):
        if seen[0] > budget or cost >= best[0]:
            return
        seen[0] += 1
        if len(path) == length:
            best[0], best[1] = cost, list(path)
            return
        last = path[-1]
        nbrs = sorted(adj.get(last, ()), key=lambda n: ecost(last, n) + ncost(n))
        for n in nbrs:
            if n in visited:
                continue
            visited.add(n)
            path.append(n)
            dfs(path, visited, cost + ecost(last, n) + ncost(n))
            path.pop()
            visited.discard(n)

    starts = sorted(adj, key=lambda q: (ncost(q), -len(adj[q])))
    for s in starts:
        if seen[0] > budget:
            break
        dfs([s], {s}, ncost(s))
    if best[1] is None:
        raise RuntimeError(f"no path of {length} qubits found in the coupling map")
    return best[1], best[0]


def find_best_ladder(adj: dict, ecost, ncost, rungs: int, budget: int = 400000):
    """Cheapest 2 x `rungs` ladder subgraph, or None if the map has no 4-cycles.

    Heavy-hex maps (Eagle/Heron) have no 4-cycles at all, so this returns None
    there.  Square-lattice maps do contain ladders, and a ladder reproduces the
    paper's topology far better than a line: data on one rail, hidden on the
    other, every D_i-H_i cross coupling on a rung, both intra-register chains on
    the rails -- the double ring minus its two closing edges, at 3 RZZ layers.
    """
    best = [float("inf"), None]
    seen = [0]

    def extend(rail_a, rail_b, used, cost):
        if seen[0] > budget or cost >= best[0]:
            return
        seen[0] += 1
        if len(rail_a) == rungs:
            best[0], best[1] = cost, (list(rail_a), list(rail_b))
            return
        for na in sorted(adj.get(rail_a[-1], ()), key=lambda n: ecost(rail_a[-1], n)):
            if na in used:
                continue
            for nb in adj.get(rail_b[-1], ()):
                if nb in used or nb == na or nb not in adj.get(na, ()):
                    continue
                c = (cost + ecost(rail_a[-1], na) + ecost(rail_b[-1], nb)
                     + ecost(na, nb) + ncost(na) + ncost(nb))
                rail_a.append(na); rail_b.append(nb); used.update((na, nb))
                extend(rail_a, rail_b, used, c)
                rail_a.pop(); rail_b.pop(); used.discard(na); used.discard(nb)

    for a in sorted(adj, key=lambda q: ncost(q)):
        if seen[0] > budget:
            break
        for b in adj.get(a, ()):
            if b <= a:
                continue
            extend([a], [b], {a, b}, ecost(a, b) + ncost(a) + ncost(b))
    return best[1], best[0]


def describe_coupling(adj: dict) -> str:
    deg = defaultdict(int)
    for q, ns in adj.items():
        deg[len(ns)] += 1
    tri = sum(1 for a in adj for b in adj[a] for c in adj[a]
              if b < c and c in adj.get(b, ()))
    squares = 0
    for a in adj:
        for b in adj[a]:
            for c in adj[b]:
                if c == a:
                    continue
                squares += len((adj[c] & adj[a]) - {b})
    shape = "square-lattice-like" if squares else "heavy-hex-like (no 4-cycles)"
    degs = ", ".join(f"deg{k}={v}" for k, v in sorted(deg.items()))
    return f"{len(adj)} qubits, {degs}, 4-cycles={squares // 8}, triangles={tri // 3} -> {shape}"


def resolve_layout(backend, topo: Topology):
    """layout[chain position] = physical qubit, chosen to need zero SWAPs."""
    adj = coupling_adjacency(backend)
    ecost = lambda a, b: edge_cost(backend, a, b)
    ncost = lambda q: node_cost(backend, q)
    log(f"coupling map: {describe_coupling(adj)}")

    if topo.kind == "hw_ladder":
        found, cost = find_best_ladder(adj, ecost, ncost, topo.n_D)
        if found is None:
            raise RuntimeError(
                "no 2xN ladder in this coupling map (heavy-hex has no 4-cycles); "
                "use --topology hw_line"
            )
        rail_a, rail_b = found
        layout = [0] * topo.n_total
        for i in range(topo.n_D):
            layout[topo.pos_D[i]] = rail_a[i]
            layout[topo.pos_H[i]] = rail_b[i]
        log(f"ladder layout cost={cost:.4f} railD={rail_a} railH={rail_b}")
        return layout

    path, cost = find_best_path(adj, ecost, ncost, topo.n_total)
    log(f"path layout cost={cost:.4f} physical={path}")
    return list(path)


def contract_to_used(tqc: QuantumCircuit):
    """Shrink a backend-width circuit to just the physical qubits it touches.

    A 156-qubit transpiled circuit cannot be simulated, but the 12-14 qubits it
    actually uses can.  The returned circuit keeps the exact native gate sequence
    the QPU would run; only the qubit indices are renumbered.
    """
    used = sorted({tqc.find_bit(q).index for inst in tqc.data for q in inst.qubits})
    mapping = {p: i for i, p in enumerate(used)}
    qr = QuantumRegister(len(used), "q")
    small = QuantumCircuit(qr, *tqc.cregs)
    for inst in tqc.data:
        small.append(inst.operation,
                     [qr[mapping[tqc.find_bit(q).index]] for q in inst.qubits],
                     list(inst.clbits))
    small.global_phase = tqc.global_phase
    return small, used


def device_noise_model(backend, used: list):
    """Aer noise model restricted to `used` physical qubits and renumbered.

    Built from the backend target, so it carries that device's per-qubit T1/T2,
    per-gate error rates, reset error and readout asymmetry.  Errors on qubits
    outside the chosen layout are dropped rather than remapped.
    """
    from qiskit_aer.noise import NoiseModel
    from qiskit_aer.noise.device import (basic_device_gate_errors,
                                         basic_device_readout_errors)
    mapping = {p: i for i, p in enumerate(used)}
    nm = NoiseModel()
    n_gate = n_read = 0
    for name, qubits, error in basic_device_gate_errors(target=backend.target):
        if all(q in mapping for q in qubits):
            nm.add_quantum_error(error, name, [mapping[q] for q in qubits],
                                 warnings=False)
            n_gate += 1
    for qubits, error in basic_device_readout_errors(target=backend.target):
        if qubits[0] in mapping:
            nm.add_readout_error(error, [mapping[qubits[0]]], warnings=False)
            n_read += 1
    log(f"noise model: {n_gate} gate errors, {n_read} readout errors on "
        f"{len(used)} qubits")
    if n_gate == 0:
        log("WARNING: the target carried no gate errors; the run will be noiseless")
    return nm


def _params_survive(tqc: QuantumCircuit, pv: dict) -> bool:
    have = set(tqc.parameters)
    return all(all(p in have for p in vec) for vec in pv.values() if vec is not None)


def transpile_once(qc: QuantumCircuit, backend, cfg: Cfg, topo: Topology,
                   kind: str = "hardware", pv: dict | None = None):
    """Transpile the parametric circuit a single time, pinned to a chosen layout.

    Higher optimisation levels can fold a symbolic gate away when its angle
    happens to act trivially on the state the pass sees, which would silently
    drop a trainable parameter.  When that happens the level is stepped down
    until every parameter survives.

    Returns (circuit_to_execute, layout, noise_model, optimisation level used).
    """
    if backend is None:
        return qc, None, None, cfg.opt_level
    layout = resolve_layout(backend, topo)
    level = cfg.opt_level
    while True:
        pm = generate_preset_pass_manager(optimization_level=level,
                                          backend=backend, initial_layout=layout)
        tqc = pm.run(qc)
        if pv is None or level == 0 or _params_survive(tqc, pv):
            break
        level -= 1
        log(f"optimisation level {level + 1} dropped trainable parameters; "
            f"retrying at level {level}")
    ops = tqc.count_ops()
    two_q = sum(int(ops.get(g, 0)) for g in ("cz", "ecr", "cx", "rzz"))
    # A SWAP-free mapping needs exactly one native RZZ per edge per layer per step,
    # i.e. 1 gate with fractional RZZ and 2 with a CZ/ECR basis.  Anything above
    # that is routing overhead the layout failed to avoid.
    logical_ops = qc.count_ops()
    n_ent = logical_ops.get("rzz", 0) + logical_ops.get("rxx", 0)
    per_ent = 1 if int(ops.get("rzz", 0)) else 2
    expected = n_ent * per_ent
    log(f"transpiled: depth={tqc.depth()} 2q_gates={two_q} "
        f"(SWAP-free would be {expected}) ops={dict(ops)}")
    if expected and two_q > expected * 1.05:
        log(f"WARNING: {two_q - expected} extra two-qubit gates -- the router had to "
            f"add SWAPs, this topology does not embed in the coupling map")
    if kind == "hardware":
        return tqc, layout, None, level
    if not cfg.noisy:
        log("(calibration probe only: execution runs the logical circuit, noiseless)")
        return qc, layout, None, level
    small, used = contract_to_used(tqc)
    log(f"noisy simulation on {len(used)} contracted qubits (physical {used})")
    if len(used) > cfg.max_sim_qubits:
        raise RuntimeError(
            f"{len(used)} qubits after routing exceeds --max-sim-qubits "
            f"({cfg.max_sim_qubits}); raise it, use --topology hw_line, or drop --noisy")
    if len(used) > 20:
        log(f"WARNING: {len(used)}-qubit statevector, this arm will be slow")
    return small, layout, device_noise_model(backend, used), level


# --- execution ---------------------------------------------------------------

class Runner:
    """Submits PUBs and returns DataBins, tracking real QPU usage per job.

    A PUB is (circuit, parameter_values[N, P], shots).  One PUB therefore carries
    N different parameter settings -- all SPSA perturbations of an epoch, or all
    200 forecast windows -- so an epoch costs one job, not N.

    On hardware every model's PUBs go into a single job.  On the simulator each
    model is run separately so each can carry its own noise model.
    """

    def __init__(self, cfg: Cfg, backend, kind: str, exec_mode: str = "batch"):
        self.cfg = cfg
        self.backend = backend
        self.kind = kind
        self.exec_mode = exec_mode
        self._ctx = None
        self.jobs = 0
        self.shots = 0
        self.pubs = 0
        self.qpu_seconds = 0.0
        self.job_log: list = []
        # Error-suppression level that the service actually accepts.  Gate
        # twirling is rejected outright on circuits containing native fractional
        # RZZ, so the first success pins the level and later epochs start there
        # instead of burning two rejected jobs each time.
        self.supp_level = 0

    # -- context ---------------------------------------------------------------
    def __enter__(self):
        if self.kind == "hardware" and self.exec_mode in ("batch", "session"):
            self._open_ctx()
        return self

    def __exit__(self, *exc):
        self._close_ctx()
        return False

    def _open_ctx(self):
        from qiskit_ibm_runtime import Batch, Session
        cls = Batch if self.exec_mode == "batch" else Session
        self._ctx = cls(backend=self.backend)
        log(f"opened {self.exec_mode} {getattr(self._ctx, 'session_id', '')}")

    def _close_ctx(self):
        if self._ctx is not None:
            try:
                self._ctx.close()
            except Exception:
                pass
            self._ctx = None

    # -- sampler ---------------------------------------------------------------
    SUPP_LEVELS = ("full", "no_gate_twirl", "plain")

    def _sampler(self, shots_per_randomization=None, level: int = 0,
                 noise_model=None):
        if self.kind != "hardware":
            from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
            # vary the seed per job, otherwise repeated snapshots of an unchanged
            # circuit would return byte-identical shots
            opts = ({"backend_options": {"noise_model": noise_model}}
                    if noise_model is not None else None)
            return AerSamplerV2(seed=self.cfg.seed + self.jobs, options=opts)

        from qiskit_ibm_runtime import SamplerV2
        mode = self._ctx if self._ctx is not None else self.backend
        s = SamplerV2(mode=mode)
        if level >= 2:
            s.options.twirling.enable_gates = False
            s.options.twirling.enable_measure = False
            s.options.dynamical_decoupling.enable = False
            return s
        gates = self.cfg.twirling and level < 1
        s.options.twirling.enable_gates = gates
        s.options.twirling.enable_measure = self.cfg.twirling
        if self.cfg.twirling:
            s.options.twirling.num_randomizations = "auto"
            s.options.twirling.shots_per_randomization = (
                shots_per_randomization if shots_per_randomization else "auto")
        if self.cfg.dd == "runtime":
            s.options.dynamical_decoupling.enable = True
            s.options.dynamical_decoupling.sequence_type = self.cfg.dd_sequence
        return s

    # -- run -------------------------------------------------------------------
    def run(self, pubs: list, tag: str = "", shots_per_randomization=None,
            noise_model=None) -> list:
        """pubs: [(circuit, values[N, P] or None, shots)] -> [DataBin]."""
        last_exc = None
        for attempt in range(self.cfg.max_retries):
            level = min(self.supp_level + attempt, len(self.SUPP_LEVELS) - 1)
            try:
                t0 = time.time()
                sampler = self._sampler(shots_per_randomization, level=level,
                                        noise_model=noise_model)
                job = sampler.run(pubs)
                result = job.result()
                wall = time.time() - t0
                usage = self._usage(job)
                self.jobs += 1
                self.pubs += len(pubs)
                n_shots = sum(int(p[2]) * (1 if p[1] is None else len(p[1])) for p in pubs)
                self.shots += n_shots
                if usage == usage:  # not NaN
                    self.qpu_seconds += usage
                if level > self.supp_level:
                    log(f"error suppression pinned to '{self.SUPP_LEVELS[level]}' "
                        f"for the rest of the run")
                    self.supp_level = level
                self.job_log.append({
                    "tag": tag, "job_id": getattr(job, "job_id", lambda: "")(),
                    "pubs": len(pubs), "shots": n_shots,
                    "suppression": self.SUPP_LEVELS[level],
                    "wall_s": round(wall, 2),
                    "qpu_s": None if usage != usage else round(usage, 3),
                })
                return [result[i].data for i in range(len(pubs))]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log(f"job failed ({tag}, attempt {attempt + 1}, "
                    f"{self.SUPP_LEVELS[level]}): {exc}")
                # A rejected option set is not a broken batch; only recycle the
                # context when the error actually points at the session.
                text = str(exc).lower()
                if (self.exec_mode in ("batch", "session")
                        and any(w in text for w in ("session", "batch", "closed",
                                                    "expired", "timeout"))):
                    self._close_ctx()
                    self._open_ctx()
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"job '{tag}' failed after retries") from last_exc

    @staticmethod
    def _usage(job) -> float:
        """Seconds of QPU time actually billed for this job."""
        try:
            u = job.usage()
            if isinstance(u, (int, float)):
                return float(u)
        except Exception:
            pass
        try:
            m = job.metrics() or {}
            u = m.get("usage")
            if isinstance(u, dict):
                for k in ("quantum_seconds", "seconds", "executionSpans"):
                    if isinstance(u.get(k), (int, float)):
                        return float(u[k])
            if isinstance(m.get("usage_seconds"), (int, float)):
                return float(m["usage_seconds"])
        except Exception:
            pass
        return float("nan")

    def summary(self) -> dict:
        return {"jobs": self.jobs, "pubs": self.pubs, "shots": self.shots,
                "qpu_seconds": round(self.qpu_seconds, 2)}


# --- result decoding ---------------------------------------------------------

def decode_register(databin, name: str, n_bits: int) -> np.ndarray:
    """DataBin -> int array (..., n_bits) with out[..., i] = clbit i.

    Register bit i holds data qubit i, and index 0 is the most significant bit of
    the class integer, matching hqrnn's _bits_to_int / _denormalize convention.
    """
    ba = getattr(databin, name)
    arr = np.asarray(ba.array, dtype=np.uint8)
    bits = np.unpackbits(arr, axis=-1, bitorder="big")   # clbit n-1 ... clbit 0
    bits = bits[..., -n_bits:][..., ::-1]                # clbit 0 ... clbit n-1
    return bits.astype(np.int8)


def selfcheck_decoder(databin, name: str, n_bits: int) -> None:
    """Cross-check the fast decoder against Qiskit's own bitstring rendering."""
    try:
        ba = getattr(databin, name)
        strings = ba.get_bitstrings()
        if not strings:
            return
        ref = np.array([[int(c) for c in reversed(s[-n_bits:])] for s in strings[:8]],
                       dtype=np.int8)
        got = decode_register(databin, name, n_bits).reshape(-1, n_bits)[:8]
        if not np.array_equal(ref, got):
            raise RuntimeError("bit-order self-check FAILED -- decode_register is wrong")
        log(f"bit-order self-check passed on register '{name}'")
    except AttributeError:
        pass


def bits_to_class(bits: np.ndarray, n_D: int) -> np.ndarray:
    return bits.astype(np.int64) @ (2 ** np.arange(n_D - 1, -1, -1, dtype=np.int64))


# =============================================================================
# 6. SPSA
# =============================================================================

class NumpyAdam:
    """Adam with a per-step learning rate, matching the repo's optax chain.

    The repo builds optax.chain(clip_by_global_norm, scale_by_adam,
    add_decayed_weights) and then applies -lr * updates, i.e. the weight decay is
    decoupled and the clip runs on the raw gradient.  Both are off by default
    here so behaviour is unchanged unless --clip-norm / --weight-decay are given.
    """

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                 clip_norm=0.0, weight_decay=0.0):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.clip_norm = float(clip_norm)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m: dict = {}
        self.v: dict = {}

    def update(self, params: dict, grads: dict, lr: float | None = None) -> dict:
        self.t += 1
        step = float(self.lr if lr is None else lr)
        if not self.m:
            self.m = {k: np.zeros_like(np.asarray(v, dtype=np.float64))
                      for k, v in params.items()}
            self.v = {k: np.zeros_like(np.asarray(v, dtype=np.float64))
                      for k, v in params.items()}
        g_all = {k: np.asarray(grads[k], dtype=np.float64) for k in params}
        if self.clip_norm > 0.0:
            total = math.sqrt(sum(float((g * g).sum()) for g in g_all.values()))
            if total > self.clip_norm:
                scale = self.clip_norm / (total + 1e-12)
                g_all = {k: g * scale for k, g in g_all.items()}
        out = {}
        for k in params:
            g = g_all[k]
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            p = np.asarray(params[k], dtype=np.float64)
            direction = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay > 0.0:
                direction = direction + self.weight_decay * p
            out[k] = p - step * direction
        return out

    def state(self) -> dict:
        return {"t": self.t, "m": self.m, "v": self.v, "lr": self.lr}

    def load(self, st: dict) -> None:
        self.t, self.m, self.v = st["t"], st["m"], st["v"]


def sample_rademacher(params: dict, rng: np.random.Generator) -> dict:
    return {k: rng.choice(np.array([-1.0, 1.0]), size=np.asarray(v).shape)
            for k, v in params.items()}


def perturb(params: dict, delta: dict, sign: float, c: float) -> dict:
    return {k: np.asarray(params[k], dtype=np.float64) + sign * c * np.asarray(delta[k])
            for k in params}



# =============================================================================
# 6b. FFF scheduler -- find / fight / flee, with noise-adaptive fight entry
# =============================================================================
#
# A reimplementation of hqrnn/scheduler/scheduler.py and hqrnn/FFF_mode/
# controller.py without the Config / JAX coupling, plus one layer the original
# does not need.
#
# The original decides everything from the shape of the loss curve: a relative
# drop of 25%, a coefficient of variation above 0.25 over a 50-epoch window, or
# any new all-time best each promote FIND -> FIGHT.  Those constants were tuned
# against an analytic loss (PennyLane shots=None + autodiff), which is smooth.
# An SPSA loss read from finite shots carries an estimator noise that does NOT
# shrink as training proceeds, so all three tests fire on noise alone and the
# controller lives in FIGHT -- the failure this module's adaptive layer removes.
#
# Two mechanisms:
#
#   1. Calibration.  NoiseTracker measures the loss series' own noise floor and
#      every bar is expressed as a multiple of it, so the bar tracks the actual
#      measurement quality instead of being a constant.  The "new best" path is
#      calibrated too: a best only counts when it beats the old one by
#      --fff-best-z noise sigmas, which is the path that fires most often on a
#      noisy loss.
#   2. Rate control.  One gain multiplies both bars and is driven towards
#      --fff-target-fights entries per 1000 epochs -- raised fast when fights are
#      too frequent, lowered slowly when too rare.  Calibration alone cannot fix
#      a bar that is wrong in shape rather than in scale; the gain closes the
#      loop around the thing actually being complained about.


class NoiseTracker:
    """Noise floor of a loss series, estimated from its short-lag differences.

    Consecutive losses differ by the real parameter movement plus the estimator's
    own noise.  Across one epoch the movement is small next to the noise, so a
    robust scale estimator on the first differences isolates it:

        sigma = 1.4826 * median(|delta|) / sqrt(2)

    1.4826 turns a median absolute deviation into a Gaussian sigma; the sqrt(2)
    undoes the variance doubling of a difference of two draws.  Taking the median
    rather than the mean means a genuine step change contributes one large term
    instead of inflating the whole estimate.
    """

    def __init__(self, window: int):
        self.window = int(max(8, window))
        self.diffs = deque(maxlen=self.window)
        self.vals = deque(maxlen=self.window)
        self._prev = None

    def add(self, loss: float) -> None:
        if not np.isfinite(loss):
            return
        if self._prev is not None:
            self.diffs.append(abs(loss - self._prev))
        self._prev = float(loss)
        self.vals.append(float(loss))

    def ready(self) -> bool:
        return len(self.diffs) >= max(8, self.window // 4)

    def sigma(self) -> float:
        if not self.ready():
            return float("nan")
        return float(1.4826 * np.median(self.diffs) / np.sqrt(2.0))

    def cv(self) -> float:
        sd = self.sigma()
        if sd != sd or not self.vals:
            return float("nan")
        return float(sd / (abs(np.mean(self.vals)) + 1e-12))

    def state(self) -> dict:
        return {"diffs": list(self.diffs), "vals": list(self.vals), "prev": self._prev}

    def load(self, st: dict) -> None:
        self.diffs = deque(st["diffs"], maxlen=self.window)
        self.vals = deque(st["vals"], maxlen=self.window)
        self._prev = st["prev"]


@dataclass
class FffState:
    mode: str = "find"                 # find | fight | flee
    best_loss: float = float("inf")
    ema_loss: float = float("inf")
    no_improve: int = 0
    cooldown: int = 0
    flee_left: int = 0
    fight_count: int = 0
    flee_count: int = 0
    finished: bool = False
    fight_entry_lr: float = 0.0
    vol_strikes: int = 0
    is_super: bool = False
    super_len: int = 0
    fight_start: int = -1


class CosineRestarts:
    """Warm-restart cosine schedule with the fight and warm-start overrides.

    Cycle 0 lasts --fff-restart-period epochs and peaks at the model's own SPSA
    learning rate, so turning the scheduler on does not move where training
    starts.  Each restart multiplies the period by --fff-restart-mult and the
    peak by --fff-restart-decay.  FIGHT replaces the cycle with an exponential
    decay from the LR at entry; the warm start after a fight re-anneals to a peak
    reduced in proportion to how long that fight lasted.
    """

    def __init__(self, cfg: "Cfg", lr_init: float, max_epochs: int):
        self.cfg = cfg
        self.lr_init = float(lr_init)
        self.lr_min = float(lr_init) * float(cfg.fff_lr_min_ratio)
        self.max_epochs = int(max(1, max_epochs))
        self.cycles = self._build()
        self.idx = 0
        self.anchor = 0
        self.warm_active = False
        self.warm_start = -1
        self.warm_steps = 0
        self.warm_peak = None
        self.warm_decay = False
        self.fight_active = False
        self.fight_start = -1
        self.fight_entry_lr = 0.0

    def _build(self) -> list:
        cycles, made = [], 0
        period = max(1, int(self.cfg.fff_restart_period))
        peak = self.lr_init
        while made < self.max_epochs:
            steps = min(period, self.max_epochs - made)
            cycles.append({"steps": int(steps), "peak": float(peak),
                           "is_final": made + steps >= self.max_epochs})
            made += steps
            if not cycles[-1]["is_final"]:
                period = max(1, int(period * self.cfg.fff_restart_mult))
                peak *= self.cfg.fff_restart_decay
        return cycles

    def _cosine(self, t: int, cyc: dict) -> float:
        total = max(1, cyc["steps"])
        if t >= total:
            return self.lr_min
        cos = 0.5 * (1.0 + math.cos(math.pi * t / total))
        return max(self.lr_min, self.lr_min + (cyc["peak"] - self.lr_min) * cos)

    def _warmup(self, t: int, T: int, peak: float) -> float:
        if T <= 0:
            return peak
        s = min(max(t, 0), T)
        return self.lr_min + (peak - self.lr_min) * (s / T)

    def lr(self, step: int, st: FffState) -> float:
        cur = self.cycles[self.idx]
        if self.warm_decay and (step - self.anchor) >= cur["steps"]:
            self.warm_decay = False

        fighting = st.mode == "fight"
        if self.fight_active and not fighting:          # fight just ended
            self.fight_active = False
            length = max(1, step - (self.fight_start if self.fight_start >= 0 else step))
            if self.cfg.fff_warmup_ratio > 0:
                self.warm_active = True
                self.warm_start = step
                self.warm_steps = max(1, int(round(length * self.cfg.fff_warmup_ratio)))
            new_peak = max(self.lr_min,
                           cur["peak"] * self.cfg.fff_restart_decay
                           * max(0.0, 1.0 - length / self.max_epochs))
            if self.idx < len(self.cycles) - 1:
                self.idx += 1
            cur = self.cycles[self.idx]
            cur["peak"] = float(new_peak)
            self.anchor = step
            self.warm_peak = float(new_peak)
            self.warm_decay = False

        if fighting:
            if not self.fight_active:
                self.fight_active = True
                self.fight_start = step
                base = (cur["peak"] if self.warm_active
                        else self._cosine(step - self.anchor, cur))
                self.fight_entry_lr = max(float(base), float(st.fight_entry_lr or 0.0))
            prog = step - self.fight_start
            start = max(self.fight_entry_lr, self.lr_min)
            return max(self.lr_min,
                       self.lr_min + (start - self.lr_min)
                       * math.exp(-self.cfg.fff_fight_decay * prog))

        if self.warm_active:
            t = step - self.warm_start
            peak = float(self.warm_peak if self.warm_peak is not None else cur["peak"])
            if t < self.warm_steps:
                return max(self.lr_min, self._warmup(t, self.warm_steps, peak))
            self.anchor = step
            self.warm_active = False
            self.warm_peak = None
            self.warm_decay = True
            return max(self.lr_min, peak)

        t = step - self.anchor
        if t >= cur["steps"] and self.idx < len(self.cycles) - 1:
            self.idx += 1
            self.anchor = step
            cur = self.cycles[self.idx]
            t = 0
        n_warm = max(1, int(round(cur["steps"] * self.cfg.fff_warmup_ratio)))
        if t < n_warm and not self.warm_decay:
            return max(self.lr_min, self._warmup(t, n_warm, cur["peak"]))
        return self._cosine(t, cur)


class FffScheduler:
    """Mode controller + LR schedule for one model.

    Call order per epoch mirrors the repo's trainer:

        lr = sched.lr(epoch)                 # from the state as of epoch start
        params = opt.update(params, grads, lr=lr)
        stop = sched.observe(loss, epoch)    # mode update from that epoch's loss
    """

    def __init__(self, cfg: "Cfg", lr_init: float, max_epochs: int, tag: str = ""):
        self.cfg = cfg
        self.tag = f" [{tag}]" if tag else ""
        self.max_epochs = int(max(1, max_epochs))
        self.sched = CosineRestarts(cfg, lr_init, self.max_epochs)
        self.st = FffState()
        self.noise = NoiseTracker(cfg.fff_noise_window)
        self.recent = deque(maxlen=int(max(2, cfg.fff_vol_window)))
        self.hits = deque(maxlen=int(max(2, cfg.fff_super_window)))
        self.gain = 1.0
        self.adapt_anchor = 0
        self.adapt_fights = 0
        self.history: list = []
        self._lr = float(lr_init)
        log(f"FFF{self.tag}: peak LR {lr_init:g}, floor {self.sched.lr_min:g}, "
            f"{len(self.sched.cycles)} cosine cycles, "
            f"adaptive entry {'on' if cfg.fff_adapt else 'off'}")

    # -- learning rate ---------------------------------------------------------
    @property
    def finished(self) -> bool:
        return self.st.finished

    def multiplier(self) -> float:
        if self.st.mode != "fight":
            return 1.0
        base = 1.0 / max(1e-9, self.cfg.fff_find_fight_rate)
        if self.st.is_super:
            return max(base * self.cfg.fff_super_boost, 0.1)
        return base

    def lr(self, epoch: int) -> float:
        base = float(self.sched.lr(int(epoch), self.st))
        self._lr = max(base * self.multiplier(), self.sched.lr_min)
        return self._lr

    # -- adaptive thresholds ---------------------------------------------------
    def thresholds(self, epoch: int):
        """(sharp-drop bar, volatility bar, new-best margin) for this epoch."""
        c = self.cfg
        prog = min(1.0, epoch / self.max_epochs)
        drop = c.fff_drop_start - (c.fff_drop_start - c.fff_drop_end) * prog
        vol = c.fff_vol_start - (c.fff_vol_start - c.fff_vol_end) * prog
        margin = 0.0
        if c.fff_adapt:
            ncv, sd = self.noise.cv(), self.noise.sigma()
            if ncv == ncv:
                drop = self.gain * max(drop, c.fff_drop_z * ncv)
                vol = self.gain * max(vol, c.fff_vol_kappa * ncv)
            if sd == sd:
                margin = c.fff_best_z * sd
        return drop, vol, margin

    def _adapt(self, epoch: int) -> None:
        c = self.cfg
        if not c.fff_adapt or epoch - self.adapt_anchor < c.fff_adapt_every:
            return
        span = max(1, epoch - self.adapt_anchor)
        rate = (self.st.fight_count - self.adapt_fights) / span * 1000.0
        old = self.gain
        if rate > c.fff_target_fights:
            self.gain *= c.fff_gain_up               # trigger-happy: raise the bars
        elif rate < 0.5 * c.fff_target_fights:
            self.gain *= c.fff_gain_down             # too quiet: ease them back down
        self.gain = float(np.clip(self.gain, c.fff_gain_min, c.fff_gain_max))
        if abs(self.gain - old) > 1e-9:
            log(f"FFF{self.tag}: fight rate {rate:.1f}/1000 vs target "
                f"{c.fff_target_fights:.1f} -> gain {old:.2f} -> {self.gain:.2f}")
        self.adapt_anchor, self.adapt_fights = epoch, self.st.fight_count

    # -- mode transitions ------------------------------------------------------
    def _enter_fight(self, epoch: int, why: str) -> None:
        st = self.st
        st.mode = "fight"
        st.no_improve = 0
        st.cooldown = self.cfg.fff_cooldown
        st.fight_count += 1
        st.fight_entry_lr = self._lr
        st.is_super = False
        st.super_len = 0
        st.vol_strikes = 0
        st.fight_start = epoch
        self.hits.clear()
        self.recent.clear()
        log(f"FFF{self.tag}: epoch {epoch} FIND -> FIGHT ({why})")

    def _check_super(self, epoch: int) -> None:
        c, st = self.cfg, self.st
        if st.is_super:
            st.super_len += 1
            if st.super_len > c.fff_super_max:
                st.is_super = False
                st.super_len = 0
                log(f"FFF{self.tag}: super-fight ended after {c.fff_super_max} epochs")
            return
        if len(self.hits) < self.hits.maxlen:
            return
        rate = sum(self.hits) / len(self.hits)
        if rate >= c.fff_super_thresh:
            st.is_super = True
            st.super_len = 1
            log(f"FFF{self.tag}: super-fight (hit rate {rate:.2f})")

    def observe(self, loss: float, epoch: int) -> bool:
        """Feed one epoch's loss. Returns True once the run should stop."""
        c, st = self.cfg, self.st
        if st.finished:
            return True
        self.noise.add(loss)
        prev = loss if st.ema_loss == float("inf") else st.ema_loss
        rel_drop = (prev - loss) / (abs(prev) + 1e-12)
        st.ema_loss = c.fff_ema_beta * prev + (1.0 - c.fff_ema_beta) * loss
        if st.cooldown > 0:
            st.cooldown -= 1

        drop_bar, vol_bar, margin = self.thresholds(epoch)
        # a new best only counts as progress when it clears the noise floor; the
        # caller still checkpoints on the raw best, this only gates mode changes
        real_gain = bool(loss < st.best_loss - margin)
        if loss < st.best_loss:
            st.best_loss = float(loss)

        cv = float("nan")
        protecting = epoch < self.max_epochs * c.fff_protect
        if not protecting:
            if st.mode == "fight":
                st.no_improve = 0 if real_gain else st.no_improve + 1

            if st.mode == "find":
                self.recent.append(loss)
                vol_hit = False
                if len(self.recent) == self.recent.maxlen:
                    a = np.asarray(self.recent, dtype=np.float64)
                    cv = float(a.std() / (abs(a.mean()) + 1e-12))
                    st.vol_strikes = st.vol_strikes + 1 if cv > vol_bar else 0
                    vol_hit = st.vol_strikes >= c.fff_vol_patience
                if st.flee_count >= c.fff_stop_after:
                    st.finished = True
                    log(f"FFF{self.tag}: stop condition reached "
                        f"({st.flee_count} flee events)")
                elif st.cooldown == 0 and c.fff_best_entry and real_gain:
                    self._enter_fight(epoch, f"new best {st.best_loss:.6f}")
                elif st.cooldown == 0 and rel_drop >= drop_bar:
                    self._enter_fight(epoch,
                                      f"sharp drop {rel_drop:.4f} >= {drop_bar:.4f}")
                elif st.cooldown == 0 and vol_hit:
                    self._enter_fight(epoch, f"volatility CV {cv:.4f} > {vol_bar:.4f}")

            elif st.mode == "fight":
                self.hits.append(real_gain)
                self._check_super(epoch)
                if st.no_improve >= c.fff_patience and st.cooldown == 0:
                    st.mode = "flee"
                    st.cooldown = c.fff_cooldown
                    st.flee_count += 1
                    st.is_super = False
                    st.super_len = 0
                    length = max(1, epoch - (st.fight_start
                                             if st.fight_start >= 0 else epoch))
                    st.flee_left = max(1, int(np.ceil(length * c.fff_warmup_ratio)))
                    log(f"FFF{self.tag}: epoch {epoch} FIGHT -> FLEE "
                        f"(fight_len={length}, flee={st.flee_left})")

            elif st.mode == "flee":
                st.flee_left -= 1
                if st.flee_left <= 0 and st.cooldown == 0:
                    st.mode = "find"
                    st.no_improve = 0
                    st.cooldown = c.fff_cooldown
                    st.ema_loss = loss
                    st.vol_strikes = 0
                    self.recent.clear()
                    log(f"FFF{self.tag}: epoch {epoch} FLEE -> FIND")

            self._adapt(epoch)

        self.history.append({
            "epoch": epoch, "loss": loss, "lr": self._lr, "mode": st.mode,
            "super_fight": int(st.is_super), "ema": st.ema_loss,
            "rel_drop": rel_drop, "drop_bar": drop_bar,
            "cv": cv, "vol_bar": vol_bar, "best_margin": margin,
            "noise_sigma": self.noise.sigma(), "noise_cv": self.noise.cv(),
            "gain": self.gain, "fights": st.fight_count, "flees": st.flee_count,
            "no_improve": st.no_improve, "best": st.best_loss,
            "protected": int(protecting),
        })
        return st.finished

    # -- io --------------------------------------------------------------------
    def finalize(self, out_dir: Path, name: str = "fff") -> None:
        if not self.history:
            return
        df = pd.DataFrame(self.history)
        df.to_csv(out_dir / f"{name}_history.csv", index=False)
        colours = {"find": "#1976d2", "fight": "#c62828", "flee": "#f9a825"}
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ep = df["epoch"].to_numpy()
        mode = df["mode"].to_numpy()
        start = 0
        for i in range(1, len(mode) + 1):
            if i == len(mode) or mode[i] != mode[start]:
                for ax in (a1, a2):
                    ax.axvspan(ep[start], ep[i - 1] + 1,
                               color=colours.get(mode[start], "0.8"), alpha=0.12, lw=0)
                start = i
        a1.plot(ep, df["lr"], color="0.15", lw=1.2)
        a1.set_yscale("log")
        a1.set_ylabel("learning rate")
        a1.set_title("FFF schedule  (blue=find, red=fight, amber=flee)", fontsize=11)
        a1.grid(alpha=0.3)
        a2.plot(ep, df["rel_drop"], lw=0.7, color="#2e7d32", alpha=0.7,
                label="relative drop")
        a2.plot(ep, df["drop_bar"], lw=1.4, color="#2e7d32", ls="--",
                label="sharp-drop bar")
        a2.plot(ep, df["cv"], lw=0.7, color="#6a1b9a", alpha=0.7, label="window CV")
        a2.plot(ep, df["vol_bar"], lw=1.4, color="#6a1b9a", ls="--",
                label="volatility bar")
        a2.plot(ep, df["noise_cv"], lw=1.0, color="0.35", ls=":",
                label="measured noise CV")
        a2.set_xlabel("SPSA epoch")
        a2.set_ylabel("relative units")
        a2.set_ylim(0, float(np.nanpercentile(
            df[["drop_bar", "vol_bar"]].to_numpy(dtype=float), 99)) * 2.2 + 1e-6)
        a2.legend(fontsize=7, ncol=3)
        a2.grid(alpha=0.3)
        ax3 = a2.twinx()
        ax3.plot(ep, df["gain"], lw=1.2, color="#ef6c00", alpha=0.9)
        ax3.set_ylabel("adaptive gain", color="#ef6c00", fontsize=9)
        ax3.tick_params(axis="y", labelcolor="#ef6c00", labelsize=8)
        plt.tight_layout()
        fig.savefig(out_dir / f"{name}_schedule.png", dpi=125, bbox_inches="tight")
        plt.close(fig)
        log(f"FFF{self.tag}: {self.st.fight_count} fights, {self.st.flee_count} flees, "
            f"final gain {self.gain:.2f}, LR {self._lr:g}")

    def state(self) -> dict:
        return {
            "st": asdict(self.st),
            "sched": {k: getattr(self.sched, k) for k in
                      ("idx", "anchor", "warm_active", "warm_start", "warm_steps",
                       "warm_peak", "warm_decay", "fight_active", "fight_start",
                       "fight_entry_lr")},
            "cycles": self.sched.cycles, "noise": self.noise.state(),
            "recent": list(self.recent), "hits": list(self.hits), "gain": self.gain,
            "adapt_anchor": self.adapt_anchor, "adapt_fights": self.adapt_fights,
            "history": self.history, "lr": self._lr,
        }

    def load_state(self, s: dict) -> None:
        self.st = FffState(**s["st"])
        for k, v in s["sched"].items():
            setattr(self.sched, k, v)
        self.sched.cycles = s["cycles"]
        self.noise.load(s["noise"])
        self.recent = deque(s["recent"], maxlen=self.recent.maxlen)
        self.hits = deque(s["hits"], maxlen=self.hits.maxlen)
        self.gain = s["gain"]
        self.adapt_anchor, self.adapt_fights = s["adapt_anchor"], s["adapt_fights"]
        self.history = s["history"]
        self._lr = s["lr"]


def make_fff(cfg: "Cfg", lr_init: float, tag: str):
    return FffScheduler(cfg, lr_init, cfg.epochs, tag) if cfg.fff else None



# =============================================================================
# 7. MMD losses
# =============================================================================

def _rbf_mean(A: np.ndarray, B: np.ndarray, gamma: float) -> float:
    a2 = np.sum(A * A, axis=1)
    b2 = np.sum(B * B, axis=1)
    d2 = np.maximum(a2[:, None] + b2[None, :] - 2.0 * A @ B.T, 0.0)
    return float(np.exp(-gamma * d2).mean())


def mmd(P: np.ndarray, Q: np.ndarray, sigma: float) -> float:
    g = 1.0 / (2.0 * sigma ** 2)
    return _rbf_mean(P, P, g) - 2.0 * _rbf_mean(P, Q, g) + _rbf_mean(Q, Q, g)


def mmd_loss_model1(class_idx: np.ndarray, contexts: np.ndarray,
                    dist_map: np.ndarray, sigma: float) -> float:
    """class_idx (B, shots) of last-step classes vs the per-context target
    distribution -- the model-1 branch of Trainer_SPSA.mmd_value_only, with every
    hardware shot used as one sample of P."""
    B, S = class_idx.shape
    K = dist_map.shape[1]
    P = np.eye(K, dtype=np.float32)[class_idx.reshape(-1)]
    Q = np.repeat(dist_map[contexts].astype(np.float32), S, axis=0)
    return float(max(0.0, mmd(P, Q, sigma)))


def gap_matching_mmd(gen0: np.ndarray, gen1: np.ndarray,
                     real0: np.ndarray, real1: np.ndarray, target: float,
                     sigma: float, lam: float, k: float):
    """Model-3 Gap-Matching MMD.

    Returns (total, L0, L1).  The two per-class terms are what actually drives
    training; the total is only their smooth maximum, so a single curve hides
    which class the optimiser is currently fighting.
    """
    l0 = mmd(gen0, real0, sigma) + lam * (mmd(gen0, real1, sigma) - target) ** 2
    l1 = mmd(gen1, real1, sigma) + lam * (mmd(gen1, real0, sigma) - target) ** 2
    arr = np.array([k * l0, k * l1], dtype=np.float64)
    total = float((arr.max() + np.log(np.exp(arr - arr.max()).sum())) / k)
    return total, float(l0), float(l1)


# =============================================================================
# 8. Metrics
# =============================================================================

def _bin(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a) > 0.5).astype(np.float32)


def _sqrtm(a: np.ndarray) -> np.ndarray:
    """scipy.linalg.sqrtm across the removal of its `disp` keyword.

    SciPy used to return (matrix, errest) when called with disp=False and now
    takes no such argument, returning the matrix alone -- so the old call raises
    TypeError on current SciPy.  Both shapes are normalised to the matrix.
    """
    try:
        out = linalg.sqrtm(a, disp=False)
    except TypeError:
        out = linalg.sqrtm(a)
    return out[0] if isinstance(out, tuple) else out


def metric_bfd(real, gen, eps=1e-6) -> float:
    r, g = _bin(real), _bin(gen)
    m1, m2 = r.mean(0), g.mean(0)
    s1 = np.cov(r, rowvar=False) + np.eye(r.shape[1]) * eps
    s2 = np.cov(g, rowvar=False) + np.eye(g.shape[1]) * eps
    diff = m1 - m2
    cov_m = _sqrtm(s1.dot(s2))
    if np.isnan(cov_m).any():
        cov_m = _sqrtm(s1.dot(s2) + np.eye(s1.shape[0]) * eps)
    if np.iscomplexobj(cov_m):
        cov_m = cov_m.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(cov_m))


def metric_uniqueness(gen) -> float:
    packed = np.packbits(_bin(gen).astype(np.uint8), axis=1)
    return float(len({r.tobytes() for r in packed}) / len(packed) * 100)


def metric_novelty(real, gen) -> float:
    to_set = lambda a: {r.tobytes() for r in np.packbits(_bin(a).astype(np.uint8), axis=1)}
    real_s = to_set(real)
    gp = np.packbits(_bin(gen).astype(np.uint8), axis=1)
    return float(sum(1 for r in gp if r.tobytes() not in real_s) / len(gp) * 100)


def metric_recall(real, gen, k=5, n=500, rng=None) -> float:
    rng = rng or np.random.default_rng(0)
    rb, gb = _bin(real), _bin(gen)
    nr, ng = min(n, len(rb)), min(n, len(gb))
    R = rb[rng.choice(len(rb), nr, replace=False)]
    G = gb[rng.choice(len(gb), ng, replace=False)]
    rr = cdist(R, R, metric="hamming") * rb.shape[1]
    rg = cdist(R, G, metric="hamming") * rb.shape[1]
    return float(np.mean(rg.min(axis=1) <= np.sort(rr, axis=1)[:, k]) * 100)


def metric_nn_class_acc(real, real_lbl, gen0, gen1, k=5, n=1000, rng=None) -> float:
    """Conditioning accuracy with no classifier and no filtering.

    Each generated sample takes the majority class of its k nearest real images in
    Hamming distance; that is scored against the label it was generated under.
    Chance is 50%.  Unlike CNN accuracy on CNN-selected samples -- which returns
    100% for a generator with no conditioning at all -- this cannot be gamed by
    the selection step, because there is no selection step.
    """
    rng = rng or np.random.default_rng(0)
    R = _bin(real)
    lab = np.asarray(real_lbl)
    hits = []
    for gen, want in ((gen0, 0), (gen1, 1)):
        G = _bin(gen)
        if len(G) > n:
            G = G[rng.choice(len(G), n, replace=False)]
        d = cdist(G, R, metric="hamming")
        nn = np.argsort(d, axis=1)[:, :k]
        vote = (lab[nn].mean(axis=1) > 0.5).astype(int)
        hits.append(vote == want)
    return float(np.concatenate(hits).mean() * 100)


def score_samples(real_imgs, cnn, s0: np.ndarray, s1: np.ndarray,
                  prefix: str = "", seed: int = 0) -> dict:
    """Every generative metric for one pair of label-conditioned sample sets.

    Module level so `--rescore` can rebuild the table from saved .npy samples
    without constructing a Model3 (and therefore without a backend).
    """
    gen = np.concatenate([s0, s1])
    lbl = np.array([0] * len(s0) + [1] * len(s1))
    out = {}
    if cnn is not None and cnn.state is not None:
        p = cnn.prob(gen)
        key = ("CNN_Acc" if prefix == "raw_" else
               "SelfScore_NOT_ACCURACY")   # circular: these samples were
        out[prefix + key] = float(np.mean((p > 0.5).astype(int) == lbl) * 100)
        #        chosen by this same CNN, so it reads ~100% for any generator
        #        that merely produces digit-like samples, conditioned or not
    nov = metric_novelty(real_imgs, gen)
    rec = metric_recall(real_imgs, gen, rng=np.random.default_rng(seed))
    out.update({
        prefix + "BFD": metric_bfd(real_imgs, gen),
        prefix + "Uniqueness": metric_uniqueness(gen),
        prefix + "Novelty": nov,
        prefix + "Recall": rec,
        prefix + "HMean": 2 * rec * nov / (rec + nov + 1e-9),
    })
    return out


def timeseries_metrics(df: pd.DataFrame, rate_col: str, segment: str) -> dict | None:
    """Long/cash strategy metrics.

    The predicted *rate* column is used directly; deriving it from the cumulative
    predicted price curve (as an earlier version of this pipeline did) mixes in
    the drift of that curve and corrupts MAE, DA and IC.
    """
    d = df.dropna(subset=[rate_col, "Actual_Rate", "Prev_Actual"])
    if len(d) < 10:
        return None
    act = d["Actual_Rate"].to_numpy(dtype=float)     # percent
    pred = d[rate_col].to_numpy(dtype=float)         # percent
    pos = (pred > 0).astype(float)                   # hold asset, else cash
    strat = pos * (act / 100.0)
    cum = np.cumprod(1.0 + strat)
    sharpe = (strat.mean() * 252) / (strat.std() * np.sqrt(252)) if strat.std() else 0.0
    ic = float(np.corrcoef(act, pred)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    return {
        "Segment": segment,
        "Return(%)": float((cum[-1] - 1.0) * 100),
        "Sharpe": float(sharpe),
        "MDD(%)": float((cum / np.maximum.accumulate(cum) - 1.0).min() * 100),
        "MAE(Rate)": float(np.mean(np.abs(act - pred))),
        "RMSE(Rate)": float(np.sqrt(np.mean((act - pred) ** 2))),
        "DA(%)": float(np.mean(np.sign(act) == np.sign(pred)) * 100),
        "IC": ic,
        "n": int(len(d)),
    }


# =============================================================================
# 9. KDE stack + animation
# =============================================================================

def kde_stack(images: np.ndarray, size: int = 500, scale: float = 60.0,
              sigma: float = 22.5, gamma: float = 4.0) -> np.ndarray:
    """Centre-of-mass aligned KDE stack.

    Same rendering as hqrnn's visualiser, but the Gaussians are deposited into a
    histogram and blurred once instead of being evaluated per pixel per point.
    100 frames of a few thousand images stay cheap.
    """
    canvas = np.zeros((size, size), dtype=np.float32)
    c = size / 2.0
    imgs = np.asarray(images, dtype=np.float32).reshape(len(images), 7, 7)
    ys, xs = [], []
    for m2 in imgs:
        if m2.sum() == 0:
            continue
        cy, cx = ndimage.center_of_mass(m2)
        if np.isnan(cy) or np.isnan(cx):
            continue
        active = np.argwhere(m2 > 0.5)
        if len(active) == 0:
            continue
        ys.append(c + (active[:, 0] - cy) * scale)
        xs.append(c + (active[:, 1] - cx) * scale)
    if not ys:
        return canvas
    yy = np.rint(np.concatenate(ys)).astype(np.int64)
    xx = np.rint(np.concatenate(xs)).astype(np.int64)
    keep = (yy >= 0) & (yy < size) & (xx >= 0) & (xx < size)
    np.add.at(canvas, (yy[keep], xx[keep]), 1.0)
    canvas = ndimage.gaussian_filter(canvas, sigma=sigma, mode="constant")
    if canvas.max() > 0:
        canvas = np.power(canvas / canvas.max(), gamma)
    return canvas


def save_frame_pair(kde0, kde1, epoch, loss, path: Path, subtitle: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    head = f"HQRNN model 3 - epoch {epoch}"
    if loss is not None and loss == loss:
        head += f" | MMD {loss:.5f}"
    fig.suptitle(head + (f"\n{subtitle}" if subtitle else ""),
                 fontsize=12, fontweight="bold")
    for ax, kde, d in zip(axes, (kde0, kde1), (0, 1)):
        ax.imshow(kde, cmap="Greys_r", vmin=0, vmax=1)
        ax.set_title(f"Digit {d}", fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def animate(frames: list, out_base: Path, fps: int = 6, make_mp4: bool = True) -> list:
    """PNG frame paths -> GIF (always) and MP4 (when a writer is available)."""
    written = []
    if not frames:
        return written
    try:
        from PIL import Image
        imgs = [Image.open(f).convert("RGB") for f in frames]
        w = min(i.width for i in imgs)
        h = min(i.height for i in imgs)
        imgs = [i.resize((w, h)) for i in imgs]
        gif = out_base.with_suffix(".gif")
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(fps, 1)), loop=0, optimize=False)
        written.append(gif)
    except Exception as exc:  # noqa: BLE001
        log(f"GIF failed: {exc}")

    if make_mp4:
        try:
            import matplotlib.animation as manim
            from PIL import Image
            imgs = [np.asarray(Image.open(f).convert("RGB")) for f in frames]
            fig = plt.figure(figsize=(imgs[0].shape[1] / 100, imgs[0].shape[0] / 100),
                             dpi=100)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            im = ax.imshow(imgs[0])
            ani = manim.FuncAnimation(
                fig, lambda i: (im.set_data(imgs[i]),), frames=len(imgs), interval=1000 / fps)
            mp4 = out_base.with_suffix(".mp4")
            ani.save(mp4, writer=manim.FFMpegWriter(fps=fps))
            plt.close(fig)
            written.append(mp4)
        except Exception as exc:  # noqa: BLE001
            log(f"MP4 skipped ({exc.__class__.__name__}); GIF was still written")
    return written


def resolve_zz_mode(cfg: Cfg, backend) -> str:
    """"folded" whenever native fractional RZZ is in the target, so every angle
    already lands in [0, pi/2] and no angle-folding pass is needed on unbound
    parameters."""
    if cfg.zz_angle in ("free", "folded"):
        return cfg.zz_angle
    if backend is None:
        return "free"
    try:
        if "rzz" in backend.target.operation_names:
            log("native fractional RZZ available -> ZZ angles parameterised into [0, pi/2]")
            return "folded"
    except Exception:
        pass
    return "free"


# =============================================================================
# 10. Model 1 -- S&P 500, SPSA + teacher forcing
# =============================================================================

def load_snp_dates(config) -> np.ndarray:
    ds = config.dataset_cfg
    df = pd.read_csv(ds.csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    start = pd.to_datetime(f"{ds.start_year}-{ds.start_month:02d}-{ds.start_day:02d}")
    df = df[df["Date"] >= start].reset_index(drop=True)
    if ds.total_days is not None:
        df = df.head(ds.total_days + config.model_cfg.seq_len + 1)
    return df["Date"].values


class Model1:
    name = "model1"

    def __init__(self, cfg: Cfg, out_dir: Path, backend, kind: str, shared=None):
        from hqrnn.config.base import Config, ExperimentConfig
        from hqrnn.data_handler.snp import SnpDataHandler

        self.cfg = cfg
        self.dir = out_dir / "model1"
        (self.dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.dir / "predictions").mkdir(parents=True, exist_ok=True)

        config = Config(model=1)
        config.exp_cfg = ExperimentConfig(loss_function="mmd", collapse_type="hard",
                                          learning_mode="teacher_forcing")
        config.model_cfg.depth = cfg.depth1
        self.config = config
        mc = config.model_cfg
        self.n_D, self.n_H, self.seq_len, self.depth = mc.n_D, mc.n_H, mc.seq_len, mc.depth

        # `shared` lets an ablation sweep reuse the data handler across arms
        if shared is not None and "handler" in shared:
            self.handler, self.dates = shared["handler"], shared["dates"]
        else:
            self.handler = SnpDataHandler(config)
            self.dates = load_snp_dates(config)
            if shared is not None:
                shared["handler"], shared["dates"] = self.handler, self.dates

        self.X_train = np.asarray(self.handler.X_train, dtype=np.float64)
        self.X_all = np.concatenate(
            [np.asarray(self.handler.X_train), np.asarray(self.handler.X_test)], axis=0
        ).astype(np.float64)
        self.C_train = np.asarray(self.handler.context_assignments_train, dtype=np.int64)
        self.dist_map = np.asarray(self.handler.dist_map, dtype=np.float64)
        self.S = float(self.handler.S)
        self.M = 2 ** (self.n_D - 1)
        self.train_size = int(self.handler.train_size)
        log(f"model1: {len(self.X_all)} windows "
            f"({self.train_size} train / {len(self.X_all) - self.train_size} test), "
            f"{self.dist_map.shape[0]} contexts")

        self.topo = make_topology(cfg.topology, self.n_D, self.n_H, cfg.line_pattern)
        rows = list(range(self.seq_len)) if cfg.m1_record_all_rows else [self.seq_len - 1]
        self.pred_reg = f"row{self.seq_len - 1}"
        qc, pv = build_circuit(self.topo, self.depth, self.seq_len,
                               use_reset=True, use_encoding=True, record_rows=rows,
                               entangler=cfg.entangler, final_1q=cfg.final_1q,
                               label_gate=cfg.label_gate)
        self.logical_depth = qc.depth()
        self.tqc, self.layout, self.noise_model, self.used_opt_level = transpile_once(
            qc, backend, cfg, self.topo, kind, pv)
        self.zz_mode = resolve_zz_mode(cfg, backend)
        self.binder = Binder(self.tqc, pv, self.zz_mode)

        rng = np.random.default_rng(cfg.seed)
        self.params = init_params(self.depth, self.topo.n_total, self.topo.n_edges,
                                  self.n_H, 1, rng, cfg.final_1q)
        self.rng = rng
        self.opt = NumpyAdam(lr=cfg.m1_lr, clip_norm=cfg.clip_norm,
                             weight_decay=cfg.weight_decay)
        self.fff = make_fff(cfg, cfg.m1_lr, "model1")
        self.best_params = deepcopy(self.params)
        self.best_loss = float("inf")
        self.loss_history: list = []
        self.metrics_history: list = []
        self.frames: list = []
        self._pending = None
        log(f"model1: topology={self.topo.kind} [{self.topo.describe()}], "
            f"{self.topo.n_edges} RZZ edges in {len(self.topo.layers)} layers, "
            f"{n_trainable(self.params)} trainable parameters, "
            f"logical depth={self.logical_depth}")

    # -- training --------------------------------------------------------------
    def train_pubs(self, epoch: int):
        idx = self.rng.choice(len(self.X_train),
                              min(self.cfg.m1_batch, len(self.X_train)), replace=False)
        Xb, Cb = self.X_train[idx], self.C_train[idx]
        B, M = len(idx), self.cfg.spsa_avg
        deltas, rows, encs = [], [], []
        for _ in range(M):
            d = sample_rademacher(self.params, self.rng)
            deltas.append(d)
            rows += [perturb(self.params, d, +1.0, self.cfg.m1_c)] * B
            rows += [perturb(self.params, d, -1.0, self.cfg.m1_c)] * B
            encs += list(Xb) + list(Xb)
        vals = self.binder.values(rows, [0] * len(rows), encs)
        self._pending = (deltas, Cb, B)
        return [(self.tqc, vals, self.cfg.m1_shots)]

    def train_consume(self, epoch: int, datas: list):
        deltas, Cb, B = self._pending
        cls = bits_to_class(decode_register(datas[0], self.pred_reg, self.n_D), self.n_D)
        sigma = self.cfg.mmd_sigma1
        grads = {k: np.zeros_like(np.asarray(v, dtype=np.float64))
                 for k, v in self.params.items()}
        losses = []
        for m, delta in enumerate(deltas):
            o = 2 * B * m
            l_plus = mmd_loss_model1(cls[o:o + B], Cb, self.dist_map, sigma)
            l_minus = mmd_loss_model1(cls[o + B:o + 2 * B], Cb, self.dist_map, sigma)
            losses.append(0.5 * (l_plus + l_minus))
            scale = (l_plus - l_minus) / (2.0 * self.cfg.m1_c)
            for k in grads:
                grads[k] += scale * np.asarray(delta[k]) / len(deltas)
        loss = float(np.mean(losses))
        self.loss_history.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = deepcopy(self.params)
        # LR comes from the state as of epoch start, the mode update from the loss
        # this epoch produced -- the order the repo's trainer uses.
        lr = self.fff.lr(epoch) if self.fff else None
        self.params = self.opt.update(self.params, grads, lr=lr)
        if self.fff:
            self.fff.observe(loss, epoch)
        return loss

    # -- evaluation ------------------------------------------------------------
    def eval_pubs(self, epoch: int):
        n = len(self.X_all)
        vals = self.binder.values([self.params] * n, [0] * n, list(self.X_all))
        return [(self.tqc, vals, self.cfg.m1_pred_shots)]

    def eval_consume(self, epoch: int, datas: list):
        bits = decode_register(datas[0], self.pred_reg, self.n_D)   # (200, shots, n_D)
        idx = bits_to_class(bits, self.n_D).astype(np.float64)
        rates = ((idx / (self.M - 0.5)) - 1.0) * self.S              # fractions
        df = self.build_df(rates)
        csv = self.dir / "predictions" / f"predictions_epoch_{epoch:04d}.csv"
        df.to_csv(csv, index=False)
        np.save(self.dir / "predictions" / f"rates_epoch_{epoch:04d}.npy", rates)

        row = {"epoch": epoch,
               "loss": self.loss_history[-1] if self.loss_history else np.nan}
        for seg, sub in (("all", df), ("train", df[df.Split == "train"]),
                         ("test", df[df.Split == "test"])):
            m = timeseries_metrics(sub, "Pred_Rate", seg)
            if m:
                for k, v in m.items():
                    if k != "Segment":
                        row[f"{seg}_{k}"] = v
        self.metrics_history.append(row)
        frame = self.dir / "frames" / f"epoch_{epoch:04d}.png"
        self.plot_curve(df, frame, epoch)
        self.frames.append(frame)
        return row

    def build_df(self, rates: np.ndarray) -> pd.DataFrame:
        """rates: (n_windows, n_shots) sampled rates as fractions."""
        n = rates.shape[0]
        ov = np.asarray(self.handler.original_values, dtype=np.float64)
        prev_actual = ov[self.seq_len: self.seq_len + n]
        actual = ov[self.seq_len + 1: self.seq_len + n + 1]
        actual_rate = (actual - prev_actual) / prev_actual * 100.0

        agg_mean = rates.mean(axis=1) * 100.0                      # percent
        up = (rates > 0).sum(axis=1)
        down = (rates < 0).sum(axis=1)
        agg_up_ratio = up / rates.shape[1]                          # "2/10 -> 0.2 %"
        agg_sign_vote = (up - down) / rates.shape[1]
        chosen = {"mean": agg_mean, "up_ratio": agg_up_ratio,
                  "sign_vote": agg_sign_vote}[self.cfg.m1_aggregate]

        base = prev_actual[0]
        pred_value = base * np.cumprod(1.0 + chosen / 100.0)
        dates = (pd.to_datetime(self.dates[self.seq_len + 1: self.seq_len + n + 1])
                 if len(self.dates) >= self.seq_len + n + 1 else pd.NaT)
        return pd.DataFrame({
            "Day": np.arange(1, n + 1),
            "Date": dates,
            "Split": np.where(np.arange(n) < self.train_size, "train", "test"),
            "Prev_Actual": prev_actual,
            "Actual_Value": actual,
            "Actual_Rate": actual_rate,
            "HQRNN_Value": pred_value,
            "Pred_Rate": chosen,
            "Pred_Rate_mean": agg_mean,
            "Pred_Rate_up_ratio": agg_up_ratio,
            "Pred_Rate_sign_vote": agg_sign_vote,
            "Pred_Rate_std": rates.std(axis=1) * 100.0,
            "n_up": up,
            "n_down": down,
        })

    def plot_curve(self, df: pd.DataFrame, path: Path, epoch: int) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        fig.suptitle(
            f"HQRNN model 1 (S&P 500) - epoch {epoch} | "
            f"{self.cfg.m1_pred_shots} inferences/date, aggregate={self.cfg.m1_aggregate}",
            fontsize=13, fontweight="bold")
        d = df["Day"].to_numpy()
        ax1.plot(d, df["Actual_Value"], color="0.45", lw=2.2, label="Actual")
        ax1.plot(d, df["HQRNN_Value"], color="firebrick", lw=1.6, label="HQRNN")
        ax1.axvline(self.train_size, color="r", ls=":", lw=1.6, label="train/test split")
        ax1.set_ylabel("Open value"); ax1.legend(loc="upper left"); ax1.grid(alpha=0.35)
        ax2.plot(d, df["Actual_Rate"], color="0.45", lw=1.2, label="Actual rate (%)")
        ax2.plot(d, df["Pred_Rate"], color="firebrick", lw=1.1, label="Predicted rate (%)")
        ax2.axvline(self.train_size, color="r", ls=":", lw=1.6)
        ax2.axhline(0.0, color="k", lw=0.7, alpha=0.5)
        ax2.set_xlabel("Day"); ax2.set_ylabel("Daily rate (%)")
        ax2.legend(loc="upper left"); ax2.grid(alpha=0.35)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)

    # -- finalisation ----------------------------------------------------------
    def finalize(self) -> None:
        if self.fff:
            self.fff.finalize(self.dir)
        if self.loss_history:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(self.loss_history, lw=1.0, color="steelblue")
            ax.set_xlabel("SPSA epoch"); ax.set_ylabel("MMD loss")
            ax.set_title("Model 1 SPSA loss"); ax.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(self.dir / "spsa_loss_curve.png", dpi=120)
            plt.close(fig)
            pd.DataFrame({"epoch": np.arange(len(self.loss_history)),
                          "loss": self.loss_history}).to_csv(
                self.dir / "loss_history.csv", index=False)
        if self.metrics_history:
            mh = pd.DataFrame(self.metrics_history)
            mh.to_csv(self.dir / "metrics_history.csv", index=False)
            for col, lbl in (("test_DA(%)", "Directional accuracy, test (%)"),
                             ("test_MAE(Rate)", "MAE of rate, test"),
                             ("test_Return(%)", "Return, test (%)")):
                if col in mh:
                    fig, ax = plt.subplots(figsize=(9, 3.4))
                    ax.plot(mh["epoch"], mh[col], "o-", ms=3, color="darkgreen")
                    ax.set_xlabel("epoch"); ax.set_ylabel(lbl); ax.grid(alpha=0.3)
                    plt.tight_layout()
                    safe = col.replace('%', 'pct').replace('(', '_').replace(')', '')
                    fig.savefig(self.dir / f"history_{safe}.png", dpi=110)
                    plt.close(fig)
            log("model 1 final metrics (last snapshot):")
            last = mh.iloc[-1]
            for seg in ("all", "train", "test"):
                keys = [k for k in mh.columns if k.startswith(seg + "_")]
                if keys:
                    log("  " + seg + ": " + "  ".join(
                        f"{k.split('_', 1)[1]}={last[k]:.4f}" for k in keys
                        if isinstance(last[k], (int, float))))
        out = animate(self.frames, self.dir / "prediction_evolution",
                      fps=self.cfg.fps, make_mp4=self.cfg.mp4)
        for f in out:
            log(f"model 1 animation: {f}")
        with open(self.dir / "trained_params.pkl", "wb") as f:
            pickle.dump({"params": self.params, "best_params": self.best_params,
                         "best_loss": self.best_loss, "topology": self.topo.kind,
                         "zz_mode": self.zz_mode, "layout": self.layout}, f)

    def summary_row(self) -> dict:
        row = {"final_loss": self.loss_history[-1] if self.loss_history else np.nan,
               "best_loss": self.best_loss, "opt_level_used": self.used_opt_level}
        if self.fff:
            row.update({"fff_fights": self.fff.st.fight_count,
                        "fff_flees": self.fff.st.flee_count,
                        "fff_gain": round(self.fff.gain, 3),
                        "fff_final_lr": self.fff._lr,
                        "fff_noise_cv": self.fff.noise.cv()})
        h = np.asarray(self.loss_history, dtype=float)
        if len(h):
            k = max(1, len(h) // 5)
            row["loss_tail_mean"] = float(h[-k:].mean())
            row["loss_tail_std"] = float(h[-k:].std())
        if self.metrics_history:
            last = self.metrics_history[-1]
            for k in ("test_DA(%)", "test_MAE(Rate)", "test_IC", "test_Return(%)",
                      "all_DA(%)", "all_MAE(Rate)"):
                if k in last:
                    row[k] = last[k]
        return row

    # -- resume ----------------------------------------------------------------
    def state(self) -> dict:
        return {"params": self.params, "best_params": self.best_params,
                "best_loss": self.best_loss, "opt": self.opt.state(),
                "rng": self.rng.bit_generator.state,
                "loss_history": self.loss_history,
                "metrics_history": self.metrics_history,
                "fff": self.fff.state() if self.fff else None,
                "frames": [str(f) for f in self.frames]}

    def load_state(self, st: dict) -> None:
        self.params = st["params"]; self.best_params = st["best_params"]
        self.best_loss = st["best_loss"]; self.opt.load(st["opt"])
        self.rng.bit_generator.state = st["rng"]
        self.loss_history = st["loss_history"]; self.metrics_history = st["metrics_history"]
        if self.fff and st.get("fff"):
            self.fff.load_state(st["fff"])
        self.frames = [Path(f) for f in st["frames"]]


# =============================================================================
# 11. Model 3 -- 7x7 digit generation, SPSA + Gap-Matching MMD
# =============================================================================

def load_digit_images(csv_path: str, d0: int, d1: int):
    df = pd.read_csv(csv_path)
    df = df[df["label"].isin([d0, d1])]
    imgs = (df.iloc[:, 1:50].to_numpy(dtype=float) > 0.5).astype(np.float32)
    lbls = (df["label"].to_numpy() == d1).astype(np.int32)
    return imgs, lbls


class CnnFilter:
    """Small classical CNN used only to rank generated samples (and report
    CNN accuracy).  Entirely optional: --no-cnn drops the jax dependency."""

    _probe: tuple | None = None

    def __init__(self, seed: int):
        self.state = None
        self.seed = seed

    @classmethod
    def deps_ok(cls) -> tuple:
        """Try the jax/flax/optax import in a child process first.

        A jax/jaxlib version mismatch, a clashing native dependency (tensorstore
        via orbax) or a CUDA plugin can abort the interpreter during import --
        glibc prints something like "free(): double free detected in tcache" and
        the process dies.  try/except cannot catch that, so an optional component
        would kill a multi-hour run at startup.  Probing in a subprocess keeps the
        crash over there and costs one interpreter start.
        """
        if cls._probe is None:
            code = ("import jax, jax.numpy, optax\n"
                    "from flax import linen\n"
                    "from flax.training import train_state\n")
            try:
                r = subprocess.run([sys.executable, "-c", code],
                                   capture_output=True, timeout=300)
            except Exception as exc:  # noqa: BLE001
                cls._probe = (False, f"import probe could not run ({exc})")
                return cls._probe
            if r.returncode == 0:
                cls._probe = (True, "")
            else:
                lines = (r.stderr or b"").decode("utf-8", "replace").strip().splitlines()
                why = lines[-1] if lines else f"exit code {r.returncode}"
                if r.returncode < 0:
                    why = (f"the import CRASHED the interpreter "
                           f"(signal {-r.returncode}): {why}")
                cls._probe = (False, why)
        return cls._probe

    def fit(self, imgs: np.ndarray, lbls: np.ndarray, epochs: int) -> bool:
        ok, why = self.deps_ok()
        if not ok:
            log(f"CNN disabled ({why})")
            log("    the CNN only ranks generated samples for the cnn_* columns; "
                "NN_Cond_Acc and every raw_* metric are computed without it. "
                "Pass --no-cnn to skip this probe entirely.")
            return False
        try:
            import jax
            import jax.numpy as jnp
            import optax
            from flax import linen as nn
            from flax.training import train_state as flax_train_state
            from jax import random as jrandom
        except Exception as exc:  # noqa: BLE001
            log(f"CNN disabled ({exc.__class__.__name__}: {exc})")
            return False

        class CNN7x7(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = jnp.pad(x, ((0, 0), (0, 1), (0, 1), (0, 0)))
                x = nn.relu(nn.Conv(32, (3, 3), padding="SAME")(x))
                x = nn.avg_pool(x, (2, 2), (2, 2))
                x = nn.relu(nn.Conv(64, (3, 3), padding="SAME")(x))
                x = nn.avg_pool(x, (2, 2), (2, 2))
                x = nn.relu(nn.Dense(128)(x.reshape((x.shape[0], -1))))
                return nn.Dense(1)(x)

        model = CNN7x7()
        key = jrandom.PRNGKey(self.seed)
        params = model.init(key, jnp.ones((1, 7, 7, 1)))
        st = flax_train_state.TrainState.create(apply_fn=model.apply, params=params,
                                                tx=optax.adam(1e-3))
        X = jnp.asarray(imgs.reshape(-1, 7, 7, 1))
        Y = jnp.asarray(lbls)

        @jax.jit
        def step(s, xb, yb):
            def lf(p):
                logits = s.apply_fn(p, xb).squeeze(-1)
                return optax.sigmoid_binary_cross_entropy(
                    logits, yb.astype(jnp.float32)).mean()
            loss, g = jax.value_and_grad(lf)(s.params)
            return s.apply_gradients(grads=g), loss

        for _ in tqdm(range(epochs), desc="CNN", leave=False):
            key, sk = jrandom.split(key)
            perm = jrandom.permutation(sk, len(X))
            for i in range(0, len(X), 32):
                sel = perm[i:i + 32]
                st, _ = step(st, X[sel], Y[sel])
        self.state = st
        self._jax = jax
        self._jnp = jnp
        log("CNN discriminator trained")
        return True

    def prob(self, imgs: np.ndarray) -> np.ndarray:
        if self.state is None:
            return np.full(len(imgs), 0.5)
        x = self._jnp.asarray(np.asarray(imgs).reshape(-1, 7, 7, 1))
        return np.asarray(self._jax.nn.sigmoid(
            self.state.apply_fn(self.state.params, x).squeeze(-1)))

    def select(self, imgs: np.ndarray, digit: int, n: int) -> np.ndarray:
        if self.state is None or n >= len(imgs):
            return imgs
        p = self.prob(imgs)
        score = p if digit == 1 else 1.0 - p
        return imgs[np.argsort(score)[::-1][:n]]


class Model3:
    name = "model3"

    def __init__(self, cfg: Cfg, out_dir: Path, backend, kind: str, shared=None):
        from hqrnn.config.base import Config

        self.cfg = cfg
        self.dir = out_dir / "model3"
        for sub in ("frames_raw", "frames_cnn", "samples"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)

        config = Config(model=3)
        config.model_cfg.depth = cfg.depth3
        self.config = config
        mc = config.model_cfg
        self.n_D, self.n_H, self.seq_len, self.depth = mc.n_D, mc.n_H, mc.seq_len, mc.depth

        ds = config.dataset_cfg
        if shared is not None and "digits" in shared:
            self.real_imgs, self.real_lbls = shared["digits"]
        else:
            self.real_imgs, self.real_lbls = load_digit_images(
                ds.csv_path, ds.first_digit, ds.second_digit)
            if shared is not None:
                shared["digits"] = (self.real_imgs, self.real_lbls)
        self.real0 = self.real_imgs[self.real_lbls == 0]
        self.real1 = self.real_imgs[self.real_lbls == 1]
        log(f"model3: {len(self.real0)} images of digit {ds.first_digit}, "
            f"{len(self.real1)} of digit {ds.second_digit}")

        rng = np.random.default_rng(cfg.seed)
        i0 = rng.choice(len(self.real0), min(cfg.m3_n_real, len(self.real0)), replace=False)
        i1 = rng.choice(len(self.real1), min(cfg.m3_n_real, len(self.real1)), replace=False)
        self.R0 = self.real0[i0]
        self.R1 = self.real1[i1]
        self.target_mmd = mmd(self.R0, self.R1, cfg.mmd_sigma3)
        log(f"model3: target MMD(R0, R1) = {self.target_mmd:.6f}")

        if shared is not None and "cnn" in shared:
            self.cnn = shared["cnn"]
        else:
            self.cnn = CnnFilter(cfg.seed)
            if cfg.m3_cnn:
                self.cnn.fit(self.real_imgs, self.real_lbls, cfg.m3_cnn_epochs)
            if shared is not None:
                shared["cnn"] = self.cnn

        self.topo = make_topology(cfg.topology, self.n_D, self.n_H, cfg.line_pattern)
        qc, pv = build_circuit(self.topo, self.depth, self.seq_len,
                               use_reset=False, use_encoding=False,
                               record_rows=list(range(self.seq_len)),
                               entangler=cfg.entangler, final_1q=cfg.final_1q,
                               label_gate=cfg.label_gate)
        self.logical_depth = qc.depth()
        self.tqc, self.layout, self.noise_model, self.used_opt_level = transpile_once(
            qc, backend, cfg, self.topo, kind, pv)
        self.zz_mode = resolve_zz_mode(cfg, backend)
        self.binder = Binder(self.tqc, pv, self.zz_mode)

        self.params = init_params(self.depth, self.topo.n_total, self.topo.n_edges,
                                  self.n_H, 2, rng, cfg.final_1q)
        self.rng = rng
        self.opt = NumpyAdam(lr=cfg.m3_lr, clip_norm=cfg.clip_norm,
                             weight_decay=cfg.weight_decay)
        self.fff = make_fff(cfg, cfg.m3_lr, "model3")
        self.best_params = deepcopy(self.params)
        self.best_loss = float("inf")
        self.loss_history: list = []
        self.l0_history: list = []
        self.l1_history: list = []
        self.metrics_history: list = []
        self.frames_raw: list = []
        self.frames_cnn: list = []
        self._pending = None
        log(f"model3: label gradient = "
            f"{'split (L0 -> lab[0], L1 -> lab[1])' if cfg.split_label_grad else 'shared total'}"
            f", {n_trainable(self.params) - self.params['lab'].size} shared parameters "
            f"vs {self.params['lab'].size} class-specific")
        log(f"model3: topology={self.topo.kind} [{self.topo.describe()}], "
            f"{self.topo.n_edges} RZZ edges in {len(self.topo.layers)} layers, "
            f"{n_trainable(self.params)} trainable parameters, "
            f"logical depth={self.logical_depth}")

    # -- helpers ---------------------------------------------------------------
    def _images(self, databin, index) -> np.ndarray:
        """DataBin row -> (shots, 49) binary images."""
        rows = [decode_register(databin, f"row{t}", self.n_D)[index]
                for t in range(self.seq_len)]
        return np.stack(rows, axis=1).reshape(rows[0].shape[0], -1).astype(np.float32)

    # -- training --------------------------------------------------------------
    def train_pubs(self, epoch: int):
        # One job carries every direction: 4 parameter rows per direction, laid out
        # as (plus/0, plus/1, minus/0, minus/1) repeated M times.
        M = self.cfg.spsa_avg
        deltas, rows, labels = [], [], []
        for _ in range(M):
            d = sample_rademacher(self.params, self.rng)
            deltas.append(d)
            pp = perturb(self.params, d, +1.0, self.cfg.m3_c)
            pm = perturb(self.params, d, -1.0, self.cfg.m3_c)
            rows += [pp, pp, pm, pm]
            labels += [0, 1, 0, 1]
        self._pending = deltas
        return [(self.tqc, self.binder.values(rows, labels), self.cfg.m3_shots)]

    def train_consume(self, epoch: int, datas: list):
        deltas = self._pending
        db, c = datas[0], self.cfg
        grads = {k: np.zeros_like(np.asarray(v, dtype=np.float64))
                 for k, v in self.params.items()}
        losses, l0s, l1s = [], [], []
        for m, delta in enumerate(deltas):
            b = 4 * m
            l_plus, l0p, l1p = gap_matching_mmd(
                self._images(db, b), self._images(db, b + 1), self.R0, self.R1,
                self.target_mmd, c.mmd_sigma3, c.mmd_lambda, c.mmd_k)
            l_minus, l0m, l1m = gap_matching_mmd(
                self._images(db, b + 2), self._images(db, b + 3), self.R0, self.R1,
                self.target_mmd, c.mmd_sigma3, c.mmd_lambda, c.mmd_k)
            losses.append(0.5 * (l_plus + l_minus))
            l0s.append(0.5 * (l0p + l0m)); l1s.append(0.5 * (l1p + l1m))
            scale = (l_plus - l_minus) / (2.0 * c.m3_c)
            for k in grads:
                if k == "lab" and c.split_label_grad:
                    # lab[0] only ever reaches the loss through L0 and lab[1] through
                    # L1, so the smooth-max total feeds each of them the other class's
                    # gradient as pure noise, and suppresses whichever class is
                    # currently the easier one.  Both terms are already measured.
                    s0 = (l0p - l0m) / (2.0 * c.m3_c)
                    s1 = (l1p - l1m) / (2.0 * c.m3_c)
                    d_lab = np.asarray(delta[k])
                    grads[k][0] += s0 * d_lab[0] / len(deltas)
                    grads[k][1] += s1 * d_lab[1] / len(deltas)
                else:
                    grads[k] += scale * np.asarray(delta[k]) / len(deltas)
        loss = float(np.mean(losses))
        self.loss_history.append(loss)
        self.l0_history.append(float(np.mean(l0s)))
        self.l1_history.append(float(np.mean(l1s)))
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = deepcopy(self.params)
        lr = self.fff.lr(epoch) if self.fff else None
        self.params = self.opt.update(self.params, grads, lr=lr)
        if self.fff:
            self.fff.observe(loss, epoch)
        return loss

    # -- snapshot --------------------------------------------------------------
    def eval_pubs(self, epoch: int, shots: int | None = None):
        vals = self.binder.values([self.params, self.params], [0, 1])
        return [(self.tqc, vals, shots or self.cfg.m3_gen_shots)]

    def eval_consume(self, epoch: int, datas: list, final: bool = False):
        db = datas[0]
        gen0, gen1 = self._images(db, 0), self._images(db, 1)
        np.save(self.dir / "samples" / f"gen0_epoch_{epoch:04d}.npy", gen0.astype(np.int8))
        np.save(self.dir / "samples" / f"gen1_epoch_{epoch:04d}.npy", gen1.astype(np.int8))

        n_sel = self.cfg.m3_select or max(1, int(0.4 * len(gen0)))
        sel0 = self.cnn.select(gen0, 0, n_sel)
        sel1 = self.cnn.select(gen1, 1, n_sel)
        loss = self.loss_history[-1] if self.loss_history else float("nan")

        f_raw = self.dir / "frames_raw" / f"epoch_{epoch:04d}.png"
        save_frame_pair(kde_stack(gen0), kde_stack(gen1), epoch, loss, f_raw,
                        subtitle=f"raw samples ({len(gen0)} shots per digit)")
        self.frames_raw.append(f_raw)

        f_cnn = self.dir / "frames_cnn" / f"epoch_{epoch:04d}.png"
        save_frame_pair(kde_stack(sel0), kde_stack(sel1), epoch, loss, f_cnn,
                        subtitle=f"CNN-filtered top {n_sel} per digit")
        self.frames_cnn.append(f_cnn)

        row = {"epoch": epoch, "loss": loss}
        row.update(self.score(sel0, sel1, prefix="cnn_"))
        row.update(self.score(gen0, gen1, prefix="raw_"))
        # the only conditioning number in this table that no classifier or
        # selection step can inflate
        row["NN_Cond_Acc"] = metric_nn_class_acc(
            self.real_imgs, self.real_lbls, gen0, gen1,
            rng=np.random.default_rng(self.cfg.seed))
        self.metrics_history.append(row)
        if final:
            self.plot_final(gen0, gen1, sel0, sel1, epoch)
        return row

    def score(self, s0: np.ndarray, s1: np.ndarray, prefix: str = "") -> dict:
        return score_samples(self.real_imgs, self.cnn, s0, s1, prefix, self.cfg.seed)

    def plot_final(self, gen0, gen1, sel0, sel1, epoch: int) -> None:
        fig = plt.figure(figsize=(16, 11))
        fig.suptitle(f"HQRNN model 3 - epoch {epoch} "
                     f"[topology={self.topo.kind}, depth={self.depth}]",
                     fontsize=13, fontweight="bold")
        for col, (imgs, d) in enumerate(((sel0, 0), (sel1, 1))):
            ax = fig.add_subplot(3, 2, 1 + col)
            ax.imshow(kde_stack(imgs), cmap="Greys_r"); ax.axis("off")
            ax.set_title(f"KDE density - digit {d} (CNN filtered)")
        for r, (pair, label) in enumerate((((sel0, sel1), "CNN filtered"),
                                           ((gen0, gen1), "Raw samples"))):
            for col, imgs in enumerate(pair):
                ax = fig.add_subplot(3, 2, 3 + 2 * r + col)
                ax.axis("off")
                ax.set_title(f"{label} - digit {col}")
                n_show = min(8, len(imgs))
                for k in range(n_show):
                    sub = ax.inset_axes([k / n_show, 0, 1 / n_show, 1])
                    sub.imshow(imgs[k].reshape(7, 7), cmap="binary", vmin=0, vmax=1)
                    sub.axis("off")
        plt.tight_layout()
        fig.savefig(self.dir / "results.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # -- finalisation ----------------------------------------------------------
    def finalize(self) -> None:
        if self.fff:
            self.fff.finalize(self.dir)
        if self.loss_history:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(self.loss_history, lw=1.0, color="steelblue")
            if self.l0_history:
                ax.plot(self.l0_history, lw=0.9, color="#2e7d32", alpha=0.8,
                        label="L0  (digit 0 term)")
                ax.plot(self.l1_history, lw=0.9, color="#c62828", alpha=0.8,
                        label="L1  (digit 1 term)")
                ax.legend(fontsize=8)
            ax.set_xlabel("SPSA epoch"); ax.set_ylabel("Gap-Matching MMD")
            ax.set_title("Model 3 SPSA loss"); ax.grid(alpha=0.3)
            plt.tight_layout()
            fig.savefig(self.dir / "spsa_loss_curve.png", dpi=120)
            plt.close(fig)
            cols = {"epoch": np.arange(len(self.loss_history)), "loss": self.loss_history}
            if self.l0_history:
                cols["L0"] = self.l0_history
                cols["L1"] = self.l1_history
            pd.DataFrame(cols).to_csv(self.dir / "loss_history.csv", index=False)
        if self.metrics_history:
            mh = pd.DataFrame(self.metrics_history)
            mh.to_csv(self.dir / "metrics_history.csv", index=False)
            log("model 3 final metrics (last snapshot):")
            last = mh.iloc[-1]
            log("  " + "  ".join(f"{k}={last[k]:.4f}" for k in mh.columns
                                 if k != "epoch" and isinstance(last[k], (int, float))))
        for frames, base in ((self.frames_raw, self.dir / "kde_evolution_raw"),
                             (self.frames_cnn, self.dir / "kde_evolution_cnn")):
            for f in animate(frames, base, fps=self.cfg.fps, make_mp4=self.cfg.mp4):
                log(f"model 3 animation: {f}")
        with open(self.dir / "trained_params.pkl", "wb") as f:
            pickle.dump({"params": self.params, "best_params": self.best_params,
                         "best_loss": self.best_loss, "topology": self.topo.kind,
                         "zz_mode": self.zz_mode, "layout": self.layout}, f)

    def summary_row(self) -> dict:
        row = {"final_loss": self.loss_history[-1] if self.loss_history else np.nan,
               "best_loss": self.best_loss, "opt_level_used": self.used_opt_level}
        if self.fff:
            row.update({"fff_fights": self.fff.st.fight_count,
                        "fff_flees": self.fff.st.flee_count,
                        "fff_gain": round(self.fff.gain, 3),
                        "fff_final_lr": self.fff._lr,
                        "fff_noise_cv": self.fff.noise.cv()})
        h = np.asarray(self.loss_history, dtype=float)
        if len(h):
            k = max(1, len(h) // 5)
            row["loss_tail_mean"] = float(h[-k:].mean())
            row["loss_tail_std"] = float(h[-k:].std())
        if self.metrics_history:
            # every metric, raw and CNN-filtered.  raw_ is the honest one:
            # cnn_CNN_Acc is circular, since the samples were selected by that
            # same CNN, so it is near 100% by construction.
            for k, v in self.metrics_history[-1].items():
                if k.startswith(("raw_", "cnn_")) or k == "NN_Cond_Acc":
                    row[k] = v
        return row

    def state(self) -> dict:
        return {"params": self.params, "best_params": self.best_params,
                "best_loss": self.best_loss, "opt": self.opt.state(),
                "rng": self.rng.bit_generator.state,
                "loss_history": self.loss_history,
                "l0_history": self.l0_history, "l1_history": self.l1_history,
                "metrics_history": self.metrics_history,
                "frames_raw": [str(f) for f in self.frames_raw],
                "fff": self.fff.state() if self.fff else None,
                "frames_cnn": [str(f) for f in self.frames_cnn]}

    def load_state(self, st: dict) -> None:
        self.params = st["params"]; self.best_params = st["best_params"]
        self.best_loss = st["best_loss"]; self.opt.load(st["opt"])
        self.rng.bit_generator.state = st["rng"]
        self.loss_history = st["loss_history"]; self.metrics_history = st["metrics_history"]
        self.l0_history = st.get("l0_history", []); self.l1_history = st.get("l1_history", [])
        self.frames_raw = [Path(f) for f in st["frames_raw"]]
        if self.fff and st.get("fff"):
            self.fff.load_state(st["fff"])
        self.frames_cnn = [Path(f) for f in st["frames_cnn"]]


# =============================================================================
# 12. Orchestration
# =============================================================================

# One-factor-at-a-time sweep.  Arm 0 is the configuration expected to perform
# best; every other arm changes exactly one thing.  Arms 5 and 6 are split so the
# noise benefit of native fractional RZZ is separated from the effect of
# restricting the ZZ angles to [0, pi/2].
ABLATION_ARMS = [
    ("00_base_lam0.5_d1_zz", "baseline: lambda 0.5, depth 1, IsingZZ", {}),
    # --- Gap-Matching weight: how hard the loss pushes the two classes apart --
    ("01_lambda_0.0", "class-separation term removed", {"mmd_lambda": 0.0}),
    ("02_lambda_1.0", "class separation x2",           {"mmd_lambda": 1.0}),
    ("03_lambda_1.5", "class separation x3",           {"mmd_lambda": 1.5}),
    ("04_lambda_2.0", "class separation x4",           {"mmd_lambda": 2.0}),
    # --- ansatz depth: expressivity against accumulated hardware error --------
    ("05_depth_2", "ansatz depth 2", {"depth3": 2, "depth1": 2}),
    ("06_depth_3", "ansatz depth 3", {"depth3": 3, "depth1": 3}),
    ("07_depth_4", "ansatz depth 4", {"depth3": 4, "depth1": 4}),
    # --- two-qubit generator --------------------------------------------------
    ("08_entangler_xx", "IsingXX instead of IsingZZ", {"entangler": "xx"}),
]


def estimate(cfg: Cfg, models: list) -> dict:
    n_evals = cfg.epochs // cfg.eval_every
    shots = 0
    detail = {}
    for m in models:
        if isinstance(m, Model1):
            tr = (cfg.epochs * 2 * cfg.spsa_avg
                  * min(cfg.m1_batch, len(m.X_train)) * cfg.m1_shots)
            ev = n_evals * len(m.X_all) * cfg.m1_pred_shots
            detail["model1_train_shots"] = tr
            detail["model1_eval_shots"] = ev
            shots += tr + ev
        else:
            tr = cfg.epochs * 4 * cfg.spsa_avg * cfg.m3_shots
            ev = n_evals * 2 * cfg.m3_gen_shots + 2 * cfg.m3_final_gen
            detail["model3_train_shots"] = tr
            detail["model3_gen_shots"] = ev
            shots += tr + ev
    jobs = cfg.epochs + n_evals * len(models) + (1 if any(
        isinstance(m, Model3) for m in models) else 0)
    per_shot_us = max((m.seq_len * 4.0 + 20.0) for m in models)   # crude
    detail.update({
        "total_shots": shots, "jobs": jobs, "snapshots": n_evals,
        "rough_qpu_minutes": round(shots * per_shot_us * 1e-6 / 60.0, 1),
    })
    return detail


def confirm(cfg: Cfg, est: dict, kind: str) -> bool:
    target = ("IBM QPU" if kind == "hardware"
              else ("Aer + device noise" if cfg.noisy else "Aer (noiseless)"))
    print("\n" + "=" * 74)
    print(f"  Execution target : {target}   mode={cfg.exec_mode}")
    print(f"  Epochs           : {cfg.epochs} (snapshot every {cfg.eval_every})")
    for k, v in est.items():
        print(f"  {k:<18}: {v:,}" if isinstance(v, int) else f"  {k:<18}: {v}")
    if kind == "hardware" and cfg.exec_mode == "session":
        print("  WARNING: session mode bills wall-clock time, including the classical")
        print("           gaps between jobs. Use --mode batch for per-job billing.")
    print("=" * 74)
    if cfg.yes or kind != "hardware":
        return True
    try:
        return input("proceed? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def build_models(cfg: Cfg, out: Path, backend, kind: str, shared=None) -> list:
    want = [1, 3] if cfg.only == 0 else [cfg.only]
    models: list = []
    if 1 in want:
        models.append(Model1(cfg, out, backend, kind, shared))
    if 3 in want:
        models.append(Model3(cfg, out, backend, kind, shared))
    return models


def run_once(cfg: Cfg, out: Path, backend, kind: str, shared=None,
             quiet: bool = False) -> dict:
    """Train and evaluate every requested model for cfg.epochs, then finalise."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    models = build_models(cfg, out, backend, kind, shared)
    est = estimate(cfg, models)
    if not quiet and not confirm(cfg, est, kind):
        log("aborted before submitting anything")
        return {}

    state_path = out / "state.pkl"
    start_epoch = 0
    if cfg.resume and state_path.exists():
        with open(state_path, "rb") as f:
            st = pickle.load(f)
        start_epoch = int(st["epoch"]) + 1
        for m in models:
            if m.name in st:
                m.load_state(st[m.name])
        log(f"resumed at epoch {start_epoch}")

    if kind == "hardware" and cfg.twirling:
        try:
            if any("rzz" in m.tqc.count_ops() for m in models):
                log("native fractional RZZ is in the circuit; gate twirling is not "
                    "supported with it, so only measurement twirling is enabled")
        except Exception:
            pass

    # On hardware every model's PUBs ride in one job.  On the simulator each model
    # is run separately so each can carry its own contracted circuit and noise
    # model; there is no per-job cost there.
    one_job = kind == "hardware"
    checked = False
    snap_fail: dict = {}
    start_level = 0
    if kind == "hardware" and cfg.twirling:
        try:
            if any("rzz" in m.tqc.count_ops() for m in models):
                start_level = 1
        except Exception:
            pass
    t_start = time.time()

    with Runner(cfg, backend, kind, cfg.exec_mode) as runner:
        runner.supp_level = start_level
        pbar = tqdm(range(start_epoch, cfg.epochs), desc=out.name, initial=start_epoch,
                    total=cfg.epochs)
        for epoch in pbar:
            losses = {}
            try:
                if one_job:
                    pubs, spans = [], []
                    for m in models:
                        p = m.train_pubs(epoch)
                        spans.append((len(pubs), len(p)))
                        pubs.extend(p)
                    datas = runner.run(pubs, tag=f"train_e{epoch}")
                    if not checked:
                        m0 = models[0]
                        reg = m0.pred_reg if isinstance(m0, Model1) else "row0"
                        selfcheck_decoder(datas[0], reg, m0.n_D)
                        checked = True
                    for m, (off, n) in zip(models, spans):
                        losses[m.name] = m.train_consume(epoch, datas[off:off + n])
                else:
                    for m in models:
                        d = runner.run(m.train_pubs(epoch), tag=f"train_{m.name}_e{epoch}",
                                       noise_model=m.noise_model)
                        if not checked:
                            reg = m.pred_reg if isinstance(m, Model1) else "row0"
                            selfcheck_decoder(d[0], reg, m.n_D)
                            checked = True
                        losses[m.name] = m.train_consume(epoch, d)
            except Exception as exc:  # noqa: BLE001
                log(f"epoch {epoch}: skipped ({exc})")
                continue

            if (epoch + 1) % cfg.eval_every == 0:
                for m in models:
                    try:
                        spr = 1 if isinstance(m, Model1) else None
                        d = runner.run(m.eval_pubs(epoch + 1),
                                       tag=f"eval_{m.name}_e{epoch + 1}",
                                       shots_per_randomization=spr,
                                       noise_model=None if one_job else m.noise_model)
                        m.eval_consume(epoch + 1, d)
                    except Exception as exc:  # noqa: BLE001
                        # A snapshot failure that repeats identically is a bug, not
                        # bad luck, and printing it 500 times buries that.  Say it
                        # loudly once, then only every 50th time.
                        key = (m.name, f"{exc.__class__.__name__}: {exc}")
                        snap_fail[key] = snap_fail.get(key, 0) + 1
                        n_fail = snap_fail[key]
                        if n_fail == 1:
                            log(f"epoch {epoch + 1}: {m.name} SNAPSHOT FAILED -- "
                                f"{key[1]}")
                            log(f"    metrics_history.csv will be empty while this "
                                f"persists; samples/*.npy are still written, so "
                                f"`--rescore --out {out}` can rebuild the table "
                                f"afterwards without re-running anything")
                        elif n_fail % 50 == 0:
                            log(f"epoch {epoch + 1}: {m.name} snapshot still failing "
                                f"({n_fail}x): {key[1]}")
                with open(state_path, "wb") as f:
                    pickle.dump({"epoch": epoch, **{m.name: m.state() for m in models}}, f)
                pd.DataFrame(runner.job_log).to_csv(out / "job_log.csv", index=False)

            post = {**{k: f"{v:.5f}" for k, v in losses.items()},
                    "qpu_s": f"{runner.qpu_seconds:.1f}", "jobs": runner.jobs}
            fffs = [m.fff for m in models if getattr(m, "fff", None) is not None]
            if fffs:
                post["mode"] = "/".join(f.st.mode[:2] for f in fffs)
                post["lr"] = "/".join(f"{f._lr:.2g}" for f in fffs)
            pbar.set_postfix(post)
            if fffs and all(f.finished for f in fffs):
                log(f"FFF: every model hit its stop condition at epoch {epoch}; "
                    f"finishing early")
                break

        # final, higher-statistics generation for model 3
        for m in models:
            if isinstance(m, Model3):
                try:
                    d = runner.run(m.eval_pubs(cfg.epochs, shots=cfg.m3_final_gen),
                                   tag="final_gen",
                                   noise_model=None if one_job else m.noise_model)
                    m.eval_consume(cfg.epochs, d, final=True)
                except Exception as exc:  # noqa: BLE001
                    log(f"final generation failed ({exc})")

        summary = runner.summary()
        pd.DataFrame(runner.job_log).to_csv(out / "job_log.csv", index=False)

    for m in models:
        m.finalize()

    summary["wall_seconds"] = round(time.time() - t_start, 1)
    summary["estimate"] = est
    for m in models:
        summary[m.name] = m.summary_row()
        summary[f"{m.name}_2q_gates"] = int(sum(
            m.tqc.count_ops().get(g, 0) for g in ("cz", "ecr", "cx", "rzz")))
        summary[f"{m.name}_depth"] = int(m.tqc.depth())
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def rescore_model3(cfg: Cfg, out: Path) -> None:
    """Rebuild model 3's metrics table from the samples a finished run left behind.

    eval_consume writes gen0/gen1 to samples/ *before* it scores them, so a run
    whose snapshots all died inside a metric still has every sample on disk.  This
    recomputes metrics_history.csv from those files -- no circuits, no simulator,
    no QPU, and no need to repeat the training.
    """
    from hqrnn.config.base import Config

    d = out / "model3"
    sdir = d / "samples"
    files = sorted(sdir.glob("gen0_epoch_*.npy"))
    if not files:
        raise SystemExit(f"no gen0_epoch_*.npy under {sdir}")
    log(f"rescore: {len(files)} snapshots in {sdir}")

    config = Config(model=3)
    ds = config.dataset_cfg
    real, lbl = load_digit_images(ds.csv_path, ds.first_digit, ds.second_digit)
    cnn = CnnFilter(cfg.seed)
    if cfg.m3_cnn:
        cnn.fit(real, lbl, cfg.m3_cnn_epochs)

    hist = None
    lf = d / "loss_history.csv"
    if lf.exists():
        hist = pd.read_csv(lf)
        log(f"rescore: joining losses from {lf.name} ({len(hist)} epochs)")

    rows = []
    for p0 in tqdm(files, desc="rescore"):
        ep = int(p0.stem.rsplit("_", 1)[-1])
        p1 = sdir / f"gen1_epoch_{ep:04d}.npy"
        if not p1.exists():
            continue
        gen0 = np.load(p0).astype(np.float32)
        gen1 = np.load(p1).astype(np.float32)
        n_sel = cfg.m3_select or max(1, int(0.4 * len(gen0)))
        sel0, sel1 = cnn.select(gen0, 0, n_sel), cnn.select(gen1, 1, n_sel)
        # the snapshot at epoch e reports the loss of SPSA epoch e-1
        loss = float("nan")
        if hist is not None and 0 <= ep - 1 < len(hist):
            loss = float(hist["loss"].iloc[ep - 1])
        row = {"epoch": ep, "loss": loss}
        row.update(score_samples(real, cnn, sel0, sel1, "cnn_", cfg.seed))
        row.update(score_samples(real, cnn, gen0, gen1, "raw_", cfg.seed))
        row["NN_Cond_Acc"] = metric_nn_class_acc(
            real, lbl, gen0, gen1, rng=np.random.default_rng(cfg.seed))
        rows.append(row)

    mh = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
    csv = d / "metrics_history.csv"
    mh.to_csv(csv, index=False)
    log(f"rescore: wrote {csv} ({len(mh)} rows)")

    panels = [("NN_Cond_Acc", "conditioning, nearest-neighbour (%)", 50.0),
              ("raw_CNN_Acc", "conditioning, CNN on all samples (%)", 50.0),
              ("raw_BFD", "BFD [raw] (lower is better)", None),
              ("loss", "Gap-Matching MMD", 0.2447)]
    panels = [p for p in panels if p[0] in mh and mh[p[0]].notna().any()]
    if panels:
        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.5 * len(panels)),
                                 sharex=True, squeeze=False)
        for ax, (col, label, ref) in zip(axes[:, 0], panels):
            ax.plot(mh["epoch"], mh[col], "o-", ms=2.5, lw=1.0, color="#1565c0")
            if ref is not None:
                ax.axhline(ref, color="0.4", ls="--", lw=1.0,
                           label=("chance" if ref == 50.0 else
                                  "no-conditioning floor"))
                ax.legend(fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.grid(alpha=0.3)
        axes[-1, 0].set_xlabel("epoch")
        fig.suptitle(f"model 3 metrics rebuilt from samples -- {out.name}",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        png = d / "metrics_history.png"
        fig.savefig(png, dpi=125, bbox_inches="tight")
        plt.close(fig)
        log(f"rescore: wrote {png}")

    last = mh.iloc[-1]
    print("\n" + "=" * 74)
    print(f"  RESCORED model 3 -- last snapshot (epoch {int(last['epoch'])})")
    print("=" * 74)
    for k in mh.columns:
        if k != "epoch" and isinstance(last[k], (int, float)):
            print(f"  {k:<32}: {last[k]:.4f}")
    print("=" * 74)


def print_summary(summary: dict, out: Path, kind: str) -> None:
    print("\n" + "=" * 74)
    print("  RUN SUMMARY")
    print("=" * 74)
    print(f"  jobs submitted      : {summary.get('jobs')}")
    print(f"  PUBs                : {summary.get('pubs')}")
    print(f"  shots               : {summary.get('shots', 0):,}")
    print(f"  billed QPU seconds  : "
          f"{summary.get('qpu_seconds') if kind == 'hardware' else 'n/a (simulator)'}")
    print(f"  wall clock seconds  : {summary.get('wall_seconds')}")
    print(f"  outputs             : {out.resolve()}")
    print("=" * 74)


def run_ablation(cfg: Cfg, root: Path) -> None:
    """Run every arm of the sweep back to back and write one comparison table."""
    if cfg.noisy and not (cfg.fake or cfg.calibrate_from):
        raise SystemExit("--noisy needs --fake ibm_fez or --calibrate-from <backend>")
    if cfg.noisy:
        log("NOTE: --dd and --no-twirling have no effect in local simulation. "
            "Runtime dynamical decoupling and twirling are applied server-side, "
            "so arms 07 and 08 only differ on real hardware.")

    shared: dict = {}          # data handler, digit images and CNN reused across arms
    rows: list = []
    t0 = time.time()
    for i, (name, note, overrides) in enumerate(ABLATION_ARMS):
        arm_out = root / name
        log("=" * 70)
        log(f"ARM {i}/{len(ABLATION_ARMS) - 1}  {name}  ({note})")
        if rows:
            done = time.time() - t0
            left = done / len(rows) * (len(ABLATION_ARMS) - len(rows))
            log(f"elapsed {done / 60:.1f} min, roughly {left / 60:.1f} min left")
        log("=" * 70)

        arm_cfg = replace(cfg, ablation=False, out=str(arm_out), yes=True, **overrides)
        row = {"arm": name, "note": note,
               "changed": ", ".join(f"{k}={v}" for k, v in overrides.items()) or "-"}
        try:
            backend, kind = open_backend(arm_cfg, min_qubits=14)
            summary = run_once(arm_cfg, arm_out, backend, kind, shared, quiet=True)
            row["status"] = "ok"
            row["wall_s"] = summary.get("wall_seconds")
            row["qpu_s"] = summary.get("qpu_seconds")
            for mname in ("model1", "model3"):
                if mname in summary:
                    row[f"{mname}_depth"] = summary.get(f"{mname}_depth")
                    row[f"{mname}_2q"] = summary.get(f"{mname}_2q_gates")
                    for k, v in summary[mname].items():
                        row[f"{mname}_{k}"] = v
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"skipped: {exc}"
            log(f"ARM {name} skipped: {exc}")
        rows.append(row)
        ddf = compare_ablation(root, rows)
        log(f"comparison updated: ablation_summary.csv, ablation_deltas.csv, "
            f"ablation_metrics.png, ablation_cost.png, ablation_loss_curves.png "
            f"({len(rows)} arms)")

    print_ablation_table(pd.DataFrame(rows), ddf)
    print(f"total {(time.time() - t0) / 60:.1f} min -> {root.resolve()}")


# --- ablation comparison -----------------------------------------------------
#
# Direction is +1 when higher is better and -1 when lower is better, so the
# delta table can mark an arm as an improvement without the reader having to
# remember which way each metric points.
_GEN_METRICS = [("BFD", -1), ("Uniqueness", +1), ("Novelty", +1),
                ("Recall", +1), ("HMean", +1)]

COMPARE_METRICS = (
    # conditioning first: both of these are honest, chance is 50%
    [("model3_NN_Cond_Acc", "model 3 conditioning, nearest-neighbour (%)", +1, None),
     ("model3_raw_CNN_Acc", "model 3 conditioning, CNN on all samples (%)", +1, None),
     ("model3_loss_tail_mean", "model 3 loss (tail mean)", -1, "model3_loss_tail_std"),
     ("model3_best_loss", "model 3 best loss", -1, None)]
    + [(f"model3_raw_{n}", f"model 3 {n} [raw]", d, None) for n, d in _GEN_METRICS]
    + [(f"model3_cnn_{n}", f"model 3 {n} [CNN filtered]", d, None) for n, d in _GEN_METRICS]
    + [("model1_loss_tail_mean", "model 1 MMD loss", -1, "model1_loss_tail_std"),
       ("model1_test_DA(%)", "model 1 test DA (%)", +1, None),
       ("model1_test_MAE(Rate)", "model 1 test MAE of rate", -1, None),
       ("model1_test_IC", "model 1 test IC", +1, None)]
)
COST_METRICS = [
    ("model1_depth", "model 1 transpiled depth"),
    ("model1_2q", "model 1 two-qubit gates"),
    ("model3_depth", "model 3 transpiled depth"),
    ("model3_2q", "model 3 two-qubit gates"),
]


def _barh(ax, names, vals, errs, title, direction, base_val):
    y = np.arange(len(names))
    colours = []
    for v in vals:
        if v != v or base_val != base_val:
            colours.append("0.7")
        elif (v - base_val) * direction > 0:
            colours.append("#2e7d32")          # better than baseline
        elif (v - base_val) * direction < 0:
            colours.append("#c62828")          # worse
        else:
            colours.append("#546e7a")
    ax.barh(y, vals, xerr=errs, color=colours, height=0.65,
            error_kw={"ecolor": "0.35", "elinewidth": 1, "capsize": 2})
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    if base_val == base_val:
        ax.axvline(base_val, color="0.25", ls="--", lw=1)
    ax.set_title(f"{title}  ({'higher' if direction > 0 else 'lower'} is better)",
                 fontsize=9)
    ax.grid(axis="x", alpha=0.3)


def compare_ablation(root: Path, rows: list) -> pd.DataFrame:
    """Write the comparison table, the deltas against the baseline and the plots.

    Called after every arm, so a sweep that is stopped early still leaves a
    usable comparison behind.
    """
    df = pd.DataFrame(rows)
    df.to_csv(root / "ablation_summary.csv", index=False)
    ok = df[df["status"] == "ok"] if "status" in df else df
    if ok.empty:
        return df

    base = ok[ok["arm"].str.startswith("00_")]
    base = base.iloc[0] if len(base) else None
    names = ok["arm"].tolist()

    # -- deltas against the baseline ------------------------------------------
    deltas = [{"arm": r["arm"], "changed": r.get("changed", "")} for _, r in ok.iterrows()]
    for col, label, direction, _ in COMPARE_METRICS:
        if col not in ok:
            continue
        b = float(base[col]) if base is not None and col in base else float("nan")
        for d, (_, r) in zip(deltas, ok.iterrows()):
            v = r.get(col, float("nan"))
            d[col] = v
            if v == v and b == b:
                d[f"{col}_delta"] = v - b
                d[f"{col}_better"] = bool((v - b) * direction > 0)
    ddf = pd.DataFrame(deltas)
    ddf.to_csv(root / "ablation_deltas.csv", index=False)

    # -- metric panels ---------------------------------------------------------
    present = [m for m in COMPARE_METRICS if m[0] in ok and ok[m[0]].notna().any()]
    if present:
        n = len(present)
        cols = 3
        rows_n = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(13, 2.6 * rows_n))
        axes = np.atleast_1d(axes).ravel()
        for ax, (col, label, direction, errcol) in zip(axes, present):
            vals = ok[col].astype(float).to_numpy()
            errs = (ok[errcol].astype(float).to_numpy()
                    if errcol and errcol in ok else None)
            b = float(base[col]) if base is not None and col in base else float("nan")
            _barh(ax, names, vals, errs, label, direction, b)
        for ax in axes[len(present):]:
            ax.axis("off")
        fig.suptitle("Ablation: metrics against the baseline (dashed line)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(root / "ablation_metrics.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # -- circuit cost ----------------------------------------------------------
    cost = [m for m in COST_METRICS if m[0] in ok and ok[m[0]].notna().any()]
    if cost:
        fig, axes = plt.subplots(1, len(cost), figsize=(4.2 * len(cost), 3.4))
        axes = np.atleast_1d(axes).ravel()
        for ax, (col, label) in zip(axes, cost):
            vals = ok[col].astype(float).to_numpy()
            b = float(base[col]) if base is not None and col in base else float("nan")
            _barh(ax, names, vals, None, label, -1, b)
        fig.suptitle("Ablation: circuit cost", fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(root / "ablation_cost.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # -- overlaid SPSA loss curves --------------------------------------------
    curves = {}
    for arm in names:
        for model in ("model1", "model3"):
            f = root / arm / model / "loss_history.csv"
            if f.exists():
                try:
                    curves.setdefault(model, {})[arm] = pd.read_csv(f)["loss"].to_numpy()
                except Exception:
                    pass
    if curves:
        fig, axes = plt.subplots(1, len(curves), figsize=(7.5 * len(curves), 4.2),
                                 squeeze=False)
        for ax, (model, per_arm) in zip(axes[0], curves.items()):
            for arm, h in per_arm.items():
                w = max(1, len(h) // 20)
                smooth = np.convolve(h, np.ones(w) / w, mode="valid")
                ax.plot(np.arange(len(smooth)), smooth, lw=2.0 if arm.startswith("00_")
                        else 1.1, label=arm, alpha=1.0 if arm.startswith("00_") else 0.8)
            ax.set_xlabel("SPSA epoch"); ax.set_ylabel("MMD loss")
            ax.set_title(f"{model} loss, moving average"); ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(root / "ablation_loss_curves.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    return ddf


def print_ablation_table(df: pd.DataFrame, ddf: pd.DataFrame) -> None:
    print("\n" + "=" * 110)
    print("  ABLATION SUMMARY")
    print("=" * 110)
    cols = [c for c in ("arm", "changed", "status", "model3_depth", "model3_2q",
                        "model3_loss_tail_mean", "model3_NN_Cond_Acc",
                        "model3_raw_CNN_Acc", "model3_raw_BFD", "model3_raw_Recall",
                        "model3_raw_HMean", "qpu_s", "wall_s") if c in df.columns]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if ddf is not None and not ddf.empty:
        print("\n  BEST ARM PER METRIC")
        print("-" * 110)
        for col, label, direction, errcol in COMPARE_METRICS:
            if col not in ddf or ddf[col].notna().sum() == 0:
                continue
            idx = ddf[col].idxmax() if direction > 0 else ddf[col].idxmin()
            best = ddf.loc[idx]
            note = ""
            if errcol and errcol in df.columns:
                sd = df.loc[df["arm"] == best["arm"], errcol]
                if len(sd) and sd.iloc[0] == sd.iloc[0]:
                    note = f"   (arm noise sd {float(sd.iloc[0]):.4f})"
            print(f"  {label:<34} {best['arm']:<18} {float(best[col]):>10.4f}{note}")
        print("-" * 110)
        print("  Single seed per arm, so treat any gap smaller than the noise sd as a tie.")
        print("  Novelty alone is misleading: a generator producing noise scores near 100%.")
        print("  Read it together with Recall, and prefer raw_ over cnn_.")
        print("  Conditioning: use NN_Cond_Acc or raw_CNN_Acc, chance is 50%. The old")
        print("  cnn_CNN_Acc is gone; it read 100% even for a generator with zero")
        print("  conditioning, because the CNN scored the samples it had just picked.")
    print("=" * 110)


def main() -> None:
    cfg = parse_args()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.noisy and cfg.hardware:
        raise SystemExit("--noisy is for local simulation; drop --hardware")
    if cfg.noisy and not (cfg.fake or cfg.calibrate_from):
        raise SystemExit("--noisy needs --fake ibm_fez or --calibrate-from <backend>")

    if cfg.rescore:
        rescore_model3(cfg, out)
        return

    if cfg.ablation:
        run_ablation(cfg, out)
        return

    if cfg.hardware and cfg.exec_mode == "session":
        log("WARNING: --mode session bills wall clock, not execution time")
    backend, kind = open_backend(cfg, min_qubits=14)
    if kind == "hardware" and cfg.only in (0, 1):
        try:
            if "rzz" in backend.target.operation_names:
                log("NOTE: model 1 uses mid-circuit reset while this backend was "
                    "opened with fractional gates. If the service rejects that "
                    "combination, rerun with --fractional off.")
        except Exception:
            pass
    if cfg.noisy:
        log("NOTE: --dd and --no-twirling have no effect in local simulation; "
            "both are applied server-side by the Runtime.")

    summary = run_once(cfg, out, backend, kind)
    if summary:
        print_summary(summary, out, kind)


if __name__ == "__main__":
    main()
