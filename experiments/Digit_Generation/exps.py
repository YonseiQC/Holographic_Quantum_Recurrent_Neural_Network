# --- 0. Imports

import os
import pickle
import threading
import warnings
import multiprocessing
import concurrent.futures
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random, jit, lax
from jax import tree_util
import optax
from scipy import linalg, ndimage
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import msgpack
from flax import linen as nn
from flax.training import train_state as flax_train_state

warnings.filterwarnings("ignore")

_N_CPU = multiprocessing.cpu_count()
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={_N_CPU}")
os.environ.setdefault("XLA_THREAD_POOL_SIZE", str(_N_CPU))
os.environ.setdefault("OMP_NUM_THREADS",      str(_N_CPU))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_N_CPU))
os.environ.setdefault("MKL_NUM_THREADS",      str(_N_CPU))

try:
    from hqrnn.config.base import Config
    from hqrnn.quantum.model import QuantumModel
    from hqrnn.scheduler.scheduler import Scheduler
    from hqrnn.FFF_mode.controller import ModeController
    from hqrnn.scheduler.loss_history import LossHistory
    from hqrnn.data_handler.digits import DigitDataHandler
    _HQRNN_OK = True
except ImportError:
    Config = QuantumModel = Scheduler = ModeController = None
    LossHistory = DigitDataHandler = None
    _HQRNN_OK = False


# --- 1. Paths & Global Settings

GLOBAL_SEED         = 42
N_SAMPLES_PER_DIGIT = 5000
KDE_SAMPLES         = 2000
BATCH_SIZE          = 32
MAX_EPOCHS          = 20000

DATASET_URL = (
    "https://raw.githubusercontent.com/w-jiwonchoi/"
    "HQRNN_dataset/main/model_3_digit.csv"
)

CKPT_DIR        = Path("./checkpoints")
HQRNN_CKPT      = CKPT_DIR / "HQRNN.pkl"
NATIVE_CKPT_DIR = CKPT_DIR / "native"

ABLATION_CKPTS = {
    "noGapMatching":       CKPT_DIR / "ablation" / "noGapMatching.pkl",
    "noMMD":               CKPT_DIR / "ablation" / "noMMD.pkl",
    "noFFF":               CKPT_DIR / "ablation" / "noFFF.pkl",
    "noAdaptiveWeighting": CKPT_DIR / "ablation" / "noAdaptiveWeighting.pkl",
}

EXP_COMP = Path("./experiments/comparison")
EXP_ABL  = Path("./experiments/ablation")

for d in (EXP_COMP, EXP_ABL, NATIVE_CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

np.random.seed(GLOBAL_SEED)
_tqdm_lock = threading.Lock()


# --- 2. Dataset

_df = pd.read_csv(DATASET_URL)

_imgs_raw, _lbls_raw = [], []
for _, row in _df.iterrows():
    lbl = int(row["label"])
    if lbl in [0, 1]:
        _imgs_raw.append((row.iloc[1:50].values > 0.5).astype(np.float32).flatten())
        _lbls_raw.append(lbl)

images_dataset = jnp.array(_imgs_raw)
labels_dataset = jnp.array(_lbls_raw)
DATASET_FLAT   = np.array(images_dataset)
DATASET_LABELS = np.array(labels_dataset)


# --- 3. HQRNN Globals

if _HQRNN_OK:
    _cfg_global = Config(model=3)
    _dh         = DigitDataHandler(_cfg_global)
    TARGET_MMD  = _dh.calculate_target_mmd(sigma=_cfg_global.loss_cfg.mmd_sigma)
    _D_BIT      = _cfg_global.model_cfg.n_D
    BIT_TABLE   = jnp.array(
        [[(j >> (_D_BIT - 1 - i)) & 1 for i in range(_D_BIT)]
         for j in range(2 ** _D_BIT)],
        dtype=jnp.float32)
else:
    _cfg_global = None
    TARGET_MMD  = None
    _D_BIT      = 7
    BIT_TABLE   = None


# --- 4. Shared Helpers

@jit
def gumbel_sigmoid(logits, key, temp=1.0):
    u = random.uniform(key, shape=logits.shape, minval=1e-5, maxval=1.0 - 1e-5)
    return jax.nn.sigmoid((logits - jnp.log(-jnp.log(u))) / temp)

@jit
def vectorization(X_batch):
    powers = 2 ** jnp.arange(6, -1, -1)
    return jnp.sum(X_batch * powers[None, None, :], axis=2).astype(jnp.int32)

def ensure_serializable(obj):
    if isinstance(obj, dict):          return {k: ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [ensure_serializable(x) for x in obj]
    if hasattr(obj, "__array__"):      return float(obj) if obj.size == 1 else obj.tolist()
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):return float(obj)
    return obj


# --- 5. Classical Model Architectures

HIDDEN_DIM_MADE = 4

def create_masks_made(input_dim, hidden_dim):
    np.random.seed(GLOBAL_SEED)
    m_input  = np.arange(input_dim)
    m_hidden = np.random.randint(m_input.min(), input_dim - 1, size=hidden_dim)
    return (jnp.array((m_input[:, None] <= m_hidden[None, :]).astype(np.float32)),
            jnp.array((m_hidden[:, None] < m_input[None, :]).astype(np.float32)))

W1_mask_made, W2_mask_made = create_masks_made(49, HIDDEN_DIM_MADE)

def init_params_made(key):
    k1, k2, k3 = random.split(key, 3)
    return {
        "W1":        random.normal(k1, (49, HIDDEN_DIM_MADE)) * jnp.sqrt(2.0 / 49),
        "b1":        jnp.zeros(HIDDEN_DIM_MADE),
        "W2":        random.normal(k2, (HIDDEN_DIM_MADE, 49)) * jnp.sqrt(2.0 / HIDDEN_DIM_MADE),
        "b2":        jnp.zeros(49),
        "label_emb": random.normal(k3, (2, HIDDEN_DIM_MADE)) * 0.1,
    }

@jit
def forward_made_soft(params, key, label, temp):
    x = jnp.zeros(49)
    def step(i, val):
        key, x = val
        key, sk = random.split(key)
        h      = jax.nn.relu(x @ (params["W1"] * W1_mask_made) + params["b1"] + params["label_emb"][label])
        logits = h @ (params["W2"] * W2_mask_made) + params["b2"]
        return key, x.at[i].set(gumbel_sigmoid(logits[i], sk, temp))
    _, x_out = lax.fori_loop(0, 49, step, (key, x))
    return x_out

def generate_made(params, key, label):
    x = jnp.zeros(49)
    def step(i, val):
        key, x = val
        key, sk = random.split(key)
        h      = jax.nn.relu(x @ (params["W1"] * W1_mask_made) + params["b1"] + params["label_emb"][label])
        logits = h @ (params["W2"] * W2_mask_made) + params["b2"]
        return key, x.at[i].set(random.bernoulli(sk, jax.nn.sigmoid(logits[i])).astype(jnp.float32))
    _, x_out = lax.fori_loop(0, 49, step, (key, x))
    return x_out

@jit
def native_loss_made(params, x, label, key):
    h      = jax.nn.relu(x @ (params["W1"] * W1_mask_made) + params["b1"] + params["label_emb"][label])
    logits = h @ (params["W2"] * W2_mask_made) + params["b2"]
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, x))


HIDDEN_DIM_NADE = 4

def init_params_nade(key):
    k1, k2, k3 = random.split(key, 3)
    return {
        "W":         random.normal(k1, (49, HIDDEN_DIM_NADE)) * 0.1,
        "c":         jnp.zeros(HIDDEN_DIM_NADE),
        "V":         random.normal(k2, (HIDDEN_DIM_NADE, 49)) * 0.1,
        "b":         jnp.zeros(49),
        "label_emb": random.normal(k3, (2, HIDDEN_DIM_NADE)) * 0.1,
    }

@jit
def forward_nade_soft(params, key, label, temp):
    a = params["c"] + params["label_emb"][label]
    def step(val, i):
        key, a = val
        key, sk = random.split(key)
        h      = jax.nn.sigmoid(a)
        logit  = jnp.dot(h, params["V"][:, i]) + params["b"][i]
        pixel  = gumbel_sigmoid(logit, sk, temp)
        return (key, a + params["W"][i] * pixel), pixel
    _, x_out = lax.scan(step, (key, a), jnp.arange(49))
    return x_out

def generate_nade(params, key, label):
    a = params["c"] + params["label_emb"][label]
    def step(val, i):
        key, a = val
        key, sk = random.split(key)
        h      = jax.nn.sigmoid(a)
        prob   = jax.nn.sigmoid(jnp.dot(h, params["V"][:, i]) + params["b"][i])
        pixel  = random.bernoulli(sk, prob).astype(jnp.float32)
        return (key, a + params["W"][i] * pixel), pixel
    _, x_out = lax.scan(step, (key, a), jnp.arange(49))
    return x_out

@jit
def native_loss_nade(params, x, label, key):
    a = params["c"] + params["label_emb"][label]
    def step(a, i):
        h      = jax.nn.sigmoid(a)
        logit  = jnp.dot(h, params["V"][:, i]) + params["b"][i]
        loss_i = optax.sigmoid_binary_cross_entropy(logit, x[i])
        return a + params["W"][i] * x[i], loss_i
    _, losses = lax.scan(step, a, jnp.arange(49))
    return jnp.mean(losses)


RNN_H, RNN_O = 8, 6

def init_params_pixelrnn(key):
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)
    return {
        "W_ih":        random.normal(k1, (1,     4 * RNN_H)) * 0.1,
        "W_hh":        random.normal(k2, (RNN_H, 4 * RNN_H)) * 0.1,
        "b":           jnp.zeros(4 * RNN_H),
        "label_emb_h": random.normal(k3, (2, RNN_H)) * 0.1,
        "label_emb_c": random.normal(k4, (2, RNN_H)) * 0.1,
        "W_h1":        random.normal(k5, (RNN_H, RNN_O)) * 0.1,
        "b_h1":        jnp.zeros(RNN_O),
        "W_ho":        random.normal(k6, (RNN_O, 1)) * 0.1,
        "b_o":         jnp.zeros(1),
    }

@jit
def _lstm_step(params, h, c, x_t):
    gates = jnp.array([x_t]) @ params["W_ih"] + h @ params["W_hh"] + params["b"]
    i_g   = jax.nn.sigmoid(gates[:RNN_H])
    f_g   = jax.nn.sigmoid(gates[RNN_H:  2 * RNN_H])
    g_g   = jnp.tanh(gates[2 * RNN_H:   3 * RNN_H])
    o_g   = jax.nn.sigmoid(gates[3 * RNN_H: 4 * RNN_H])
    c_n   = f_g * c + i_g * g_g
    h_n   = o_g * jnp.tanh(c_n)
    logit = (jax.nn.relu(h_n @ params["W_h1"] + params["b_h1"]) @ params["W_ho"] + params["b_o"])[0]
    return h_n, c_n, logit

@jit
def forward_pixelrnn_soft(params, key, label, temp):
    h, c = params["label_emb_h"][label], params["label_emb_c"][label]
    x    = jnp.zeros(49)
    def body(i, carry):
        h, c, x, key = carry
        key, sk = random.split(key)
        h, c, logit = _lstm_step(params, h, c, jnp.where(i == 0, 0.0, x[i - 1]))
        return h, c, x.at[i].set(gumbel_sigmoid(logit, sk, temp)), key
    _, _, x_out, _ = lax.fori_loop(0, 49, body, (h, c, x, key))
    return x_out

def generate_pixelrnn(params, key, label):
    h, c = params["label_emb_h"][label], params["label_emb_c"][label]
    x    = jnp.zeros(49)
    def body(i, carry):
        h, c, x, key = carry
        key, sk = random.split(key)
        h, c, logit = _lstm_step(params, h, c, jnp.where(i == 0, 0.0, x[i - 1]))
        return (h, c, x.at[i].set(random.bernoulli(sk, jax.nn.sigmoid(logit)).astype(jnp.float32)), key)
    _, _, x_out, _ = lax.fori_loop(0, 49, body, (h, c, x, key))
    return x_out

@jit
def native_loss_pixelrnn(params, x, label, key):
    h, c = params["label_emb_h"][label], params["label_emb_c"][label]
    def body(carry, i):
        h, c = carry
        h, c, logit = _lstm_step(params, h, c, jnp.where(i == 0, 0.0, x[i - 1]))
        return (h, c), optax.sigmoid_binary_cross_entropy(logit, x[i])
    _, losses = lax.scan(body, (h, c), jnp.arange(49))
    return jnp.mean(losses)


CNN_H = 6

def _make_pcnn_masks():
    m = jnp.ones((3, 3), dtype=jnp.float32).at[1, 2].set(0).at[2, :].set(0)
    return m.at[1, 1].set(0), m

_MASK_A_CNN, _MASK_B_CNN = _make_pcnn_masks()

def init_params_pixelcnn(key):
    k1, k2, k3, k4, k5 = random.split(key, 5)
    return {
        "label_emb": random.normal(k1, (2, CNN_H))           * 0.1,
        "conv1_w":   random.normal(k2, (3, 3, 1,     CNN_H)) * jnp.sqrt(2.0 / 9),
        "conv1_b":   jnp.zeros(CNN_H),
        "conv2_w":   random.normal(k3, (3, 3, CNN_H, CNN_H)) * jnp.sqrt(2.0 / (9 * CNN_H)),
        "conv2_b":   jnp.zeros(CNN_H),
        "conv3_w":   random.normal(k4, (CNN_H, 1))           * jnp.sqrt(2.0 / CNN_H),
        "conv3_b":   jnp.zeros(1),
        "output_w":  random.normal(k5, (1, 2))               * jnp.sqrt(2.0),
        "output_b":  jnp.zeros(2),
    }

@jit
def _masked_conv(x, kernel, bias, mask):
    mk  = kernel * mask[:, :, None, None]
    out = jax.lax.conv_general_dilated(
        x.transpose(2, 0, 1)[None], mk.transpose(3, 2, 0, 1),
        (1, 1), ((1, 1), (1, 1)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return out[0].transpose(1, 2, 0) + bias

@jit
def _pcnn_logits(params, x, label):
    xr = x.reshape(7, 7, 1)
    h  = jax.nn.relu(_masked_conv(xr, params["conv1_w"], params["conv1_b"], _MASK_A_CNN))
    h  = h + params["label_emb"][label][None, None, :]
    h  = jax.nn.relu(_masked_conv(h, params["conv2_w"], params["conv2_b"], _MASK_B_CNN))
    h  = jax.nn.relu(h @ params["conv3_w"] + params["conv3_b"])
    return h @ params["output_w"] + params["output_b"]

@jit
def forward_pixelcnn_soft(params, key, label, temp):
    x = jnp.zeros((7, 7))
    def step(k_flat, val):
        key, x = val
        key, sk = random.split(key)
        i, j   = k_flat // 7, k_flat % 7
        logit  = _pcnn_logits(params, x, label)[i, j, 1] - _pcnn_logits(params, x, label)[i, j, 0]
        return key, x.at[i, j].set(gumbel_sigmoid(logit, sk, temp))
    _, x_out = lax.fori_loop(0, 49, step, (key, x))
    return x_out.flatten()

def generate_pixelcnn(params, key, label):
    x = jnp.zeros((7, 7))
    def step(k_flat, val):
        key, x = val
        key, sk = random.split(key)
        i, j   = k_flat // 7, k_flat % 7
        logit  = _pcnn_logits(params, x, label)[i, j, 1] - _pcnn_logits(params, x, label)[i, j, 0]
        return key, x.at[i, j].set(random.bernoulli(sk, jax.nn.sigmoid(logit)).astype(jnp.float32))
    _, x_out = lax.fori_loop(0, 49, step, (key, x))
    return x_out.flatten()

@jit
def native_loss_pixelcnn(params, x, label, key):
    logits = _pcnn_logits(params, x, label)
    logit  = logits[:, :, 1] - logits[:, :, 0]
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logit.flatten(), x))


VAE_H, VAE_Z, VAE_E = 3, 4, 5

def init_params_vae(key):
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)
    return {
        "enc_w1":       random.normal(k1, (49,            VAE_H)) * 0.1,
        "enc_b1":       jnp.zeros(VAE_H),
        "enc_mu_w":     random.normal(k2, (VAE_H,         VAE_Z)) * 0.1,
        "enc_mu_b":     jnp.zeros(VAE_Z),
        "enc_logvar_w": random.normal(k3, (VAE_H,         VAE_Z)) * 0.1,
        "enc_logvar_b": jnp.zeros(VAE_Z),
        "label_emb":    random.normal(k4, (2,             VAE_E)) * 0.1,
        "dec_w1":       random.normal(k5, (VAE_Z + VAE_E, VAE_H)) * 0.1,
        "dec_b1":       jnp.zeros(VAE_H),
        "dec_w2":       random.normal(k6, (VAE_H,         49))    * 0.1,
        "dec_b2":       jnp.zeros(49),
    }

@jit
def _vae_encode(params, x):
    h      = jax.nn.relu(x @ params["enc_w1"] + params["enc_b1"])
    mu     = h @ params["enc_mu_w"] + params["enc_mu_b"]
    logvar = jnp.clip(h @ params["enc_logvar_w"] + params["enc_logvar_b"], -4.0, 4.0)
    return mu, logvar

@jit
def _vae_decode(params, z, label):
    z_c = jnp.concatenate([z, params["label_emb"][label]])
    h   = jax.nn.relu(z_c @ params["dec_w1"] + params["dec_b1"])
    return h @ params["dec_w2"] + params["dec_b2"]

@jit
def forward_vae_soft(params, key, label, temp):
    key, zk, sk = random.split(key, 3)
    return gumbel_sigmoid(_vae_decode(params, random.normal(zk, (VAE_Z,)), label), sk, temp)

def generate_vae(params, key, label):
    key, zk, sk = random.split(key, 3)
    return random.bernoulli(
        sk, jax.nn.sigmoid(_vae_decode(params, random.normal(zk, (VAE_Z,)), label))
    ).astype(jnp.float32)

@jit
def native_loss_vae(params, x, label, key):
    mu, logvar = _vae_encode(params, x)
    z          = mu + jnp.exp(0.5 * logvar) * random.normal(key, mu.shape)
    logits     = _vae_decode(params, z, label)
    bce        = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, x))
    kl         = -0.5 * jnp.mean(1.0 + logvar - mu ** 2 - jnp.exp(logvar))
    return bce + 0.1 * kl

@jit
def vae_kl_aux(params, X_flat):
    def _kl(x):
        mu, lv = _vae_encode(params, x)
        return -0.5 * jnp.sum(1.0 + lv - mu ** 2 - jnp.exp(lv))
    return jnp.mean(jax.vmap(_kl)(X_flat))


RBM_H          = 7
RBM_CD_K       = 3
RBM_GEN_STEPS  = 100
RBM_FAIR_STEPS = 50

def init_params_rbm(key):
    k1, k2 = random.split(key, 2)
    return {
        "W":         random.normal(k1, (49, RBM_H)) * 0.01,
        "b_v":       jnp.zeros(49),
        "b_h":       jnp.zeros(RBM_H),
        "label_emb": random.normal(k2, (2,  RBM_H)) * 0.1,
    }

@jit
def forward_rbm_soft(params, key, label, temp):
    key, ik = random.split(key)
    v  = random.bernoulli(ik, 0.5, shape=(49,)).astype(jnp.float32)
    lb = params["label_emb"][label]
    def body(_, val):
        v, k = val; k, k1, k2 = random.split(k, 3)
        h = gumbel_sigmoid(v @ params["W"] + params["b_h"] + lb, k1, temp)
        return gumbel_sigmoid(h @ params["W"].T + params["b_v"], k2, temp), k
    v_out, _ = lax.fori_loop(0, RBM_FAIR_STEPS, body, (v, key))
    return v_out

def generate_rbm(params, key, label):
    key, ik = random.split(key)
    v  = random.bernoulli(ik, 0.5, shape=(49,)).astype(jnp.float32)
    lb = params["label_emb"][label]
    def body(_, val):
        v, k = val; k, k1, k2 = random.split(k, 3)
        h   = random.bernoulli(k1, jax.nn.sigmoid(v @ params["W"] + params["b_h"] + lb)).astype(jnp.float32)
        v_n = random.bernoulli(k2, jax.nn.sigmoid(h @ params["W"].T + params["b_v"])).astype(jnp.float32)
        return v_n, k
    v_out, _ = lax.fori_loop(0, RBM_GEN_STEPS, body, (v, key))
    return v_out

@jit
def native_loss_rbm(params, x, label, key):
    lb = params["label_emb"][label]
    def free_energy(v):
        return (-jnp.dot(v, params["b_v"])
                - jnp.sum(jax.nn.softplus(v @ params["W"] + params["b_h"] + lb)))
    pos_fe = free_energy(x)
    def cd_step(i, val):
        v, k = val; k, k1, k2 = random.split(k, 3)
        h   = random.bernoulli(k1, jax.nn.sigmoid(v @ params["W"] + params["b_h"] + lb)).astype(jnp.float32)
        v_n = random.bernoulli(k2, jax.nn.sigmoid(h @ params["W"].T + params["b_v"])).astype(jnp.float32)
        return v_n, k
    v_neg, _ = lax.fori_loop(0, RBM_CD_K, cd_step, (x, key))
    v_neg    = jax.lax.stop_gradient(v_neg)
    return pos_fe - free_energy(v_neg)


# --- 6. Classical Model Registry

CLASSICAL_REGISTRY = {
    "MADE":     (init_params_made,     forward_made_soft,     generate_made,
                 native_loss_made,     None,       0.0),
    "NADE":     (init_params_nade,     forward_nade_soft,     generate_nade,
                 native_loss_nade,     None,       0.0),
    "PixelRNN": (init_params_pixelrnn, forward_pixelrnn_soft, generate_pixelrnn,
                 native_loss_pixelrnn, None,       0.0),
    "PixelCNN": (init_params_pixelcnn, forward_pixelcnn_soft, generate_pixelcnn,
                 native_loss_pixelcnn, None,       0.0),
    "CVAE":     (init_params_vae,      forward_vae_soft,      generate_vae,
                 native_loss_vae,      vae_kl_aux, 0.1),
    "RBM":      (init_params_rbm,      forward_rbm_soft,      generate_rbm,
                 native_loss_rbm,      None,       0.0),
}

MODEL_ORDER = ["Ours", "MADE", "NADE", "PixelRNN", "PixelCNN", "CVAE", "RBM"]


# --- 7. Training - Fair

def train_fair(model_name: str, init_fn, forward_soft_fn, save_dir: Path,
               aux_loss_fn=None, aux_weight: float = 0.0):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "best.pkl"
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)["params"]

    key       = random.PRNGKey(GLOBAL_SEED)
    key, ik   = random.split(key)
    params    = init_fn(ik)
    scheduler = Scheduler(_cfg_global)
    mode_ctrl = ModeController(_cfg_global)
    loss_hist = LossHistory()

    opt = optax.chain(
        optax.clip_by_global_norm(_cfg_global.regularization_cfg.clip_norm),
        optax.scale_by_adam(),
        optax.add_decayed_weights(_cfg_global.regularization_cfg.weight_decay),
        optax.scale(-1.0),
    )
    opt_state = opt.init(params)

    tgt   = jnp.asarray(TARGET_MMD, dtype=jnp.float32)
    D     = _D_BIT
    T     = _cfg_global.model_cfg.seq_len
    sigma = _cfg_global.loss_cfg.mmd_sigma
    gamma = 1.0 / (2.0 * sigma ** 2)
    lam   = _cfg_global.loss_cfg.mmd_lambda
    kk    = _cfg_global.loss_cfg.mmd_k

    @jit
    def mmd_loss(params, X, Y, L, key, temp):
        B  = X.shape[0]
        ks = random.split(key, B)
        Xg = jax.vmap(lambda k, l: forward_soft_fn(params, k, l, temp))(ks, L)
        Xg = Xg.reshape(B, T * D)
        Qb = BIT_TABLE[Y].reshape(B, T * D)

        def rbf(A, B_):
            d = A[:, None, :] - B_[None, :, :]
            return jnp.exp(-gamma * jnp.sum(d * d, axis=-1))

        def mmd_masked(P, Q, mP, mQ):
            nP = jnp.sum(mP); nQ = jnp.sum(mQ)
            Kpp = rbf(P, P); Kqq = rbf(Q, Q); Kpq = rbf(P, Q)
            tp  = jnp.sum(Kpp * (mP[:, None] * mP[None, :])) / jnp.maximum(nP ** 2, 1e-9)
            tq  = jnp.sum(Kqq * (mQ[:, None] * mQ[None, :])) / jnp.maximum(nQ ** 2, 1e-9)
            tpq = jnp.sum(Kpq * (mP[:, None] * mQ[None, :])) / jnp.maximum(nP * nQ, 1e-9)
            return jnp.where((nP > 0) & (nQ > 0), tp - 2.0 * tpq + tq, 0.0)

        m0 = (L == 0).astype(jnp.float32)
        m1 = (L == 1).astype(jnp.float32)
        L0 = mmd_masked(Xg, Qb, m0, m0) + lam * (mmd_masked(Xg, Qb, m0, m1) - tgt) ** 2
        L1 = mmd_masked(Xg, Qb, m1, m1) + lam * (mmd_masked(Xg, Qb, m1, m0) - tgt) ** 2
        return jax.nn.logsumexp(kk * jnp.stack([L0, L1])) / kk

    if aux_loss_fn is not None:
        @jit
        def total_loss(params, X, Y, L, key, temp):
            return (mmd_loss(params, X, Y, L, key, temp)
                    + aux_weight * aux_loss_fn(params, X.reshape(X.shape[0], -1)))
    else:
        total_loss = mmd_loss

    @jit
    def step(params, opt_state, X, Y, L, key, lr, temp):
        loss, grads = jax.value_and_grad(total_loss)(params, X, Y, L, key, temp)
        grads = tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        upd, ns = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, tree_util.tree_map(lambda u: lr * u, upd)), ns, loss

    best_loss, best_params = float("inf"), None
    T_START, T_END = 2.0, 0.1

    for epoch in tqdm(range(MAX_EPOCHS), desc=f"[Fair] {model_name}", leave=False):
        key, bk, lk = random.split(key, 3)
        idx = random.choice(bk, len(images_dataset), (BATCH_SIZE,), replace=False)
        Xb  = images_dataset[idx].reshape(BATCH_SIZE, 7, 7)
        Lb  = labels_dataset[idx]
        Yb  = vectorization(Xb)
        base_lr, _ = scheduler.get_lr(epoch, mode_ctrl.state)
        lr   = max(base_lr * mode_ctrl.get_lr_multiplier(), _cfg_global.training_cfg.learning_rate_min)
        temp = T_START * (T_END / T_START) ** (epoch / MAX_EPOCHS)
        params, opt_state, lv = step(params, opt_state, Xb, Yb, Lb, lk, lr, temp)
        if epoch % 10 == 0:
            loss_hist.add(epoch, {"total": float(lv)}, mode_ctrl.state.mode, lr)
            mode_ctrl.update(float(lv), base_lr, epoch)
            if float(lv) < best_loss:
                best_loss   = float(lv)
                best_params = params

    final = best_params if best_params is not None else params
    with open(ckpt_path, "wb") as f:
        pickle.dump({"params": final}, f)
    return final


# --- 8. Training - Native

def train_native(model_name: str, init_fn, native_loss_fn, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "best.pkl"
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            return pickle.load(f)["params"]

    key      = random.PRNGKey(GLOBAL_SEED)
    key, ik  = random.split(key)
    params   = init_fn(ik)

    schedule  = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=MAX_EPOCHS)
    opt_core  = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
    opt_state = opt_core.init(params)

    @jit
    def train_step(params, opt_state, X, L, key):
        def loss_fn(p):
            B    = X.shape[0]
            Xf   = X.reshape(B, 49)
            keys = random.split(key, B)
            return jnp.mean(jax.vmap(native_loss_fn, in_axes=(None, 0, 0, 0))(p, Xf, L, keys))
        loss, grads = jax.value_and_grad(loss_fn)(params)
        upd, ns     = opt_core.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), ns, loss

    best_loss, best_params = float("inf"), None
    for epoch in tqdm(range(MAX_EPOCHS), desc=f"[Native] {model_name}", leave=False):
        key, bk, lk = random.split(key, 3)
        idx = random.choice(bk, len(images_dataset), (BATCH_SIZE,), replace=False)
        Xb  = images_dataset[idx]
        Lb  = labels_dataset[idx]
        params, opt_state, lv = train_step(params, opt_state, Xb, Lb, lk)
        if float(lv) < best_loss:
            best_loss   = float(lv)
            best_params = params

    final = best_params if best_params is not None else params
    with open(ckpt_path, "wb") as f:
        pickle.dump({"params": final}, f)
    return final


# --- 9. Sample Generation

def gen_classical(params, gen_fn, digit: int, n: int) -> np.ndarray:
    key  = random.PRNGKey(GLOBAL_SEED + digit * 10_000)
    keys = random.split(key, n)
    return np.array(jax.vmap(lambda k: gen_fn(params, k, digit))(keys))

def _gen_ours_digit(pkl_path: Path, digit: int, n: int):
    if not _HQRNN_OK:
        return digit, np.zeros((0, 49), dtype=np.float32)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    params = data["params"]
    cfg    = Config(model=3)
    saved  = data.get("config", {})
    if "model_cfg" in saved:
        for k, v in saved["model_cfg"].items():
            setattr(cfg.model_cfg, k, v)
    q_model = QuantumModel(cfg)
    mdl     = cfg.model_cfg
    D, T    = mdl.n_D, mdl.seq_len
    powers  = 2 ** jnp.arange(D - 1, -1, -1)

    @partial(jit, static_argnames=["label_idx"])
    def generate_one(key, label_idx):
        h = jnp.zeros(2 ** mdl.n_H, dtype=jnp.complex64).at[0].set(1.0 + 0.0j)
        def body(t, state):
            h, x, key, rows = state
            key, sk = random.split(key)
            probs  = q_model.circuit_for_probs(params, h, x, label_idx)
            probs  = jnp.clip(probs, 0.0, 1.0) / (jnp.sum(jnp.clip(probs, 0.0, 1.0)) + 1e-9)
            idx    = random.categorical(sk, jnp.log(probs + 1e-9))
            bits   = (jnp.floor_divide(idx, powers) % 2).astype(jnp.int32)
            psi    = jnp.reshape(q_model.circuit_for_state(params, h, x, label_idx), (2 ** mdl.n_H, 2 ** D))
            h_u    = psi[:, idx].astype(jnp.complex64)
            norm   = jnp.linalg.norm(h_u)
            h_next = jnp.where(norm > 1e-9, h_u / norm, h).astype(jnp.complex64)
            return h_next, bits.astype(jnp.float32), key, rows.at[t].set(bits)
        _, _, _, rows = lax.fori_loop(
            0, T, body,
            (h, jnp.zeros(D, jnp.float32), key, jnp.zeros((T, D), jnp.int32)))
        return rows.flatten()

    base = random.PRNGKey(GLOBAL_SEED + digit * 10_000)
    out  = []
    for _ in tqdm(range(n), desc=f"  gen {pkl_path.stem} d{digit}", leave=False):
        k, base = random.split(base)
        out.append(generate_one(k, digit))
    return digit, np.array(jnp.stack(out))


# --- 10. CNN Discriminator

class CNN7x7(nn.Module):
    @nn.compact
    def __call__(self, x, training=False):
        x = jnp.pad(x, ((0,0),(0,1),(0,1),(0,0)), mode="constant")
        x = nn.relu(nn.Conv(32, (3,3), padding="SAME")(x))
        x = nn.avg_pool(x, (2,2), (2,2))
        x = nn.relu(nn.Conv(64, (3,3), padding="SAME")(x))
        x = nn.avg_pool(x, (2,2), (2,2))
        x = nn.relu(nn.Dense(128)(x.reshape((x.shape[0], -1))))
        return nn.Dense(1)(x)

def train_cnn(X_tr, Y_tr, epochs=50):
    model  = CNN7x7()
    key_c  = random.PRNGKey(GLOBAL_SEED)
    params = model.init(key_c, jnp.ones((1,7,7,1)), training=True)
    state  = flax_train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(0.001))

    @jax.jit
    def step(s, imgs, lbls):
        def lf(p):
            return optax.sigmoid_binary_cross_entropy(
                s.apply_fn(p, imgs, training=True).squeeze(-1),
                lbls.astype(jnp.float32)).mean()
        loss, g = jax.value_and_grad(lf)(s.params)
        return s.apply_gradients(grads=g), loss

    for _ in tqdm(range(epochs), desc="CNN train"):
        key_c, sk = random.split(key_c)
        perm = random.permutation(sk, len(X_tr))
        for i in range(0, len(X_tr), 32):
            state, _ = step(state, X_tr[perm[i:i+32]], Y_tr[perm[i:i+32]])
    return state

def cnn_predict(state, images):
    return np.array(jax.nn.sigmoid(
        state.apply_fn(state.params, images, training=False).squeeze(-1)))

def _load_or_train_cnn(ckpt: Path, X_tr, Y_tr):
    if ckpt.exists():
        _m = CNN7x7()
        _p = _m.init(random.PRNGKey(GLOBAL_SEED), jnp.ones((1,7,7,1)), training=True)
        with open(ckpt, "rb") as f:
            _p = {**_p, "params": pickle.load(f)["params"]}
        return flax_train_state.TrainState.create(apply_fn=_m.apply, params=_p, tx=optax.adam(0.001))
    state = train_cnn(jnp.array(X_tr), jnp.array(Y_tr))
    with open(ckpt, "wb") as f:
        pickle.dump({"params": state.params["params"]}, f)
    return state


# --- 11. Metrics

def _binarise(arr: np.ndarray) -> np.ndarray:
    return (arr > 0.5).astype(np.float32)

def metric_cnn_accuracy(samples, labels, cnn_state) -> float:
    probs = cnn_predict(cnn_state, samples.reshape(-1, 7, 7, 1))
    return float(np.mean((probs > 0.5).astype(int) == labels) * 100)

def metric_bfd(real: np.ndarray, gen: np.ndarray) -> float:
    r = _binarise(real); g = _binarise(gen)
    eps = 1e-6
    m1, m2   = r.mean(0), g.mean(0)
    s1 = np.cov(r, rowvar=False) + np.eye(r.shape[1]) * eps
    s2 = np.cov(g, rowvar=False) + np.eye(g.shape[1]) * eps
    diff     = m1 - m2
    cov_m, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if np.isnan(cov_m).any():
        cov_m, _ = linalg.sqrtm(s1.dot(s2) + np.eye(s1.shape[0]) * eps, disp=False)
    if np.iscomplexobj(cov_m):
        cov_m = cov_m.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(cov_m))

def metric_hamming_diversity(gen: np.ndarray, labels: np.ndarray) -> float:
    gb = _binarise(gen)
    divs = []
    for cls in [0, 1]:
        sub = gb[labels == cls]
        if len(sub) < 2: continue
        n   = min(1000, len(sub))
        idx = np.random.choice(len(sub), n, replace=False)
        pw  = cdist(sub[idx], sub[idx], metric="hamming")
        divs.append(np.mean(pw[np.triu_indices(n, k=1)]) * 100)
    return float(np.mean(divs)) if divs else 0.0

def metric_uniqueness(gen: np.ndarray) -> float:
    packed = np.packbits(_binarise(gen).astype(np.uint8), axis=1)
    return float(len({row.tobytes() for row in packed}) / len(packed) * 100)

def metric_novelty(real: np.ndarray, gen: np.ndarray) -> float:
    def to_set(arr):
        return {row.tobytes() for row in np.packbits(_binarise(arr).astype(np.uint8), axis=1)}
    real_set   = to_set(real)
    gen_packed = np.packbits(_binarise(gen).astype(np.uint8), axis=1)
    return float(sum(1 for row in gen_packed if row.tobytes() not in real_set) / len(gen_packed) * 100)

def metric_recall(real: np.ndarray, gen: np.ndarray, k: int = 5, n: int = 2000) -> float:
    rb = _binarise(real); gb = _binarise(gen)
    nr = min(n, len(rb)); ng = min(n, len(gb))
    R  = rb[np.random.choice(len(rb), nr, replace=False)]
    G  = gb[np.random.choice(len(gb), ng, replace=False)]
    rr = cdist(R, R, metric="hamming") * 49
    rg = cdist(R, G, metric="hamming") * 49
    return float(np.mean(rg.min(axis=1) <= np.sort(rr, axis=1)[:, k]) * 100)

def compute_all_metrics(samples, labels, cnn_state) -> dict:
    if samples.shape[1] != 49:
        samples = (samples[:, :49] if samples.shape[1] > 49
                   else np.pad(samples, ((0,0),(0, 49 - samples.shape[1]))))
    cnn_acc = metric_cnn_accuracy(samples, labels, cnn_state)
    bfd     = metric_bfd(DATASET_FLAT, samples)
    ham_div = metric_hamming_diversity(samples, labels)
    uniq    = metric_uniqueness(samples)
    novelty = metric_novelty(DATASET_FLAT, samples)
    recall  = metric_recall(DATASET_FLAT, samples)
    hmean   = 2 * recall * novelty / (recall + novelty + 1e-9)
    return {
        "cnn_acc":     cnn_acc,
        "bfd":         bfd,
        "hamming_div": ham_div,
        "uniqueness":  uniq,
        "novelty":     novelty,
        "recall":      recall,
        "hmean_r_n":   hmean,
    }

METRIC_LABELS = {
    "cnn_acc":     "CNN Acc (%)",
    "bfd":         "BFD down",
    "hamming_div": "Hamming Div (%)",
    "uniqueness":  "Uniqueness (%)",
    "novelty":     "Novelty (%)",
    "recall":      "Recall (%)",
    "hmean_r_n":   "H-Mean(R,N)",
}
METRIC_HB = {
    "cnn_acc": True, "bfd": False, "hamming_div": True,
    "uniqueness": True, "novelty": True, "recall": True, "hmean_r_n": True,
}


# --- 12. Visualisation

def _kde_stack(images: np.ndarray, size=500, scale=60.0, sigma=22.5) -> np.ndarray:
    canvas = np.zeros((size, size), dtype=np.float32)
    c      = size / 2.0
    yg, xg = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    inv2s2 = 1.0 / (2.0 * sigma ** 2)
    for img in images:
        m2 = img.reshape(7, 7)
        if m2.sum() == 0: continue
        cy, cx = ndimage.center_of_mass(m2)
        if np.isnan(cy): continue
        active = np.argwhere(m2 > 0.5)
        if len(active) == 0: continue
        ty  = c + (active[:, 0] - cy) * scale
        tx  = c + (active[:, 1] - cx) * scale
        dy2 = (yg[:, :, None] - ty[None, None, :]) ** 2
        dx2 = (xg[:, :, None] - tx[None, None, :]) ** 2
        canvas += np.exp(-(dy2 + dx2) * inv2s2).sum(axis=2)
    if canvas.max() > 0:
        canvas = np.power(canvas / canvas.max(), 4.0)
    return canvas

def plot_kde(gen_data: dict, out_path: Path, title: str = ""):
    names   = list(gen_data.keys())
    n       = len(names)
    fig, ax = plt.subplots(2, n, figsize=(max(4 * n, 8), 9))
    if n == 1: ax = ax.reshape(2, 1)
    if title: fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)

    def _task(args):
        col, name, row = args
        imgs = gen_data[name]["d0" if row == 0 else "d1"][:KDE_SAMPLES]
        return col, row, _kde_stack(imgs)

    tasks = [(col, nm, row) for col, nm in enumerate(names) for row in [0, 1]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=_N_CPU) as pool:
        kdes = {(c, r): kde for c, r, kde in pool.map(_task, tasks)}

    for col, nm in enumerate(names):
        for row in [0, 1]:
            a = ax[row, col]
            a.imshow(kdes[(col, row)], cmap="Greys_r", origin="upper")
            a.axis("off")
            if row == 0:
                a.set_title(nm, fontsize=13, fontweight="bold", pad=8)

    LEFT = 0.05
    fig.text(LEFT - 0.01, 0.74, "Digit 0", va="center", ha="right",
             rotation="vertical", fontsize=18, fontweight="bold")
    fig.text(LEFT - 0.01, 0.26, "Digit 1", va="center", ha="right",
             rotation="vertical", fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[LEFT, 0, 1, 1], pad=1.5)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

def plot_metrics_bar(results: dict, out_path: Path, title: str = "", highlight: str = "Ours"):
    keys  = list(METRIC_LABELS.keys())
    names = list(results.keys())
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    if title: fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    cmap = plt.cm.tab10(np.linspace(0, 1, len(names)))
    n2c  = dict(zip(names, cmap))
    if highlight in n2c:
        n2c[highlight] = np.array([0.85, 0.15, 0.15, 1.0])

    for idx, k in enumerate(keys):
        r, c = idx // 4, idx % 4
        ax   = axes[r, c]
        vals = [results[n].get(k, 0.0) for n in names]
        bars = ax.bar(range(len(names)), vals, color=[n2c[n] for n in names])
        ax.set_title(METRIC_LABELS[k], fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        best_v = (max if METRIC_HB[k] else min)(vals)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            if abs(v - best_v) < 1e-9:
                bar.set_edgecolor("gold"); bar.set_linewidth(2.5)

    axes[1, 3].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --- 13. Save Experiment Outputs

def save_experiment(exp_dir: Path, results: dict, gen_data: dict,
                    title: str, kde_filename: str = "kde_stack.png"):
    df = pd.DataFrame(results).T.rename(columns=METRIC_LABELS)
    df.to_csv(exp_dir / "results.csv")
    with open(exp_dir / "results.msgpack", "wb") as f:
        msgpack.pack(ensure_serializable(results), f)
    plot_metrics_bar(results, exp_dir / "metrics_bar.png", title=title)
    plot_kde(gen_data, exp_dir / kde_filename, title=title)


# --- 14. Parallel Workers

def _ablation_gen_worker(job: tuple) -> tuple:
    name, pkl_path = job
    _, d0 = _gen_ours_digit(pkl_path, 0, N_SAMPLES_PER_DIGIT)
    _, d1 = _gen_ours_digit(pkl_path, 1, N_SAMPLES_PER_DIGIT)
    return name, d0, d1

def _fair_train_gen_worker(job: tuple) -> tuple:
    name, save_dir_str = job
    init_fn, fwd_fn, gen_fn, _, aux_fn, aux_w = CLASSICAL_REGISTRY[name]
    params = train_fair(name, init_fn, fwd_fn, Path(save_dir_str), aux_fn, aux_w)
    d0     = gen_classical(params, gen_fn, 0, N_SAMPLES_PER_DIGIT)
    d1     = gen_classical(params, gen_fn, 1, N_SAMPLES_PER_DIGIT)
    return name, d0, d1

def _native_train_gen_worker(job: tuple) -> tuple:
    name, save_dir_str = job
    init_fn, _, gen_fn, native_loss_fn, _, _ = CLASSICAL_REGISTRY[name]
    params = train_native(name, init_fn, native_loss_fn, Path(save_dir_str))
    d0     = gen_classical(params, gen_fn, 0, N_SAMPLES_PER_DIGIT)
    d1     = gen_classical(params, gen_fn, 1, N_SAMPLES_PER_DIGIT)
    return name, d0, d1

def _metrics_worker(job: tuple) -> tuple:
    tag, name, smp, lbl, cnn_state = job
    return tag, name, compute_all_metrics(smp, lbl, cnn_state)


# --- 15. Main

def main():
    train_imgs = np.array(images_dataset).reshape(-1, 7, 7, 1)
    train_lbls = np.array(labels_dataset)
    cnn_state  = _load_or_train_cnn(EXP_COMP / "CNN.pkl", train_imgs, train_lbls)

    if not HQRNN_CKPT.exists():
        raise FileNotFoundError(f"HQRNN checkpoint not found: {HQRNN_CKPT}")

    abl_jobs = [("Ours", HQRNN_CKPT)]
    for aname, apath in ABLATION_CKPTS.items():
        if apath.exists():
            abl_jobs.append((aname, apath))

    abl_raw: dict[str, tuple] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(abl_jobs)) as ex:
        fs = {ex.submit(_ablation_gen_worker, job): job[0] for job in abl_jobs}
        for f in tqdm(concurrent.futures.as_completed(fs), total=len(abl_jobs), desc="Phase 1 | Abl+Ours gen"):
            try:
                name, d0, d1 = f.result()
                abl_raw[name] = (d0, d1)
            except Exception as e:
                print(f"ERROR generating {fs[f]}: {e}")

    if "Ours" not in abl_raw:
        raise RuntimeError("'Ours' generation failed.")

    ours_d0, ours_d1 = abl_raw["Ours"]
    ours_samples   = np.concatenate([ours_d0, ours_d1])
    ours_labels    = np.array([0] * len(ours_d0) + [1] * len(ours_d1))
    ours_gen_entry = {"d0": ours_d0, "d1": ours_d1, "samples": ours_samples, "labels": ours_labels}

    n_models    = len(CLASSICAL_REGISTRY)
    fair_jobs   = [(n, str(EXP_COMP / n))         for n in CLASSICAL_REGISTRY]
    native_jobs = [(n, str(NATIVE_CKPT_DIR / n))  for n in CLASSICAL_REGISTRY]

    fair_raw:   dict[str, tuple] = {}
    native_raw: dict[str, tuple] = {}

    futures_map: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2 * n_models) as ex:
        for job in fair_jobs:
            futures_map[ex.submit(_fair_train_gen_worker, job)]   = ("fair",   job[0])
        for job in native_jobs:
            futures_map[ex.submit(_native_train_gen_worker, job)] = ("native", job[0])

        for f in tqdm(concurrent.futures.as_completed(futures_map), total=len(futures_map), desc="Phase 2 | Train+Gen"):
            exp_type, expected_name = futures_map[f]
            try:
                name, d0, d1 = f.result()
                (fair_raw if exp_type == "fair" else native_raw)[name] = (d0, d1)
            except Exception as e:
                print(f"ERROR [{exp_type}] {expected_name}: {e}")

    ours_metrics = compute_all_metrics(ours_samples, ours_labels, cnn_state)

    metric_jobs = []
    for name, (d0, d1) in fair_raw.items():
        smp = np.concatenate([d0, d1])
        metric_jobs.append(("fair", name, smp, np.array([0]*len(d0) + [1]*len(d1)), cnn_state))
    for name, (d0, d1) in native_raw.items():
        smp = np.concatenate([d0, d1])
        metric_jobs.append(("native", name, smp, np.array([0]*len(d0) + [1]*len(d1)), cnn_state))
    for name, (d0, d1) in abl_raw.items():
        if name == "Ours": continue
        smp = np.concatenate([d0, d1])
        metric_jobs.append(("ablation", name, smp, np.array([0]*len(d0) + [1]*len(d1)), cnn_state))

    metrics_all = {
        "fair":     {"Ours": ours_metrics},
        "native":   {"Ours": ours_metrics},
        "ablation": {"Ours": ours_metrics},
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(metric_jobs), _N_CPU)) as ex:
        fs = {ex.submit(_metrics_worker, job): (job[0], job[1]) for job in metric_jobs}
        for f in tqdm(concurrent.futures.as_completed(fs), total=len(fs), desc="Phase 3 | Metrics"):
            try:
                tag, name, m = f.result()
                metrics_all[tag][name] = m
            except Exception as e:
                tag, name = fs[f]
                print(f"ERROR metrics [{tag}] {name}: {e}")

    fair_results = metrics_all["fair"]
    fair_gen     = {"Ours": ours_gen_entry}
    for name, (d0, d1) in fair_raw.items():
        smp = np.concatenate([d0, d1])
        fair_gen[name] = {"d0": d0, "d1": d1, "samples": smp, "labels": np.array([0]*len(d0) + [1]*len(d1))}

    save_experiment(
        EXP_COMP,
        {n: fair_results[n] for n in MODEL_ORDER if n in fair_results},
        {n: fair_gen[n]     for n in MODEL_ORDER if n in fair_gen},
        title="Exp 1: Fair Comparison",
        kde_filename="kde_stack_fair.png",
    )

    native_gen = {"Ours": ours_gen_entry}
    for name, (d0, d1) in native_raw.items():
        smp = np.concatenate([d0, d1])
        native_gen[name] = {"d0": d0, "d1": d1, "samples": smp, "labels": np.array([0]*len(d0) + [1]*len(d1))}

    plot_kde(
        {n: native_gen[n] for n in MODEL_ORDER if n in native_gen},
        EXP_COMP / "kde_stack_native.png",
        title="Exp 2: Native Comparison",
    )

    abl_results = dict(metrics_all["ablation"])
    abl_gen     = {"Ours": ours_gen_entry}
    for name, (d0, d1) in abl_raw.items():
        if name == "Ours": continue
        smp = np.concatenate([d0, d1])
        abl_gen[name] = {"d0": d0, "d1": d1, "samples": smp, "labels": np.array([0]*len(d0) + [1]*len(d1))}

    abl_order = ["Ours"] + [n for n in ABLATION_CKPTS if n in abl_gen]
    save_experiment(
        EXP_ABL,
        {n: abl_results[n] for n in abl_order if n in abl_results},
        {n: abl_gen[n]     for n in abl_order if n in abl_gen},
        title="Exp 3: Ablation Study",
    )


if __name__ == "__main__":
    main()
