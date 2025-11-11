import jax
from jax import random

# --- 11. Seed Management

_main_key = random.PRNGKey(42)

def set_seed(seed: int):
    global _main_key
    _main_key = random.PRNGKey(seed)

def get_key():
    global _main_key
    _main_key, subkey = random.split(_main_key)
    return subkey
