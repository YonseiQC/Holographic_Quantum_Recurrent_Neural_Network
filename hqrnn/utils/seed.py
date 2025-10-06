import jax
from jax import random

# 프로젝트 전체에서 일관된 시드를 제공하기 위한 마스터 키
_main_key = random.PRNGKey(42)

def get_key():
    """
    마스터 키를 분할하여 새로운 JAX PRNG 키를 생성하고 반환합니다.
    이를 통해 모든 무작위 연산이 재현 가능하도록 보장합니다.
    """
    global _main_key
    _main_key, subkey = random.split(_main_key)
    return subkey