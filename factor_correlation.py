from typing import Union
import numpy as np
import jax
from jax import jit, vmap, pmap
import jax.numpy as jnp
from time import perf_counter

Tensor = Union[np.ndarray, jnp.DeviceArray]

def zscore_1d(x: Tensor):
    "x is 1d tensor."
    mu = jnp.nanmean(x)
    std = jnp.nanstd(x)
    o = (x - mu) / std
    return o


def pearsonr_1d(x: Tensor, y: Tensor):
    """
    compute the pearson correlation of 1d tensors x and y.
    """
    x = jnp.where(jnp.isnan(y), float('nan'), x)
    y = jnp.where(jnp.isnan(x), float('nan'), y)

    xz = zscore_1d(x)
    yz = zscore_1d(y)
    xy_prod = xz * yz
    corr = jnp.nanmean(xy_prod)
    return corr


@jit
def lbatch_corr(batched_x, y):
    """auto vectorization of left batched x.
    batched_x: (n_factor, n_symbol)
    y: (n_symbol)
    """
    # (b, a), (a) -> (b)
    f = vmap(pearsonr_1d, (0, None), 0)
    return f(batched_x, y)


@jit
def rbatch_corr(x, batched_y):
    """auto vectorization of right batched y.
    x: (n_factor, n_symbol)
    batched_y: (n_factor, n_symbol)
    """
    # (a, b), (c, b) -> (a, c)
    f = vmap(lbatch_corr, (None, 0), 1)
    return f(x, batched_y)


@jit
def batched_corr(x, y):
    """auto vectorization of both batched x and y.
    both x and y are in shape: (n_date, n_factor, n_symbol).
    output in shape: (n_date, n_factor, n_factor)
    """
    # (a, b, c), (a, b, c) -> (a, b, b)
    f = vmap(rbatch_corr, (0, 0), 0)
    return f(x, y)

# @jit
# def parallel_batched_corr(x, y):
#     """Data parallel across multiple devices(GPUs).
#     """
#     return pmap(batched_corr)(x, y)


if __name__ == "__main__":
    n_date = 50
    n_factor = 2000
    n_symbol = 4000
    a = np.random.randn(n_date, n_factor, n_symbol).astype(np.float32)
    #c = np.random.randn(n_date, n_factor, n_symbol)
    #b = np.random.randn(n_symbol)
    #print(f"{a.shape=}, {b.shape=}, {c.shape=}")
    print(f"{a.shape=}")
    fake_a = np.random.randn(20,5,5)
    # o = lbatch_corr(a, b).block_until_ready()
    o = batched_corr(fake_a, fake_a).block_until_ready()


    s = perf_counter()
    o = batched_corr(a, a).block_until_ready()
    t = perf_counter()
    print(f"output shape: {o.shape}")
    print(f"time elapsed: {(t-s):2f}s")
