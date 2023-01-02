from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
from time import perf_counter
import timeit
from functools import partial
from loguru import logger


from ops import zscore, rank

def _np_rank1d(x: np.ndarray):
    "1d pct rank for internal use only."
    n = np.count_nonzero(~np.isnan(x))
    tmp = np.argsort(x)
    r = np.empty_like(x)
    # range starts from 1 to align with pct rank in pd.Dataframe
    r[tmp] = np.arange(1, len(x) + 1)
    r = r / n
    r = np.where(np.isnan(x), float('nan'), r)
    return r


def np_rank(x: np.ndarray, axis=-1):
    """functional implementation of rank."""
    return np.apply_along_axis(_np_rank1d, axis=axis, arr=x)


def np_zscore(x: np.ndarray, axis=-1, ddof=1, eps: float = 1e-5):
    """compute z_score on `axis`."""
    n = np.count_nonzero(~np.isnan(x))
    mu = np.nanmean(x, axis=axis, keepdims=True)
    std = np.nanstd(x, axis=axis, keepdims=True, ddof=ddof) + eps
    o = (x - mu) / std
    # at least 2 elements required.
    o = np.where(n < 2, float('nan'), o)
    return o


if __name__ == "__main__":
    n_date = 1000
    n_time = 240
    n_symbol = 4000
    # window = 30

    x = np.random.randn(n_date, n_time, n_symbol)
    x_fake = np.random.randn(100, 20, 100)
    y = np.random.randn(n_date, n_time, n_symbol)

    n_times = 5
    # np_rank_dt = timeit.timeit(lambda: np_rank(x, axis=-1), number=n_times) / n_times
    # logger.info(f'np_rank time elapsed: {np_rank_dt:.4f}s')

    # np_zscore_dt = timeit.timeit(lambda: np_zscore(x, axis=-1), number=n_times) / n_times
    # logger.info(f'np_zscore time elapsed: {np_zscore_dt:.4f}s')

    s = perf_counter()
    x = jnp.array(x)
    t = perf_counter()
    logger.info(f"transfer x from CPU to GPU: {t-s:.4f}s")
    # ema(x_fake, axis=1, window=window, alpha=0.5)
    # warmup
    rank(x_fake, axis=-1).block_until_ready()
    rank_dt = timeit.timeit(lambda: rank(x, axis=-1).block_until_ready(), number=n_times) / n_times
    logger.info(f'jax rank time elapsed: {rank_dt:.4f}s')

    zscore(x_fake, axis=-1).block_until_ready()
    zscore_dt = timeit.timeit(lambda: zscore(x, axis=-1).block_until_ready(), number=n_times) / n_times
    logger.info(f'zscore time elapsed: {zscore_dt:.4f}s')
