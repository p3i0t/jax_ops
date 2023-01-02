from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
from time import perf_counter
import timeit
from functools import partial
from loguru import logger

from ops.rolling_ops import (rolling_corr, rolling_kurtosis, rolling_regbeta, rolling_skew,
                         rolling_std, rolling_var, rolling_robust_zscore, rolling_zscore,
                         rolling_max, rolling_mean, rolling_min, rolling_rank_corr, ema, sma)

from ops import zscore, rank


def memory_used(x: np.ndarray):
    return np.prod(x.shape) * 4


def cal_n_split(x: np.ndarray, window: int, n_args: int = 1):
    memory_total_3090 = 16 * 10**9
    tensor_memory_ratio = 0.5

    tensor_memory_allowed = memory_total_3090 * tensor_memory_ratio / n_args
    m_total = memory_used(x) * window
    # at least 1
    return max(1, m_total // tensor_memory_allowed + 1)



def bench_unary_ops(x: jnp.DeviceArray, func: Callable):
    n_times = 10
    return timeit.timeit(lambda: func(x), number=n_times) / n_times


if __name__ == "__main__":
    n_date = 1000
    n_time = 240
    n_symbol = 4000
    window = 30

    x = np.random.randn(n_date, n_time, n_symbol)
    x_fake = np.random.randn(100, 20, 100)
    y = np.random.randn(n_date, n_time, n_symbol)

    s = perf_counter()
    x = jnp.array(x)
    t = perf_counter()
    logger.info(f"transfer x from CPU to GPU: {t-s:.4f}s")


    n_split = cal_n_split(x, window=window, n_args=1)
    print(f"unary op: {n_split=}")

    p_ema = partial(ema, axis=0, window=window, n_split=n_split)
    dt = bench_unary_ops(x, p_ema)
    logger.info(f"ema: {dt:.4f}s")

    p_sma = partial(sma, axis=0, window=window, n_split=n_split)
    dt = bench_unary_ops(x, p_sma)
    logger.info(f"sma: {dt:.4f}s")

    p_std = partial(rolling_std, axis=0, window=window, n_split=n_split)
    dt = bench_unary_ops(x, p_std)
    logger.info(f"rolling_std: {dt:.4f}s")

    p_var = partial(rolling_var, axis=0, window=window, n_split=n_split)
    dt = bench_unary_ops(x, p_var)
    logger.info(f"rolling_var: {dt:.4f}s")

    p_max = partial(rolling_max, axis=0, window=window, n_split=n_split)
    dt = bench_unary_ops(x, p_max)
    logger.info(f"rolling_max: {dt:.4f}s")

    p_min = partial(rolling_min, axis=0, window=window, n_split=n_split)
    dt = bench_unary_ops(x, p_min)
    logger.info(f"rolling_min: {dt:.4f}s")

    p_skew = partial(rolling_skew, axis=0, window=window, n_split=n_split)
    dt = bench_unary_ops(x, p_skew)
    logger.info(f"rolling_skew: {dt:.4f}s")

    p_kurtosis = partial(rolling_kurtosis, axis=0, window=window, n_split=n_split)
    dt = bench_unary_ops(x, p_kurtosis)
    logger.info(f"rolling_kurtosis: {dt:.4f}s")

    # y = jnp.array(y)
    # n_split = cal_n_split(x, window=window, n_args=2)
    # # p_corr = partial(rolling_corr, axis=0, window=window, n_split=n_split)

    # logger.info(f"binary op: {n_split}")

    # s = perf_counter()
    # o = rolling_corr(x, y, axis=0, window=window, n_split=n_split)
    # t = perf_counter()
    # # n_times = 10
    # # dt = timeit.timeit(lambda: p_corr(x, y), number=n_times) / n_times
    # logger.info(f"rolling_corr: {t-s:.4f}s")



