from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from time import perf_counter
import timeit
from functools import partial
from loguru import logger


from ops import rolling_corr, rolling_kurtosis, ema

if __name__ == "__main__":
    n_rows = 6000
    n_cols = 4000
    x = np.random.randn(n_rows, n_cols)
    y = np.random.randn(n_rows, n_cols)

    dfx = pd.DataFrame(x)
    dfy = pd.DataFrame(y)

    window = 30
    n_times = 5
    pd_rolling_corr_dt = timeit.timeit(lambda: dfx.rolling(window=window, min_periods=1).corr(dfy), number=n_times) / n_times
    logger.info(f'pd rolling corr time elapsed: {pd_rolling_corr_dt:.4f}s')

    pd_rolling_kurt_dt = timeit.timeit(lambda: dfx.rolling(window=window, min_periods=1).kurt(), number=n_times) / n_times
    logger.info(f'pd rolling kurt time elapsed: {pd_rolling_kurt_dt:.4f}s')

    s = perf_counter()
    x = jnp.array(x)
    y = jnp.array(y)
    t = perf_counter()
    logger.info(f"transfer x, y from CPU to GPU: {t-s:.4f}s")
    # ema(x_fake, axis=1, window=window, alpha=0.5)
    # warmup
    rolling_corr_dt = timeit.timeit(lambda: rolling_corr(x, y, axis=0, window=window), number=n_times) / n_times
    logger.info(f'jax rolling corr time elapsed: {rolling_corr_dt:.4f}s')

    rolling_kurt_dt = timeit.timeit(lambda: rolling_kurtosis(x, axis=0, window=window, bias=False), number=n_times) / n_times
    logger.info(f'jax rolling_kurt time elapsed: {rolling_kurt_dt:.4f}s')
