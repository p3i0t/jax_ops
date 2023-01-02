from typing import Tuple, Union, Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap


def _rank1d(x: jnp.DeviceArray):
    "1d pct rank for internal use only."
    n = jnp.count_nonzero(~jnp.isnan(x))
    tmp = jnp.argsort(x)
    r = jnp.empty_like(x)
    # range starts from 1 to align with pct rank in pd.Dataframe
    r = r.at[tmp].set(jnp.arange(1, len(x) + 1))
    r = r / n
    r = jnp.where(jnp.isnan(x), float('nan'), r)
    return r


@partial(jit, static_argnums=(1,))
def rank(x: jnp.DeviceArray, axis=-1):
    """functional implementation of rank."""
    return jnp.apply_along_axis(_rank1d, axis=axis, arr=x)


@partial(jit, static_argnums=(1, 2, 3))
def zscore(x: jnp.DeviceArray, axis=-1, ddof=1, eps: float = 1e-5):
    """compute z_score on `axis`."""
    n = jnp.count_nonzero(~jnp.isnan(x))
    mu = jnp.nanmean(x, axis=axis, keepdims=True)
    std = jnp.nanstd(x, axis=axis, keepdims=True, ddof=ddof) + eps
    o = (x - mu) / std
    # at least 2 elements required.
    o = jnp.where(n < 2, float('nan'), o)
    return o


@partial(jit, static_argnums=(1, 2))
def robust_zscore(x: jnp.DeviceArray, axis=-1, eps: float = 1e-5):
    """Robust ZScore Normalization
    Use robust statistics for Z-Score normalization:
        mean(x) = median(x)
        std(x) = MAD(x) * 1.4826
    Reference:
        https://en.wikipedia.org/wiki/Median_absolute_deviation.
    """
    mu = jnp.nanmean(x, axis=axis, keepdims=True)
    x_demean = x - mu
    demean_abs = jnp.abs(x_demean)
    mad = jnp.nanmedian(demean_abs)
    _std = mad * 1.4826 + eps

    z = jnp.clip(x_demean / _std, -3.0, 3.0)
    return z


@partial(jit, static_argnums=(1, 2))
def skew(x: jnp.DeviceArray, axis=-1, bias=True):
    """
    Skewness along given axis. The implementation here
    aligns to scipy.stats.skew. Note that the difference between bias True and False
    is not simply the difference of ddof 1 and 0.
    see https://github.com/scipy/scipy/blob/v1.9.0/scipy/stats/_stats_py.py#L1254-L1357.
    Also, pd.DataFrame.skew is unbiased by default.
    See https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.skew.html?highlight=skew#pandas.DataFrame.skew.
    The statement `Normalized by N-1` in the above link is kind of misleading.
    """
    mu = jnp.nanmean(x, axis=axis, keepdims=True)
    std = jnp.nanstd(x, axis=axis, ddof=0)

    o = jnp.nanmean((x - mu)**3, axis=axis) / std ** 3
    n = jnp.count_nonzero(~jnp.isnan(x), axis=axis)
    if bias is False:
        # correction
        r = jnp.sqrt(n * (n - 1)) / (n-2)
        o = r * o
    # at least 3 elements for skew.
    o = jnp.where(n < 3, float('nan'), o)
    return o


@partial(jit, static_argnums=(1, 2))
def kurtosis(x: jnp.DeviceArray, axis=-1, bias=True):
    """
    (biased) kurtosis along given axis (adjusted by -3). Ref:
    https://en.wikipedia.org/wiki/Kurtosis for unbiased kurtosis.
    """
    n = jnp.count_nonzero(~jnp.isnan(x), axis=axis)
    if bias is True:
        mu = jnp.nanmean(x, axis=axis, keepdims=True)
        std = jnp.nanstd(x, axis=axis, ddof=0)

        o = jnp.nanmean((x - mu)**4, axis=axis) / std ** 4 - 3.0
    else:
        # almost completely different, to be optimized possibly.

        x_demean = x - jnp.nanmean(x, axis=axis, keepdims=True)
        o = jnp.nansum(x_demean ** 4, axis=axis) / jnp.nansum(x_demean ** 2, axis=axis) ** 2
        n_123 = (n - 1) / (n - 2) / (n - 3)

        o = o * (n+1) * n * n_123 - 3.0 * (n-1) * n_123

    # at least 4 elements for kurtosis
    o = jnp.where(n < 4, float('nan'), o)
    return o


@partial(jit, static_argnums=(2,))
def pearsonr(x: jnp.DeviceArray, y: jnp.DeviceArray, axis=-1) -> jnp.DeviceArray:
    """compute the correlation of x and y (of same shape) along `axis`."""

    # critical
    # note that use jnp.where is JIT-compatible.
    x = jnp.where(jnp.isnan(y), float('nan'), x)
    y = jnp.where(jnp.isnan(x), float('nan'), y)

    # the following at[].set() call will trigger NonConcreteBoolean Error.
    # common_nan_idx = jnp.where(jnp.isnan(x*y), True, False)
    # # common_nan_idx = jnp.logical_or(jnp.isnan(x), jnp.isnan(y))
    # x = x.at[common_nan_idx].set(float('nan'))
    # y = y.at[common_nan_idx].set(float('nan'))

    ddof = 0
    x_z = zscore(x, axis=axis, ddof=ddof)
    y_z = zscore(y, axis=axis, ddof=ddof)

    xy_ele_prod = x_z * y_z
    corr = jnp.nanmean(xy_ele_prod, axis=axis)
    return corr


@partial(jit, static_argnums=(2,))
def spearmanr(x: jnp.DeviceArray, y: jnp.DeviceArray, axis=-1) -> jnp.DeviceArray:
    """compute rank IC of x, y along `axis`."""
    x = jnp.where(jnp.isnan(y), float('nan'), x)
    y = jnp.where(jnp.isnan(x), float('nan'), y)

    x_rank = rank(x, axis=axis)
    y_rank = rank(y, axis=axis)

    ddof = 0
    x_z = zscore(x_rank, axis=axis, ddof=ddof)
    y_z = zscore(y_rank, axis=axis, ddof=ddof)

    xy_ele_prod = x_z * y_z
    corr = jnp.nanmean(xy_ele_prod, axis=axis)
    return corr


@partial(jit, static_argnums=(2,))
def regbeta(x: jnp.DeviceArray, y: jnp.DeviceArray, axis=-1):
    """Evaluate \beta in regression problem y = \beta x + \alpha, where
    y, x are two vectors and \beta, \alpha are scalars. Note that we don't
    give special treatment to NaNs here, which is particularly
    troublesome in this case. This is aligned to OLS in `statsmodels`,
    no NaNs and infs are allowed."""
    # # nan alignment.
    # nan_idx = jnp.logical_or(jnp.isnan(x), jnp.isnan(y))
    # x = jnp.where(nan_idx, float('nan'), x)
    # y = jnp.where(nan_idx, float('nan'), y)
    x_demean = x - jnp.mean(x, axis=axis, keepdims=True)
    y_demean = y - jnp.mean(y, axis=axis, keepdims=True)
    # covar and var without scaling
    covar = jnp.sum(x_demean * y_demean, axis=axis)
    var = jnp.sum(x_demean ** 2, axis=axis)
    return covar / var