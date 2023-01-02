import unittest

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.stats
# from time import perf_counter

# import jax
from ops import rank, zscore, robust_zscore, spearmanr, pearsonr, skew, kurtosis, regbeta


class TestVanillaOps(unittest.TestCase):
    """This unit test aims to verify the logical correctness of the ops implemented in Jax. The correctness is
    guaranteed by comparison with the results of relevent implementations in Pandas, scipy or numpy."""
    def setUp(self) -> None:
        self.n_rows, self.n_cols = 1000, 20
        self.a = np.random.randn(self.n_rows, self.n_cols).astype(np.float32)
        self.b = np.random.randn(self.n_rows, self.n_cols).astype(np.float32)

        n_nan = np.random.randint(0, 50)
        for i, j in zip(np.random.randint(0, self.n_rows, n_nan), np.random.randint(0, self.n_cols, n_nan)):
            self.a[i, j] = float('nan')
        for i, j in zip(np.random.randint(0, self.n_rows, n_nan), np.random.randint(0, self.n_cols, n_nan)):
            self.b[i, j] = float('nan')

    def test_rank(self):
        cols = [f"col{i}" for i in range(self.n_cols)]
        df = pd.DataFrame(self.a, columns=cols)
        # s = perf_counter()
        df_rank = df[cols].rank(pct=True)
        # t = perf_counter()

        # print(f"df time elapsed: {t-s:.2f}")

        # jit_rank = jax.jit(rank, static_argnums=1)

        rank(self.a, axis=0).block_until_ready()
        # s = perf_counter()
        rank_a = rank(self.a, axis=0).block_until_ready()
        # t = perf_counter()
        # print(f"jax rank time elapsed: {t-s:.2f}")
        # print(df_rank)
        # print(rank_a)
        self.assertTrue(np.allclose(df_rank.values, rank_a, rtol=1e-3, atol=1e-3, equal_nan=True))

    def test_zscore(self):
        z = zscore(self.a, axis=-1)

        z_mean = jnp.nanmean(z, axis=-1)
        z_std = jnp.nanstd(z, axis=-1, ddof=1)
        # print(z_mean)
        # print(z_std)
        self.assertTrue(
            jnp.alltrue(jnp.abs(z_mean) < 1e-5) and
            jnp.alltrue(jnp.abs(z_std - 1) < 1e-4)
        )

    # def test_robust_zscore(self):
    #     z = robust_zscore(self.a, axis=-1)
    #
    #     z_mean = jnp.nanmean(z, axis=-1)
    #     # z_std = jnp.nanstd(z, axis=-1, ddof=1)
    #
    #     # print(z_mean)
    #     # print(z_std)
    #     self.assertTrue(
    #         jnp.alltrue(z_mean < 1e-3) # and
    #         #jnp.alltrue(jnp.abs(z_std - 1) < 1e-4)
    #     )

    def test_corr(self):
        """Test pearson and spearman correlations (IC and rank IC)."""
        cols = [f"col{i}" for i in range(self.n_cols)]
        pda = pd.DataFrame(self.a, columns=cols)
        pdb = pd.DataFrame(self.b, columns=cols)

        pd_ic_list = []
        pd_rank_ic_list = []
        for col in cols:
            ic = pda[col].corr(pdb[col], method='pearson')
            rank_ic = pda[col].corr(pdb[col], method='spearman')
            # ic = scipy.stats.pearsonr(pda[col], pdb[col])
            pd_ic_list.append(ic)
            pd_rank_ic_list.append(rank_ic)
        pd_ic = np.array(pd_ic_list)
        pd_rank_ic = np.array(pd_rank_ic_list)

        ic = pearsonr(self.a, self.b, axis=0)
        rank_ic = spearmanr(self.a, self.b, axis=0)
        self.assertTrue(
            np.allclose(pd_ic, ic, rtol=1e-3, atol=1e-5, equal_nan=True) and
            np.allclose(pd_rank_ic, rank_ic, rtol=1e-3, atol=1e-5, equal_nan=True)
        )

    # def test_sma(self):
    #     """Test simple moving average."""
    #     cols = [f"col{i}" for i in range(self.n_cols)]

    #     df = pd.DataFrame(self.a, columns=cols)
    #     window = 7

    #     res1 = df.rolling(window=window, min_periods=1).mean()
    #     res2 = sma(self.a, window=window)

    #     self.assertTrue(
    #         np.allclose(res1.values, res2, rtol=1e-3, atol=1e-5, equal_nan=True))

    # def test_ema(self):
    #     """Test exponentially decayed moving average."""
    #     cols = [f"col{i}" for i in range(self.n_cols)]

    #     df = pd.DataFrame(self.a, columns=cols)
    #     window = 10
    #     alpha = 0.8

    #     res1 = df.ewm(alpha=alpha, min_periods=1).mean()
    #     res2 = ema(self.a, window=window, alpha=alpha)
    #     # print(self.a)
    #     # print(res1.values)
    #     # print(res2)

    #     self.assertTrue(
    #         np.allclose(res1.values, res2, rtol=1e-3, atol=1e-3, equal_nan=True))

    def test_regbeta(self):
        x = np.random.randn(123)
        y = 0.345 * x + 6.789

        import statsmodels.api as sm
        m = sm.OLS(y, sm.add_constant(x))
        res1 = m.fit()

        res2 = regbeta(x, y, axis=0)
        res3 = np.mean(y - res2 * x)

        self.assertTrue(np.allclose([res3, res2], res1.params, rtol=1e-3, atol=1e-5))

    def test_skew(self):
        "note that scipy.stats.skew is not NaN-tolerant"
        a = np.random.normal(0, 1.1, (1000, 10)).astype(np.float32)
        df = pd.DataFrame(a)
        # res1 = skew(a, axis=0, ddof=0)
        # res2 = scipy.stats.skew(a, axis=0, bias=True)
        # res2 = scipy.stats.skew(a, axis=0, bias=False)
        res1 = skew(a, axis=0, bias=False)
        res2 = scipy.stats.skew(a, axis=0, bias=False)
        res3 = df.skew().values # unbiased only
        test1 = np.allclose(res1, res2, rtol=1e-3, atol=1e-5)
        test2 = np.allclose(res1, res3, rtol=1e-3, atol=1e-5)

        res1 = skew(a, axis=0, bias=True)
        res2 = scipy.stats.skew(a, axis=0, bias=True)
        test3 = np.allclose(res1, res2, rtol=1e-3, atol=1e-5)
        self.assertTrue(test1 and test2 and test3)

    def test_kurtosis(self):
        "note that scipy.stats.kurtosis is not NaN-tolerant."
        b = np.random.normal(0, 1.11, (10000, 10))
        res1 = kurtosis(b, axis=0, bias=True)
        res2 = scipy.stats.kurtosis(b, axis=0, bias=True)
        test1 = np.allclose(res1, res2, rtol=1e-3, atol=1e-5)

        res1 = kurtosis(b, axis=0, bias=False)
        res2 = scipy.stats.kurtosis(b, axis=0, bias=False)
        test2 = np.allclose(res1, res2, rtol=1e-3, atol=1e-5)

        # print(res1)
        # print("======")
        # print(res2)
        # res2 = df.skew()
        # print(res1, res2)
        self.assertTrue(test1 and test2)


if __name__ == '__main__':
    unittest.main()