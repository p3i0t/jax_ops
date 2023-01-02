import unittest
from wave import Wave_write

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.stats
# from time import perf_counter

# import jax
from ops import (rolling_mean, sma, rolling_min, rolling_max,
                rolling_kurtosis, rolling_skew, rolling_corr,
                rolling_rank_corr, rolling_regbeta,
                rolling_robust_zscore, rolling_zscore,
                rolling_std, rolling_var, ema)


class TestRollingOps(unittest.TestCase):
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


    def test_sma(self):
        """Test simple moving average."""
        cols = [f"col{i}" for i in range(self.n_cols)]

        df = pd.DataFrame(self.a, columns=cols)
        window = 7

        res1 = df.rolling(window=window, min_periods=1).mean()
        res2 = sma(self.a, axis=0, window=window)

        self.assertTrue(
            np.allclose(res1.values, res2, rtol=1e-3, atol=1e-5, equal_nan=True))


    def test_ema(self):
        """Test exponentially decayed moving average. In our implementation, `window` and `alpha`
        are independent. When window gets bigger (pereferably > 5) and alpha approaches to 1,
        ema appoaches to df.ewm(alpha=alpha, min_periods=1).mean(). """
        cols = [f"col{i}" for i in range(self.n_cols)]

        df = pd.DataFrame(self.a, columns=cols)
        window = 10
        alpha = 0.7

        res1 = df.ewm(alpha=alpha, min_periods=1).mean()
        res2 = ema(self.a, axis=0, window=window, alpha=alpha)
        # print(self.a)
        # print(res1.values)
        # print(res2)

        self.assertTrue(
            np.allclose(res1.values, res2, rtol=1e-3, atol=1e-3, equal_nan=True))

    def test_rolling_std(self):
        cols = [f"col{i}" for i in range(self.n_cols)]

        dfa = pd.DataFrame(self.a, columns=cols)
        # dfb = pd.DataFrame(self.b, columns=cols)
        window = 10

        res1 = dfa.rolling(window=window, min_periods=1).std(ddof=0)
        res2 = rolling_std(self.a, axis=0, window=window)


        # print(res1.values)
        # print(res2)
        self.assertTrue(
           np.allclose(res1.values, res2, rtol=1e-3, atol=1e-5, equal_nan=True)
        )


    def test_rolling_var(self):
        cols = [f"col{i}" for i in range(self.n_cols)]

        dfa = pd.DataFrame(self.a, columns=cols)
        # dfb = pd.DataFrame(self.b, columns=cols)
        window = 10

        res1 = dfa.rolling(window=window, min_periods=1).var(ddof=0)
        res2 = rolling_var(self.a, axis=0, window=window)


        # print(res1.values)
        # print(res2)
        self.assertTrue(
           np.allclose(res1.values, res2, rtol=1e-3, atol=1e-5, equal_nan=True)
        )

    def test_rolling_skew(self):
        cols = [f"col{i}" for i in range(self.n_cols)]

        dfa = pd.DataFrame(self.a, columns=cols)
        # dfb = pd.DataFrame(self.b, columns=cols)
        window = 10


        # unbiased for dataframe.skew
        res1 = dfa.rolling(window=window, min_periods=1).skew().values
        res2 = rolling_skew(self.a, axis=0, window=window, bias=False)

        self.assertTrue(
           np.allclose(res1, res2, rtol=1e-3, atol=1e-5, equal_nan=True)
        )

    def test_rolling_kurtosis(self):
        cols = [f"col{i}" for i in range(self.n_cols)]

        dfa = pd.DataFrame(self.a, columns=cols)
        # dfb = pd.DataFrame(self.b, columns=cols)
        window = 10

        # unbiased for dataframe.kurt
        res1 = dfa.rolling(window=window, min_periods=1).kurt().values
        res2 = rolling_kurtosis(self.a, axis=0, window=window, bias=False)

        self.assertTrue(
           np.allclose(res1, res2, rtol=1e-3, atol=1e-5, equal_nan=True)
        )


    def test_rolling_min(self):
        cols = [f"col{i}" for i in range(self.n_cols)]

        dfa = pd.DataFrame(self.a, columns=cols)
        # dfb = pd.DataFrame(self.b, columns=cols)
        window = 10

        res1 = dfa.rolling(window=window, min_periods=1).min()
        res2 = rolling_min(self.a, axis=0, window=window)

        self.assertTrue(
           np.allclose(res1.values, res2, rtol=1e-3, atol=1e-5, equal_nan=True)
        )

    def test_rolling_max(self):
        cols = [f"col{i}" for i in range(self.n_cols)]

        dfa = pd.DataFrame(self.a, columns=cols)
        # dfb = pd.DataFrame(self.b, columns=cols)
        window = 10

        res1 = dfa.rolling(window=window, min_periods=1).max()
        res2 = rolling_max(self.a, axis=0, window=window)

        self.assertTrue(
           np.allclose(res1.values, res2, rtol=1e-3, atol=1e-5, equal_nan=True)
        )

    def test_rolling_corr(self):
        cols = [f"col{i}" for i in range(self.n_cols)]

        dfa = pd.DataFrame(self.a, columns=cols)
        dfb = pd.DataFrame(self.b, columns=cols)
        window = 10

        res1_list = []
        for col in cols:
            res = dfa[col].rolling(window=window).corr(dfb[col], min_periods=1)
            res1_list.append(res)

        res1 = np.stack(res1_list, axis=1)
        res2 = rolling_corr(self.a, self.b, axis=0, window=window)

        print(res1.shape, res2.shape)
        # Be careful to the fact that dfa.rolling().corr(dfb) produces NaN as long as the
        # inputs contain NaN, while our implementation rolling_corr allows NaN in the inputs.
        # To pass the blackbox test, the following line is critical to align their behaviours.
        res2 = jnp.where(jnp.isnan(res1), float('nan'), res2)
        self.assertTrue(
            np.allclose(res1, res2, rtol=1e-3, atol=1e-5, equal_nan=True))

    # def test_rolling_rank_corr(self):
    #     cols = [f"col{i}" for i in range(self.n_cols)]

    #     dfa = pd.DataFrame(self.a, columns=cols)
    #     dfb = pd.DataFrame(self.b, columns=cols)
    #     a = np.random.randn(30)
    #     b = np.random.randn(30)
    #     dfa = pd.DataFrame({'c': a})
    #     dfb = pd.DataFrame({'c': b})
    #     window = 20

    #     # res1_list = []
    #     # for col in cols:
    #     #     res = dfa[col].rolling(window=window).corr(dfb[col], method='spearman', min_periods=1)
    #     #     res1_list.append(res)

    #     # res1 = np.stack(res1_list, axis=1)
    #     # res2 = rolling_rank_corr(self.a[:, 0], self.b[:, 0], axis=0, window=window)

    #     res1 = dfa['c'].rolling(window=window).corr(dfb['c'], method='spearman')
    #     res3 = dfa['c'].rolling(window=window).corr(dfb['c'])
    #     res2 = rolling_rank_corr(a, b, axis=0, window=window)
    #     print(res1)
    #     print(res2)
    #     print(res3)
    #     # print(res1_list[0].values[-20:])
    #     # print('=========')
    #     # print(res2[-20:])

    #     # Be careful to the fact that dfa.rolling().corr(dfb) produces NaN as long as the
    #     # inputs contain NaN, while our implementation rolling_corr allows NaN in the inputs.
    #     # To pass the blackbox test, the following line is critical to align their behaviours.
    #     # res2 = jnp.where(jnp.isnan(res1), float('nan'), res2)
    #     self.assertTrue(
    #         np.allclose(res1, res2, rtol=1e-3, atol=1e-5, equal_nan=True))


if __name__ == '__main__':
    unittest.main()