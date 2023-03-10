# jax_snippets

Some commonly used ops implemented in Jax, which can be accelerated by GPUs.
Two goals of this repo: (1) correctness. Correctness is guaranteed by black-box unittest. Ops of this repo are compared to
corresponding implementations of other well-known packages, e.g. Pandas, numpy, statsmodels.
(2) speed. How fast we could compute the ops on inputs when we exhaust
the GPUs available.

Feel free to reuse these snippets. Reach me via wangxin@bopufund.com should you have any questions or suggestions.

*Reminder*:
> Consider use these ops implementations only if your inputs are considerably large. Otherwise, these ops may be slower than
> the corresponding Pandas, numpy implementations.
> That is normal due to the overhead to use GPU (transferring time between CPU and GPU).

## Code Structure

<!-- the following generated by shell command `tree -I __pycache__` -->
```bash
.
├── factor_correlation.py
├── main.py
├── numpy_comparison.py
├── ops
│   ├── __init__.py
│   ├── rolling_ops.py
│   └── vanilla_ops.py
├── pandas_comparison.py
├── README.md
├── requirements.txt
└── test
    ├── __init__.py
    ├── test_rolling_ops.py
    └── test_vanilla_ops.py
```

## Usage

`Ops` implemented in `ops.py` can be divide in two parts: (1) basic vectorized ops (e.g. `zscore`, `rank`) using native Jax APIs. (2) rolling versions
of basic ops (e.g. rolling_zscore, rolling_corr, rolling_rank_corr).

```python
import numpy as np
import jax
import jax.numpy as jnp

from ops import rank, zscore, spearmanr, pearsonr


x = np.random.randn(100, 200, 300)

o = rank(x, axis=2)  # pct rank along axis=2
o = zscore(x, axis=1)  # zscore along axis=1
```

## Unittest

run  ``python -m unittest test_ops.py`` to start `unittest`.

If all testcases are passed, you should see:

```shell

$ python -m unittest test_ops.py

........
----------------------------------------------------------------------
Ran 8 tests in 7.034s

OK

```

The following table list the ops in our unittest and their corresponding counterparts. Note that we pay *special attention* to
the behaviours when the inputs contain NaNs.

The general rule is that *NaNs in inputs is the assumption by default*, unless specified otherwise.

| Ops   | Counterpart | NaN behaviours |
| --- | --- | --- |
| rank      | [pd.DataFrame.rank(pct=True)](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html)| default|
|z_score| - | default|
|pearsonr (normal IC)|[pd.DataFrame.corr(method="pearson")](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html?highlight=corr#pandas.DataFrame.corr)|default|
|spearman (rank IC)|[pd.DataFrame.corr(method="spearman")](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html?highlight=corr#pandas.DataFrame.corr)|default|
|ema|[pd.DataFrame.ewm(adjust=True, alpha=None)](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ewm.html?highlight=ewm#pandas.DataFrame.ewm)|default|
|regbeta|[statsmodels.OLS](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html)| *output NaN if there are NaNs in the inputs* |
|skew|[scipy.stats.skew(bias=True or False)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html)| default (at least 3 elements, otherwise NaN)|
|kurtosis|[scipy.stats.kurtosis(bias=True or False)](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html)| default (at least 4 elements, otherwise NaN)|

We do not include the rolling versions of above ops in the unittest.
The correctness of rolling versions is assumed to be automatic, based on the correctness of *unrolled* ops in the unittest.

## Performance

The speedup in practice varies for different `op`s and sizes of the inputs. So here, we provide some simple examples.

### Comparison with Numpy

We provide testing on two `op`s: cross_sectional rank, and zscore as an example.

The input simulates 1 minute bars of all stocks (~4000) of 500 days:

```python
n_date = 500
n_time = 240
n_symbol = 4000

x = np.random.randn(n_date, n_time, n_symbol)
```

On one single A10 GPU, run:

```python
python numpy_comparison.py
```

You should see:

```shell
$ python numpy_comparison.py
2022-08-23 11:03:18.315 | INFO     | __main__:<module>:53 - np_rank time elapsed: 31.3122s
2022-08-23 11:04:10.729 | INFO     | __main__:<module>:56 - np_zscore time elapsed: 10.4823s
2022-08-23 11:04:12.698 | INFO     | __main__:<module>:61 - transfer x from CPU to GPU: 1.9692s
2022-08-23 11:04:14.704 | INFO     | __main__:<module>:66 - jax rank time elapsed: 0.3471s
2022-08-23 11:04:15.147 | INFO     | __main__:<module>:70 - zscore time elapsed: 0.0555s
```

The results show ~100 times speedup for `rank` and ~200 times speedup for `zscore`. Note that, the overhead - time transferring
the input from CPU to GPU, is considerable compared to the real computation time.
Special attention should be paid in practice to this overhead if a chain of ops to be evaluated.
You should try to minimize the transferrings required.

### Comparison with Pandas

For inputs:

```python
n_rows = 6000
n_cols = 4000
x = np.random.randn(n_rows, n_cols)
y = np.random.randn(n_rows, n_cols)

dfx = pd.DataFrame(x)
dfy = pd.DataFrame(y)

window = 30
```

On a single A10 GPU, run:

```shell
python pandas_comparison.py
```

You should see:

```bash
$ python pandas_comparison.py
2022-08-23 15:20:29.582 | INFO     | __main__:<module>:26 - pd rolling corr time elapsed: 4.4219s
2022-08-23 15:20:35.999 | INFO     | __main__:<module>:29 - pd rolling kurt time elapsed: 1.2833s
2022-08-23 15:20:36.698 | INFO     | __main__:<module>:35 - transfer x, y from CPU to GPU: 0.6991s
2022-08-23 15:20:38.622 | INFO     | __main__:<module>:39 - jax rolling corr time elapsed: 0.3847s
2022-08-23 15:20:39.381 | INFO     | __main__:<module>:42 - jax rolling_kurt time elapsed: 0.1517s
```

The results show ~10 times speedup for `rolling_corr` and `rolling_kurtosis`.

### Full Performance

In all tests below, we assume that all the raw inputs are 3D Tensors (or Nd-array) in shape of *(date, minute, symbol)*, where `date` is for trading dates, `minute` is for intraday
trading minutes (240mins for A-shares, 9:30-11:30, 13:30-15:00), and `symbol` is for the stock universe. Missing values *should be and have to be* filled with NaNs. Remember that all ops implemented are
NaN-friendly.

#### Performance on single GPU

#### Performance on multiple GPUs

to be added ...




