import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def engle_granger_test(x: pd.Series,
                       y: pd.Series,
                       trend: str = 'c',
                       maxlag: int = None):
    """
    Perform the Engle–Granger two‐step cointegration test:
      1) Regress x on y (plus intercept/trend)
      2) Test ADF on residuals.

    Args:
        x       – first price series
        y       – second price series
        trend   – 'c' for constant, 'ct' for constant+trend, or 'nc' for none
        maxlag  – maximum lag to use in the ADF test (None uses default)

    Returns:
        coint_t     – test statistic
        pvalue      – p‑value of the test
        crit_values – dict of critical values at 1%, 5%, 10%
    """
    coint_t, pvalue, crit_values = coint(x, y, trend=trend, maxlag=maxlag)
    return coint_t, pvalue, crit_values


def johansen_test(df: pd.DataFrame,
                  det_order: int = 0,
                  k_ar_diff: int = 1):
    """
    Perform Johansen’s cointegration test on a DataFrame of multiple series.

    Args:
        df         – DataFrame where each column is a time series
        det_order  – deterministic trend order:
                     -1: no constant or trend
                      0: constant term only
                      1: constant + linear trend
        k_ar_diff  – number of lagged differences to include

    Returns:
        result     – statsmodels VectorErrorCorrectionResult:
                     .eig      eigenvalues
                     .lr1      trace statistics
                     .cvm      critical values for trace
                     .lr2      max‑eigen statistics
                     .cvm2     critical values for max‑eigen
    """
    result = coint_johansen(df.values, det_order, k_ar_diff)
    return result


def find_cointegrated_pairs(price_df: pd.DataFrame,
                            window: int = None,
                            pvalue_threshold: float = 0.05,
                            use_rolling: bool = False):
    """
    Scan all pairs (i, j) in price_df and return those cointegrated.

    If use_rolling=True and window is specified, each pair must be
    cointegrated (p < pvalue_threshold) in every rolling window slice.

    Args:
        price_df         – DataFrame of price series (columns = symbols)
        window           – window size for rolling test (integer)
        pvalue_threshold – significance threshold for p‑value
        use_rolling      – whether to require cointegration in all windows

    Returns:
        List of tuples: [(symbol_i, symbol_j), ...]
    """
    symbols = price_df.columns
    n = len(symbols)
    coint_pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            xi = price_df[symbols[i]]
            xj = price_df[symbols[j]]

            if use_rolling and window is not None:
                # test on non-overlapping windows
                n_windows = (len(price_df) - window) // window + 1
                pvals = []
                for k in range(n_windows):
                    start = k * window
                    end = start + window
                    xi_win = xi.iloc[start:end]
                    xj_win = xj.iloc[start:end]
                    _, pval, _ = engle_granger_test(xi_win, xj_win)
                    pvals.append(pval)
                if all(p < pvalue_threshold for p in pvals):
                    coint_pairs.append((symbols[i], symbols[j]))
            else:
                _, pval, _ = engle_granger_test(xi, xj)
                if pval < pvalue_threshold:
                    coint_pairs.append((symbols[i], symbols[j]))

    return coint_pairs





