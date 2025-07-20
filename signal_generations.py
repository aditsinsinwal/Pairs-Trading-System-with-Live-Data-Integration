import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(price_df: pd.DataFrame,
                            window: int = 252,
                            pvalue_threshold: float = 0.05):
    """
    Scan all pairs in price_df over rolling windows of length `window`
    and return those whose ADF p-value on the residual is < pvalue_threshold.
    """
    n = price_df.shape[1]
    pairs = []
    cols = price_df.columns
    for i in range(n):
        for j in range(i+1, n):
            pvalues = []
            for start in range(0, len(price_df) - window + 1, window):
                end = start + window
                series_i = price_df.iloc[start:end, i]
                series_j = price_df.iloc[start:end, j]
                _, pvalue, _ = coint(series_i, series_j)
                pvalues.append(pvalue)
            # if *all* windows are cointegrated at our threshold, keep the pair
            if all(p < pvalue_threshold for p in pvalues):
                pairs.append((cols[i], cols[j]))
    return pairs


class KalmanFilterEstimator:
    """
    Online Kalman filter to estimate the hedge ratio βₜ and intercept in:
        p1_t = α_t + β_t * p2_t + ε_t
    """

    def __init__(self, delta: float = 1e-5, obs_var: float = 1e-3):
        # state: [alpha, beta]^T
        self.delta = delta
        self.obs_var = obs_var
        self.P = np.eye(2) * 1.0          # state covariance
        self.R = np.eye(2) * delta / (1-delta)  # process covariance
        self.state = np.zeros((2,))       # initial [α₀, β₀]

    def update(self, p1: float, p2: float):
        # 1) Prediction step (state and P already account for process noise via R)
        self.P += self.R

        # 2) Observation
        H = np.array([1.0, p2])           # design row
        y = p1

        # 3) Kalman gain
        S = H @ self.P @ H.T + self.obs_var
        K = (self.P @ H.T) / S

        # 4) State update
        residual = y - (H @ self.state)
        self.state = self.state + K * residual

        # 5) Covariance update
        self.P = self.P - np.outer(K, H @ self.P)

        α, β = self.state
        spread = p1 - (β * p2 + α)
        return α, β, spread


class ZScoreSignal:
    """
    Turn a spread series into z-scores and generate entry/exit signals.
    """

    def __init__(self, lookback: int = 60,
                 entry_z: float = 2.0,
                 exit_z: float = 0.5):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def compute(self, spread_series: pd.Series):
        """
        Returns:
          z_scores: pd.Series of z_t
          signals:  pd.Series of positions (+1 long, -1 short, 0 flat)
        """
        # rolling mean and std
        mu = spread_series.rolling(self.lookback).mean()
        sigma = spread_series.rolling(self.lookback).std()

        z = (spread_series - mu) / sigma

        # generate signals
        signal = pd.Series(index=z.index, dtype='float64').fillna(method='ffill').fillna(0)
        # entry
        signal[z >  self.entry_z] = -1   # short the spread
        signal[z < -self.entry_z] = +1   # long the spread
        # exit
        exiting = z.abs() < self.exit_z
        signal[exiting] = 0

        return z, signal
