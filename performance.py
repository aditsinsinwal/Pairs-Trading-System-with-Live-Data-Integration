# performance.py

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Union


class PerformanceAnalyzer:
    """
    Comprehensive performance analytics for a trading strategy.
    Accepts either returns or equity series and computes risk & return metrics.
    """

    def __init__(self,
                 returns: Optional[pd.Series] = None,
                 equity: Optional[pd.Series] = None,
                 positions: Optional[pd.Series] = None):
        """
        Args:
            returns   : pd.Series of periodic returns (e.g. daily). Index must be datetime-like.
            equity    : pd.Series of portfolio equity values. Index must be datetime-like.
                        Used to derive returns if returns not provided.
            positions : pd.Series of normalized positions (+1, 0, -1) aligned with returns.
        """
        if returns is None and equity is None:
            raise ValueError("Either `returns` or `equity` must be provided.")
        if returns is None:
            returns = equity.pct_change().dropna()

        self.returns = returns.dropna()
        self.positions = positions.reindex(self.returns.index) if positions is not None else None
        self.periods_per_year = self._infer_periods_per_year(self.returns.index)

    @staticmethod
    def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
        freq = pd.infer_freq(index)
        if freq is None:
            warnings.warn("Could not infer frequency; defaulting to 252 periods/year.")
            return 252.0
        # Map pandas frequency codes to approximate periods per year
        mapping = {
            'B': 252, 'D': 365, 'W': 52, 'M': 12,
            'Q': 4, 'A': 1, 'H': 24 * 365,
            'T': 252 * 6.5 * 60,  # minutes in trading days
            'S': 252 * 6.5 * 60 * 60,
        }
        for code, periods in mapping.items():
            if freq.startswith(code):
                return float(periods)
        return 252.0

    def annualized_return(self) -> float:
        cumulative = (1 + self.returns).prod()
        return cumulative ** (self.periods_per_year / len(self.returns)) - 1

    def annualized_volatility(self) -> float:
        return self.returns.std(ddof=1) * np.sqrt(self.periods_per_year)

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        rf_per_period = risk_free_rate / self.periods_per_year
        excess = self.returns - rf_per_period
        return excess.mean() / excess.std(ddof=1) * np.sqrt(self.periods_per_year)

    def sortino_ratio(self,
                      risk_free_rate: float = 0.0,
                      required_return: float = 0.0) -> float:
        rf_per_period = risk_free_rate / self.periods_per_year
        excess = self.returns - rf_per_period - required_return / self.periods_per_year
        negative = excess[excess < 0]
        downside_std = negative.std(ddof=1)
        if downside_std == 0:
            return float('nan')
        return excess.mean() / downside_std * np.sqrt(self.periods_per_year)

    def max_drawdown(self) -> float:
        equity_curve = (1 + self.returns).cumprod()
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()

    def calmar_ratio(self) -> float:
        mdd = -self.max_drawdown()
        if mdd == 0:
            return float('nan')
        return self.annualized_return() / mdd

    def value_at_risk(self, level: float = 0.05) -> float:
        """
        Historical VaR at the given level (e.g., 0.05 for 5% VaR).
        """
        return -np.percentile(self.returns, level * 100)

    def conditional_value_at_risk(self, level: float = 0.05) -> float:
        """
        CVaR (Expected Shortfall) at the given level.
        """
        var_thresh = np.percentile(self.returns, level * 100)
        tail = self.returns[self.returns <= var_thresh]
        if tail.empty:
            return -var_thresh
        return -tail.mean()

    def hit_rate(self) -> float:
        return (self.returns > 0).mean()

    def turnover(self) -> float:
        if self.positions is None:
            raise ValueError("`positions` series required for turnover.")
        return self.positions.diff().abs().mean()

    def stats(self, risk_free_rate: float = 0.0) -> pd.Series:
        """
        Summarize all key metrics in a pandas Series.
        """
        data = {
            "Annualized Return":     self.annualized_return(),
            "Annualized Volatility": self.annualized_volatility(),
            "Sharpe Ratio":          self.sharpe_ratio(risk_free_rate),
            "Sortino Ratio":         self.sortino_ratio(risk_free_rate),
            "Calmar Ratio":          self.calmar_ratio(),
            "Max Drawdown":          self.max_drawdown(),
            "Hit Rate":              self.hit_rate(),
            "VaR (5%)":              self.value_at_risk(0.05),
            "CVaR (5%)":             self.conditional_value_at_risk(0.05),
        }
        if self.positions is not None:
            data["Turnover"] = self.turnover()
        return pd.Series(data)

    def monte_carlo(self,
                    n_sims: int = 1_000,
                    seed: Optional[int] = None) -> np.ndarray:
        """
        Bootstrap simulation of total return over the full horizon.
        """
        rng = np.random.default_rng(seed)
        rets = self.returns.values
        sims = rng.choice(rets, size=(n_sims, len(rets)), replace=True)
        return sims.prod(axis=1) - 1


if __name__ == "__main__":
    # Demonstration
    dates = pd.date_range(start="2025-01-01", periods=252, freq="B")
    sample_rets = pd.Series(np.random.normal(0.0005, 0.01, size=252), index=dates)
    analyzer = PerformanceAnalyzer(returns=sample_rets)
    print(analyzer.stats())
    sims = analyzer.monte_carlo(n_sims=10_000, seed=42)
    print(f"5th percentile MC return: {np.percentile(sims, 5):.2%}")
