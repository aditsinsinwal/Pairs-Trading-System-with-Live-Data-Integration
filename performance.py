import numpy as np
import pandas as pd


class MetricsCalculator:
    """
    Calculate key performance metrics for a trading strategy.
    
    Args:
        returns   – pd.Series of strategy period returns (e.g. daily or minute)
        positions – (optional) pd.Series of position sizes (e.g. +1, 0, –1) aligned with returns
    """

    def __init__(self, returns: pd.Series, positions: pd.Series = None):
        self.returns = returns.dropna()
        self.positions = positions.reindex(self.returns.index) if positions is not None else None

    def annualized_return(self, periods_per_year: int = 252) -> float:
        """
        Compound growth rate annualized.
        """
        total_ret = (1 + self.returns).prod() - 1
        n = self.returns.shape[0]
        return (1 + total_ret) ** (periods_per_year / n) - 1

    def annualized_volatility(self, periods_per_year: int = 252) -> float:
        """
        Standard deviation of returns annualized.
        """
        return self.returns.std(ddof=1) * np.sqrt(periods_per_year)

    def sharpe_ratio(self,
                     risk_free_rate: float = 0.0,
                     periods_per_year: int = 252) -> float:
        """
        (Return – Rf) / vol, annualized.
        
        Args:
            risk_free_rate – annual risk‑free rate (as decimal)
        """
        rf_per_period = risk_free_rate / periods_per_year
        excess = self.returns - rf_per_period
        return excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)

    def max_drawdown(self) -> float:
        """
        Maximum peak-to-trough drawdown as a fraction (negative).
        """
        equity = (1 + self.returns).cumprod()
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        return drawdown.min()

    def hit_rate(self) -> float:
        """
        Proportion of periods with positive returns.
        """
        wins = (self.returns > 0).sum()
        return wins / self.returns.shape[0]

    def turnover(self) -> float:
        """
        Average absolute change in normalized positions per period.
        """
        if self.positions is None:
            raise ValueError("positions series required for turnover calculation")
        # assume positions are normalized (e.g. -1,0,+1)
        return self.positions.diff().abs().mean()

    def monte_carlo_simulation(self,
                               n_sims: int = 1000,
                               random_seed: int = None) -> np.ndarray:
        """
        Bootstrap simulation of compound returns by sampling periods with replacement.
        
        Returns:
            1D array of simulated total returns over the full horizon.
        """
        rng = np.random.default_rng(random_seed)
        rets = self.returns.values
        n = rets.shape[0]
        sims = rng.choice(rets, size=(n_sims, n), replace=True)
        return sims.prod(axis=1) - 1

    def report(self,
               periods_per_year: int = 252,
               risk_free_rate: float = 0.0) -> pd.Series:
        """
        Summarize metrics in a pandas Series.
        """
        data = {
            "Annualized Return":     self.annualized_return(periods_per_year),
            "Annualized Volatility": self.annualized_volatility(periods_per_year),
            "Sharpe Ratio":          self.sharpe_ratio(risk_free_rate, periods_per_year),
            "Max Drawdown":          self.max_drawdown(),
            "Hit Rate":              self.hit_rate()
        }
        if self.positions is not None:
            data["Turnover"] = self.turnover()
        return pd.Series(data)


if __name__ == "__main__":
    # Example usage
    dates = pd.date_range("2025-01-01", periods=252, freq="B")
    # simulate random daily returns ~ N(0.0005, 0.01)
    rets = pd.Series(np.random.normal(0.0005, 0.01, size=252), index=dates)
    # fake positions switching every 20 days
    pos = pd.Series((np.arange(252) // 20) % 3 - 1, index=dates)  # -1,0,+1 cycle
    
    mc = MetricsCalculator(rets, pos)
    summary = mc.report()
    print(summary)
    sims = mc.monte_carlo_simulation(n_sims=5000, random_seed=42)
    print(f"5th percentile simulated return: {np.percentile(sims, 5):.2%}")

