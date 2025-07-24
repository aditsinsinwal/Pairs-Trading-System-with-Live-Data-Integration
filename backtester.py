
import pandas as pd
import numpy as np

class BacktestEngine:
    """
    Simple backtester for a pairs‐trading strategy.
    Simulates mark‑to‑market PnL, slippage and fixed commissions.
    """

    def __init__(self,
                 initial_capital: float = 100_000.0,
                 slippage: float = 0.0001,
                 commission_per_trade: float = 1.0,
                 unit_size: float = 1.0):
        """
        Args:
            initial_capital       – starting cash balance
            slippage              – per‑dollar slippage pct (e.g. 0.0001 = 1bp)
            commission_per_trade  – flat commission charged on each entry or exit
            unit_size             – number of “spread units” per trade
        """
        self.initial_capital = initial_capital
        self.slippage = slippage
        self.commission_per_trade = commission_per_trade
        self.unit_size = unit_size

    def run(self,
            price1: pd.Series,
            price2: pd.Series,
            signals: pd.Series,
            hedge_ratios: pd.Series) -> pd.DataFrame:
        """
        Backtest the strategy.

        Args:
            price1       – price series of asset 1
            price2       – price series of asset 2
            signals      – position series: +1 long spread, –1 short, 0 flat
            hedge_ratios – dynamic β series (same index)

        Returns:
            DataFrame with columns [p1, p2, signal, beta, equity, returns]
        """
        # align inputs
        df = pd.concat([
            price1.rename("p1"),
            price2.rename("p2"),
            signals.rename("signal"),
            hedge_ratios.rename("beta")
        ], axis=1).dropna()

        cash = self.initial_capital
        equity_curve = []
        prev_signal = 0
        prev_p1 = prev_p2 = None

        for idx, row in df.iterrows():
            p1, p2, sig, beta = row.p1, row.p2, row.signal, row.beta

            # 1) PnL from holding previous position
            if prev_p1 is not None:
                pos1 = prev_signal * self.unit_size
                pos2 = -prev_signal * beta * self.unit_size
                pnl = pos1 * (p1 - prev_p1) + pos2 * (p2 - prev_p2)
                cash += pnl

            # 2) Transaction costs when changing position
            if sig != prev_signal:
                # shares traded for each leg
                traded_shares1 = abs(sig - prev_signal) * self.unit_size
                traded_shares2 = abs(sig - prev_signal) * abs(beta) * self.unit_size

                # slippage cost ≈ slippage_pct × notional traded
                slip_cost = self.slippage * (traded_shares1 * p1 + traded_shares2 * p2)
                cash -= (slip_cost + self.commission_per_trade)

            equity_curve.append(cash)
            prev_signal, prev_p1, prev_p2 = sig, p1, p2

        df["equity"] = equity_curve
        df["returns"] = df["equity"].pct_change().fillna(0.0)
        return df


if __name__ == "__main__":
    # --- Example usage ---
    import numpy as np

    dates = pd.date_range("2025-01-01", periods=100, freq="T")
    p1 = pd.Series(100 + np.cumsum(np.random.randn(100)*0.1), index=dates)
    p2 = pd.Series(200 + np.cumsum(np.random.randn(100)*0.2), index=dates)
    beta = pd.Series(0.5, index=dates)
    # simple signal: alternate long/flat every 10 ticks
    sig = pd.Series((np.arange(100)//10 % 2)*2 - 1, index=dates)
    sig[sig == -1] = 0  # 0 or 1

    engine = BacktestEngine(initial_capital=1e5,
                            slippage=1e-4,
                            commission_per_trade=0.5,
                            unit_size=10)
    results = engine.run(p1, p2, sig, beta)
    print(results[["signal", "equity"]].head())
