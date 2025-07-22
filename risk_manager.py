import numpy as np

class PositionSizer:
    """
    Compute position sizes based on different sizing rules.
    """

    def __init__(self, fixed_fraction: float = 0.01):
        """
        Args:
            fixed_fraction – fraction of capital to risk per trade (e.g. 1% = 0.01)
        """
        self.fixed_fraction = fixed_fraction

    def kelly_size(self,
                   capital: float,
                   win_rate: float,
                   win_loss_ratio: float) -> float:
        """
        Calculate optimal bet size using the Kelly criterion.
        
        f* = (W – (1 – W) / R)
        where W = win_rate, R = win_loss_ratio
        
        Returns:
            dollars to allocate (floored at 0 if Kelly fraction is negative)
        """
        edge = win_rate - (1 - win_rate) / win_loss_ratio
        fraction = max(edge, 0.0)
        return fraction * capital

    def fixed_fraction_size(self,
                            capital: float) -> float:
        """
        Allocate a fixed fraction of capital.
        
        Returns:
            fixed_fraction * capital
        """
        return self.fixed_fraction * capital


class DrawdownController:
    """
    Monitor and enforce a maximum drawdown limit.
    """

    def __init__(self, max_drawdown: float = 0.10):
        """
        Args:
            max_drawdown – maximum allowable drawdown as a fraction of peak equity
                            (e.g. 0.10 for 10%)
        """
        self.max_drawdown = max_drawdown
        self.peak_equity = None

    def update_equity(self, equity: float) -> float:
        """
        Update peak equity and compute current drawdown.
        
        Args:
            equity – current account equity
        
        Returns:
            current drawdown fraction
        """
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = (self.peak_equity - equity) / self.peak_equity
        return drawdown

    def is_breached(self, equity: float) -> bool:
        """
        Check if drawdown exceeds the maximum allowed.
        
        Args:
            equity – current account equity
        
        Returns:
            True if drawdown > max_drawdown, else False
        """
        return self.update_equity(equity) > self.max_drawdown


if __name__ == "__main__":
    # Example usage
    capital = 100_000
    ps = PositionSizer(fixed_fraction=0.02)
    size_fixed = ps.fixed_fraction_size(capital)
    size_kelly = ps.kelly_size(capital, win_rate=0.55, win_loss_ratio=1.5)
    print(f"Fixed-fraction size: ${size_fixed:.2f}")
    print(f"Kelly size:          ${size_kelly:.2f}")

    dc = DrawdownController(max_drawdown=0.10)
    for equity in [100_000, 102_000, 98_000, 90_000, 88_000]:
        dd = dc.update_equity(equity)
        breach = dc.is_breached(equity)
        print(f"Equity: ${equity:.0f}, Drawdown: {dd:.2%}, Breached? {breach}")


ChatGPT can make mistakes
