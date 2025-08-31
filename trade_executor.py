import uuid
import time
from logger import TradeLogger

class TradeExecutor:
    def __init__(self, mode="paper"):
        """
        mode: 'paper' (default) or 'live'
        """
        self.mode = mode
        self.positions = {}
        self.logger = TradeLogger(log_file="trades.log")

    def send_order(self, symbol, qty, side, order_type="market", price=None):
        """
        Send order to broker (simulated if paper).
        Returns order_id.
        """
        order_id = str(uuid.uuid4())[:8]  # short unique ID

        order = {
            "id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,  # "buy" or "sell"
            "type": order_type,  # "market" or "limit"
            "price": price,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        if self.mode == "paper":
            self._execute_paper_order(order)
        else:
            self._execute_live_order(order)

        self.logger.log_trade(order)
        return order_id

    def _execute_paper_order(self, order):
        """Simulated fills for paper trading"""
        symbol = order["symbol"]
        qty = order["qty"] if order["side"] == "buy" else -order["qty"]
        self.positions[symbol] = self.positions.get(symbol, 0) + qty

    def _execute_live_order(self, order):
        """TODO: Implement broker API call (Alpaca, IB, etc.)"""
        raise NotImplementedError("Live trading not implemented yet.")

    def get_positions(self):
        """Return current open positions"""
        return self.positions
