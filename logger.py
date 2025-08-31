import logging

class TradeLogger:
    def __init__(self, log_file="system.log"):
        self.logger = logging.getLogger("TradingSystem")
        self.logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(ch)
            self.logger.addHandler(fh)

    def log_trade(self, order):
        """Log executed trade"""
        msg = (f"TRADE | {order['side'].upper()} {order['qty']} {order['symbol']} "
               f"at {order.get('price', 'MKT')} | ID: {order['id']}")
        self.logger.info(msg)

    def log_event(self, message):
        """Log general system event"""
        self.logger.info(f"EVENT | {message}")

    def log_error(self, message):
        """Log errors"""
        self.logger.error(f"ERROR | {message}")
