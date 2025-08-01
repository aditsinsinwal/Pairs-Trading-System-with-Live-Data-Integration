import pandas as pd
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# Tables
class PriceData(Base):
    __tablename__ = 'price_data'
    id        = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, nullable=False)
    symbol    = Column(String,  index=True, nullable=False)
    price     = Column(Float,   nullable=False)

class SignalData(Base):
    __tablename__ = 'signals'
    id          = Column(Integer, primary_key=True)
    timestamp   = Column(DateTime, index=True, nullable=False)
    symbol      = Column(String,  nullable=False)
    hedge_ratio = Column(Float,   nullable=False)
    spread      = Column(Float,   nullable=False)
    z_score     = Column(Float,   nullable=False)
    signal      = Column(Integer, nullable=False)

class BacktestResult(Base):
    __tablename__ = 'backtest_results'
    id         = Column(Integer, primary_key=True)
    timestamp  = Column(DateTime, index=True, nullable=False)
    equity     = Column(Float,   nullable=False)
    returns    = Column(Float,   nullable=False)

class PerformanceMetric(Base):
    __tablename__ = 'performance_metrics'
    name  = Column(String, primary_key=True)
    value = Column(Float,  nullable=False)

class SQLStorage:
    """
    Simple SQL storage using SQLAlchemy.
    """
    def __init__(self, conn_str: str):
        # set up DB
        self.engine  = create_engine(conn_str, echo=False, future=True)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, future=True)

    def save_price_data(self, df: pd.DataFrame):
        # convert wide to long
        tmp = df.reset_index().melt(
            id_vars='index', var_name='symbol', value_name='price'
        ).rename(columns={'index': 'timestamp'})
        tmp['timestamp'] = pd.to_datetime(tmp['timestamp'])
        tmp.to_sql('price_data', self.engine, if_exists='append', index=False)

    def save_signals(self, df: pd.DataFrame):
        # dump signals table
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.to_sql('signals', self.engine, if_exists='append', index=False)

    def save_backtest_results(self, df: pd.DataFrame):
        # dump equity and returns
        tmp = df[['equity', 'returns']].reset_index().rename(columns={'index':'timestamp'})
        tmp['timestamp'] = pd.to_datetime(tmp['timestamp'])
        tmp.to_sql('backtest_results', self.engine, if_exists='append', index=False)

    def save_performance_metrics(self, metrics: pd.Series):
        # overwrite metrics table
        tmp = metrics.reset_index().rename(columns={'index':'name', 0:'value'})
        tmp.to_sql('performance_metrics', self.engine, if_exists='replace', index=False)

    def load_price_data(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Load prices for one symbol in time range.
        """
        sql = (
            "SELECT timestamp, price"
            " FROM price_data"
            " WHERE symbol = :sym"
            " AND timestamp BETWEEN :start AND :end"
            " ORDER BY timestamp"
        )
        df = pd.read_sql_query(
            sql,
            self.engine,
            params={"sym": symbol, "start": start, "end": end},
            parse_dates=["timestamp"]
        ).set_index("timestamp")
        return df
