"""
data_loader.py
Fetches hourly OHLCV data from yfinance and engineers the raw features
consumed by the HMM regime engine (Returns, Range, VolumeVolatility).
"""

import numpy as np
import pandas as pd
import yfinance as yf

import config


class DataLoadError(RuntimeError):
    """Raised when market data cannot be fetched or is unusable."""


def fetch_ohlcv(ticker: str = config.TICKER, period_days: int = config.LOOKBACK_DAYS,
                 interval: str = config.INTERVAL) -> pd.DataFrame:
    """
    Download hourly OHLCV data from yfinance.

    IMPORTANT (yfinance quirks handled defensively):
    - Uses period="{days}d" rather than start/end (required for intraday intervals).
    - yfinance can return MultiIndex columns (ticker, field) even for a single
      symbol depending on version/settings; flatten with get_level_values(0)
      so downstream code can rely on exactly Open/High/Low/Close/Volume.
    """
    period_days = min(period_days, 730)  # Yahoo hard limit for 1h interval
    raw = yf.download(
        tickers=ticker,
        period=f"{period_days}d",
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if raw is None or raw.empty:
        raise DataLoadError(
            f"No data returned for '{ticker}'. Check the ticker symbol or your network connection."
        )

    df = raw.copy()

    # Defensively flatten MultiIndex columns (ticker, field) -> field
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataLoadError(f"Downloaded data is missing expected columns: {missing}")

    df = df[required].copy()
    df.index.name = "Datetime"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(how="all")

    if len(df) < 300:
        raise DataLoadError(
            f"Only {len(df)} bars returned for '{ticker}' — not enough history to train the model."
        )

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer the 3 HMM input features on top of raw OHLCV:
      - Returns:            Close.pct_change()
      - Range:               (High - Low) / Close
      - VolumeVolatility:    rolling coefficient of variation of Volume
                              (rolling std / rolling mean over VOLUME_VOL_WINDOW bars)

    Yahoo's hourly BTC-USD volume frequently includes zero-volume bars, which makes
    Volume.pct_change() divide by zero and poisons every rolling window it touches.
    The coefficient-of-variation form measures dispersion within a trailing window of
    raw volume instead of bar-to-bar ratios, so it stays finite even when individual
    bars report zero volume.
    """
    out = df.copy()

    out["Returns"] = out["Close"].pct_change()
    out["Range"] = (out["High"] - out["Low"]) / out["Close"]

    vol_roll_mean = out["Volume"].rolling(config.VOLUME_VOL_WINDOW, min_periods=config.VOLUME_VOL_WINDOW).mean()
    vol_roll_std = out["Volume"].rolling(config.VOLUME_VOL_WINDOW, min_periods=config.VOLUME_VOL_WINDOW).std()
    out["VolumeVolatility"] = vol_roll_std / vol_roll_mean.replace(0, np.nan)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["Returns", "Range", "VolumeVolatility"])

    return out


def load_market_data(ticker: str = config.TICKER, period_days: int = config.LOOKBACK_DAYS,
                      interval: str = config.INTERVAL) -> pd.DataFrame:
    """Convenience wrapper: fetch + engineer features in one call."""
    df = fetch_ohlcv(ticker, period_days, interval)
    df = add_features(df)
    return df
