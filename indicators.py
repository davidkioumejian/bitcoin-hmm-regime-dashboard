"""
indicators.py
Self-contained technical indicator implementations (no extra TA dependency,
avoids version/compatibility issues with third-party TA libraries).

All functions take/return pandas Series (or a DataFrame for adx) aligned to
the input index.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))

    # Edge cases where avg_loss == 0 (rs -> inf/NaN):
    #   avg_gain > 0  -> RSI = 100 (all gains, no losses)
    #   avg_gain == 0 -> RSI = 50  (perfectly flat, no movement at all)
    no_loss = avg_loss == 0
    rsi_val = rsi_val.where(~no_loss, np.where(avg_gain > 0, 100.0, 50.0))
    return rsi_val


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average Directional Index. df must have High, Low, Close columns."""
    high, low, close = df["High"], df["Low"], df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx_val


def roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of change (%) over `period` bars."""
    return (series / series.shift(period) - 1) * 100.0


def rolling_volatility_pct(returns: pd.Series, window: int) -> pd.Series:
    """Rolling std dev of returns, expressed in percent."""
    return returns.rolling(window=window, min_periods=window).std() * 100.0
