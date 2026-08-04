"""
strategy.py
The Strategy Logic: computes the 8 confirmation indicators, tallies votes,
and derives the entry gate (Regime == Bull Run AND >= MIN_VOTES_REQUIRED/8)
and the hard regime-flip exit trigger (Regime == Bear/Crash).
"""

import pandas as pd

import config
import indicators as ind

# Ordered list of the 8 confirmation condition names, used everywhere
# (voting math, UI breakdown table) so the order stays consistent.
VOTE_CONDITIONS = [
    "RSI < 90",
    "Momentum > 1%",
    "Volatility < 6%",
    "Volume > 20-SMA",
    "ADX > 25",
    "Price > EMA50",
    "Price > EMA200",
    "MACD > Signal",
]


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds all indicator columns needed for the voting system."""
    out = df.copy()

    out["RSI"] = ind.rsi(out["Close"], config.RSI_PERIOD)
    out["Momentum"] = ind.roc(out["Close"], config.MOMENTUM_PERIOD)
    out["VolatilityPct"] = ind.rolling_volatility_pct(out["Returns"], config.VOLATILITY_WINDOW)
    out["VolumeSMA"] = ind.sma(out["Volume"], config.VOLUME_SMA_PERIOD)
    out["ADX"] = ind.adx(out, config.ADX_PERIOD)
    out["EMA50"] = ind.ema(out["Close"], config.EMA_FAST)
    out["EMA200"] = ind.ema(out["Close"], config.EMA_SLOW)
    macd_line, signal_line, hist = ind.macd(
        out["Close"], config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
    )
    out["MACD"] = macd_line
    out["MACD_Signal"] = signal_line
    out["MACD_Hist"] = hist

    return out


def compute_votes(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean DataFrame, one column per confirmation condition, aligned to df.index."""
    votes = pd.DataFrame(index=df.index)
    votes["RSI < 90"] = df["RSI"] < config.RSI_MAX
    votes["Momentum > 1%"] = df["Momentum"] > config.MOMENTUM_MIN_PCT
    votes["Volatility < 6%"] = df["VolatilityPct"] < config.VOLATILITY_MAX_PCT
    votes["Volume > 20-SMA"] = df["Volume"] > df["VolumeSMA"]
    votes["ADX > 25"] = df["ADX"] > config.ADX_MIN
    votes["Price > EMA50"] = df["Close"] > df["EMA50"]
    votes["Price > EMA200"] = df["Close"] > df["EMA200"]
    votes["MACD > Signal"] = df["MACD"] > df["MACD_Signal"]

    # NaN comparisons already evaluate to False (indicator warm-up period),
    # which is the desired behavior: no confirmations during warm-up.
    votes = votes.fillna(False)
    return votes[VOTE_CONDITIONS]


def generate_signals(df: pd.DataFrame, votes: pd.DataFrame,
                      min_votes: int = config.MIN_VOTES_REQUIRED) -> pd.DataFrame:
    """
    Adds VoteCount, EntryGate (Bull Run regime + enough confirmations) and
    RegimeExit (Bear/Crash regime -> hard exit trigger) columns.
    """
    out = df.copy()
    out["VoteCount"] = votes.sum(axis=1)
    out["EntryGate"] = (out["Regime"] == config.BULL_STATE_LABEL) & (out["VoteCount"] >= min_votes)
    out["RegimeExit"] = out["Regime"] == config.BEAR_STATE_LABEL
    return out


def run_strategy_pipeline(df: pd.DataFrame, min_votes: int = config.MIN_VOTES_REQUIRED):
    """Convenience wrapper: indicators -> votes -> signals. Returns (df, votes)."""
    df = compute_indicators(df)
    votes = compute_votes(df)
    df = generate_signals(df, votes, min_votes)
    return df, votes
