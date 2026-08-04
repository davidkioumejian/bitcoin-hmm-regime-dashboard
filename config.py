"""
Shared configuration and defaults for the Regime-Based Trading App.
Centralizing these values keeps data_loader / hmm_engine / strategy / backtester / app.py in sync.
"""

# --- Data ---
TICKER = "BTC-USD"
LOOKBACK_DAYS = 730          # yfinance hard cap for 1h interval
INTERVAL = "1h"

# --- HMM Regime Engine ---
N_COMPONENTS = 7
COVARIANCE_TYPE = "full"
N_ITER = 1000
RANDOM_STATE = 42
HMM_FEATURES = ["Returns", "Range", "VolumeVolatility"]
VOLUME_VOL_WINDOW = 24       # hours, rolling coefficient of variation (std/mean) of volume
REGIME_SMOOTHING_HOURS = 12  # causal rolling-mode filter on the decoded state path (0/1 = off)

# Regime label -> display color. Values are the validated status-scale steps
# (good / critical) from the design system's fixed, reserved-meaning palette —
# not hand-picked — plus the documented muted-ink gray for the Neutral bucket.
REGIME_COLORS = {
    "Bull Run": "#0ca30c",     # status: good
    "Neutral": "#898781",      # muted ink (no reserved status meaning)
    "Bear/Crash": "#d03b3b",   # status: critical
}
# Text-safe variants (higher contrast, for labels/numbers rather than fills/marks)
REGIME_TEXT_COLORS = {
    "Bull Run": "#006300",     # success text (light-surface safe)
    "Neutral": "#52514e",      # secondary ink
    "Bear/Crash": "#d03b3b",   # critical clears 4.68:1 on light surface already
}
REGIME_SHADE_OPACITY = {
    "Bull Run": 0.14,
    "Neutral": 0.0,    # no shading for neutral, keeps the chart clean
    "Bear/Crash": 0.14,
}

# --- Strategy: 8-confirmation voting system ---
MIN_VOTES_REQUIRED = 7
RSI_PERIOD = 14
RSI_MAX = 90
MOMENTUM_PERIOD = 10          # bars, rate-of-change lookback
MOMENTUM_MIN_PCT = 1.0
VOLATILITY_WINDOW = 24        # bars, rolling std of returns
VOLATILITY_MAX_PCT = 6.0
VOLUME_SMA_PERIOD = 20
ADX_PERIOD = 14
ADX_MIN = 25
EMA_FAST = 50
EMA_SLOW = 200
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# --- Risk management ---
COOLDOWN_HOURS = 48
LEVERAGE = 2.5
INITIAL_CAPITAL = 10_000.0
FEE_PCT_PER_SIDE = 0.05       # % per side (entry/exit), realistic taker-fee assumption

BULL_STATE_LABEL = "Bull Run"
BEAR_STATE_LABEL = "Bear/Crash"
