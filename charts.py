"""
charts.py
Plotly figure builders for the dashboard: the regime-shaded candlestick chart
and the equity-curve / drawdown chart. Kept separate from app.py so the
Streamlit page-layout code stays focused on orchestration.

Color usage follows the app's fixed, validated palette (see config.py):
status green/red are reserved for Bull Run / Bear-Crash and reused
consistently for candle up/down; EMA overlays use two categorical hues that
never collide with the status colors.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

BAR_INTERVAL = pd.Timedelta(hours=1)

# Categorical accents for reference lines (never reused for status meaning)
COLOR_EMA_FAST = "#2a78d6"   # categorical slot 1 (blue)
COLOR_EMA_SLOW = "#eb6834"   # categorical slot 2 (orange)
COLOR_VOLUME = "rgba(137, 135, 129, 0.55)"   # muted ink, de-emphasized
COLOR_BUY_HOLD = "#898781"   # muted ink (context / de-emphasis line)
COLOR_STRATEGY = "#2a78d6"   # categorical slot 1 (accent line)


def compute_regime_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the bar-by-bar Regime column into contiguous runs, so the
    chart draws one shaded rectangle per regime episode instead of one per bar."""
    regime = df["Regime"]
    group_id = regime.ne(regime.shift()).cumsum()
    idx_series = df.index.to_series()

    segments = pd.DataFrame({
        "start": idx_series.groupby(group_id).first(),
        "end": idx_series.groupby(group_id).last(),
        "regime": regime.groupby(group_id).first(),
    }).reset_index(drop=True)
    return segments


def build_price_chart(df: pd.DataFrame, title: str = "BTC-USD — Price & Detected Regime") -> go.Figure:
    """Candlestick + EMA50/EMA200 overlay + volume subplot, with the plot
    background shaded green/red over Bull Run / Bear-Crash regime episodes."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.74, 0.26], vertical_spacing=0.03,
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="BTC-USD",
        increasing_line_color=config.REGIME_COLORS["Bull Run"],
        decreasing_line_color=config.REGIME_COLORS["Bear/Crash"],
        increasing_fillcolor=config.REGIME_COLORS["Bull Run"],
        decreasing_fillcolor=config.REGIME_COLORS["Bear/Crash"],
    ), row=1, col=1)

    if "EMA50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA50"], name="EMA 50", mode="lines",
            line=dict(color=COLOR_EMA_FAST, width=2),
        ), row=1, col=1)
    if "EMA200" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA200"], name="EMA 200", mode="lines",
            line=dict(color=COLOR_EMA_SLOW, width=2),
        ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker=dict(color=COLOR_VOLUME), showlegend=False,
    ), row=2, col=1)

    segments = compute_regime_segments(df)
    shaded = segments[segments["regime"].isin([config.BULL_STATE_LABEL, config.BEAR_STATE_LABEL])]
    # Built as plain shape dicts and assigned once via update_layout(shapes=...) rather
    # than looping fig.add_vrect(..., row="all") — the latter is only fine for a handful
    # of shapes; with hundreds+ of regime segments it re-resolves subplot references on
    # every call and becomes minutes-slow. yref="paper" (y0=0..y1=1) reproduces the
    # row="all" effect (spans both subplots) without that per-call cost.
    regime_shapes = [
        dict(
            type="rect", xref="x", yref="paper",
            x0=seg.start, x1=seg.end + BAR_INTERVAL, y0=0, y1=1,
            fillcolor=config.REGIME_COLORS[seg.regime],
            opacity=config.REGIME_SHADE_OPACITY[seg.regime],
            line_width=0, layer="below",
        )
        for seg in shaded.itertuples(index=False)
    ]
    fig.update_layout(shapes=regime_shapes)

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=90, b=10),
        height=650,
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=30, label="30d", step="day", stepmode="backward"),
                    dict(count=90, label="90d", step="day", stepmode="backward"),
                    dict(count=180, label="180d", step="day", stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                x=1, xanchor="right", y=1.10, yanchor="bottom",
            )
        ),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    # Log scale: Yahoo's crypto volume data spans a very wide range (including many
    # zero/near-zero bars, see data_loader.add_features), which flattens a linear axis.
    fig.update_yaxes(title_text="Volume (log)", type="log", row=2, col=1)
    return fig


def build_equity_chart(df: pd.DataFrame, initial_capital: float) -> go.Figure:
    """Strategy equity (accent) vs. Buy & Hold (de-emphasis gray) on top,
    drawdown % as a shaded area beneath — shares one x-axis."""
    buy_hold = initial_capital * (df["Close"] / df["Close"].iloc[0])
    running_max = df["Equity"].cummax()
    drawdown_pct = (df["Equity"] - running_max) / running_max * 100.0

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.04,
        subplot_titles=("Equity: Strategy vs. Buy & Hold", "Drawdown"),
    )

    fig.add_trace(go.Scatter(
        x=df.index, y=buy_hold, name="Buy & Hold (1x)", mode="lines",
        line=dict(color=COLOR_BUY_HOLD, width=2, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Equity"], name=f"Strategy ({config.LEVERAGE}x)", mode="lines",
        line=dict(color=COLOR_STRATEGY, width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=drawdown_pct, name="Drawdown", mode="lines",
        line=dict(color=config.REGIME_COLORS["Bear/Crash"], width=2),
        fill="tozeroy",
        fillcolor="rgba(208, 59, 59, 0.10)",
        showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Equity (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return fig
