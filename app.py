"""
app.py
Regime-Based Trading Dashboard — Streamlit front end.
Pipeline: data_loader -> hmm_engine -> strategy -> backtester -> this UI.
"""

import pandas as pd
import streamlit as st

import backtester
import charts
import config
import data_loader
import hmm_engine
import strategy

st.set_page_config(
    page_title="Regime-Based Trading Dashboard",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.regime-badge { border-radius: 10px; padding: 18px 22px; border-left: 5px solid; height: 100%; }
.regime-badge .rb-label { font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .04em; color: var(--text-color); opacity: .65; }
.regime-badge .rb-value { font-size: 30px; font-weight: 700; margin-top: 4px; line-height: 1.2; }
.regime-badge .rb-sub { font-size: 13px; margin-top: 6px; color: var(--text-color); opacity: .75; }
.badge-bull { background: rgba(12,163,12,0.12); border-color: #0ca30c; }
.badge-bull .rb-value { color: #0ca30c; }
.badge-bear { background: rgba(208,59,59,0.12); border-color: #d03b3b; }
.badge-bear .rb-value { color: #d03b3b; }
.badge-neutral { background: rgba(137,135,129,0.14); border-color: #898781; }
.badge-neutral .rb-value { color: var(--text-color); }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Data")
ticker = st.sidebar.text_input("Ticker", value=config.TICKER)
lookback_days = st.sidebar.slider("Lookback (days)", min_value=90, max_value=730,
                                   value=config.LOOKBACK_DAYS, step=10,
                                   help="yfinance caps hourly history at 730 days.")

st.sidebar.subheader("HMM Regime Engine")
n_components = st.sidebar.slider("Number of HMM components", min_value=3, max_value=10,
                                  value=config.N_COMPONENTS)
random_state = st.sidebar.number_input("Random seed", min_value=0, max_value=9999,
                                        value=config.RANDOM_STATE, step=1)
smoothing_hours = st.sidebar.slider("Regime smoothing (hours)", min_value=1, max_value=72,
                                     value=config.REGIME_SMOOTHING_HOURS,
                                     help="Causal rolling-mode filter on the decoded state path. "
                                          "Raw hourly Viterbi output flips almost every 3 bars; "
                                          "smoothing turns it into something that behaves like an "
                                          "actual multi-hour regime. 1 = no smoothing.")

st.sidebar.subheader("Strategy (8-Confirmation Vote)")
min_votes = st.sidebar.slider("Minimum confirmations required", min_value=1, max_value=8,
                               value=config.MIN_VOTES_REQUIRED,
                               help="Entry requires Regime == Bull Run AND at least this many of the 8 conditions.")

st.sidebar.subheader("Risk Management")
leverage = st.sidebar.slider("Leverage", min_value=1.0, max_value=5.0,
                              value=config.LEVERAGE, step=0.1)
cooldown_hours = st.sidebar.slider("Cooldown after exit (hours)", min_value=0, max_value=168,
                                    value=config.COOLDOWN_HOURS, step=1)
initial_capital = st.sidebar.number_input("Initial capital ($)", min_value=100.0,
                                           value=config.INITIAL_CAPITAL, step=500.0)
fee_pct = st.sidebar.slider("Fee per side (%)", min_value=0.0, max_value=0.50,
                             value=config.FEE_PCT_PER_SIDE, step=0.01,
                             help="Applied on both entry and exit; set to 0 for a frictionless backtest.")

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data", width="stretch"):
    st.cache_data.clear()
    st.rerun()


# --------------------------------------------------------------------------
# Cached pipeline
# --------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner="Downloading BTC-USD hourly data from Yahoo Finance...")
def load_data(ticker: str, days: int) -> pd.DataFrame:
    return data_loader.load_market_data(ticker, days)


@st.cache_resource(show_spinner="Training the 7-state Gaussian HMM regime model...")
def fit_regime_model(df: pd.DataFrame, n_components: int, random_state: int, smoothing_hours: int):
    detector = hmm_engine.RegimeDetector(n_components=n_components, random_state=random_state,
                                          smoothing_hours=smoothing_hours)
    labeled_df = detector.fit(df)
    return detector, labeled_df


@st.cache_data(show_spinner="Computing indicators and confirmation votes...")
def compute_strategy(df: pd.DataFrame, min_votes: int):
    return strategy.run_strategy_pipeline(df, min_votes)


@st.cache_data(show_spinner="Running backtest simulation...")
def run_backtest(df: pd.DataFrame, initial_capital: float, leverage: float,
                  cooldown_hours: int, fee_pct: float):
    bt = backtester.Backtester(initial_capital, leverage, cooldown_hours, fee_pct)
    return bt.run(df)


try:
    raw_df = load_data(ticker, lookback_days)
except data_loader.DataLoadError as e:
    st.error(f"**Data error:** {e}")
    st.stop()
except Exception as e:
    st.error(f"**Unexpected error fetching market data:** {e}")
    st.stop()

try:
    detector, labeled_df = fit_regime_model(raw_df, n_components, int(random_state), int(smoothing_hours))
    signal_df, votes_df = compute_strategy(labeled_df, min_votes)
    result_df, trades_df, metrics, open_position = run_backtest(
        signal_df, initial_capital, leverage, int(cooldown_hours), fee_pct
    )
except Exception as e:
    st.error(f"**Pipeline error:** {e}")
    st.stop()

latest = result_df.iloc[-1]
latest_votes = votes_df.iloc[-1]


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("📈 Regime-Based Trading Dashboard")
st.caption(
    f"{ticker} · Hourly · {result_df.index[0]:%Y-%m-%d} → {result_df.index[-1]:%Y-%m-%d %H:%M UTC} "
    f"· {len(result_df):,} bars · Last close ${latest['Close']:,.2f}"
)

# --------------------------------------------------------------------------
# Top section: Current Signal + Detected Regime
# --------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    if latest["InPosition"]:
        sub = (f"Long since {open_position['entry_time']:%Y-%m-%d %H:%M} · "
               f"Unrealized {open_position['unrealized_pnl_pct']:+.2f}%") if open_position else "Long"
        st.markdown(f"""<div class="regime-badge badge-bull">
            <div class="rb-label">Current Signal</div>
            <div class="rb-value">LONG</div>
            <div class="rb-sub">{sub}</div></div>""", unsafe_allow_html=True)
    elif latest["InCooldown"]:
        st.markdown(f"""<div class="regime-badge badge-neutral">
            <div class="rb-label">Current Signal</div>
            <div class="rb-value">COOLDOWN</div>
            <div class="rb-sub">Re-entry blocked (48h post-exit cooldown active)</div></div>""",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="regime-badge badge-neutral">
            <div class="rb-label">Current Signal</div>
            <div class="rb-value">CASH</div>
            <div class="rb-sub">Flat — awaiting Bull Run regime + {min_votes}/8 confirmations</div></div>""",
                    unsafe_allow_html=True)

with col2:
    regime = latest["Regime"]
    badge_class = {"Bull Run": "badge-bull", "Bear/Crash": "badge-bear"}.get(regime, "badge-neutral")
    vote_count = int(latest["VoteCount"])
    st.markdown(f"""<div class="regime-badge {badge_class}">
        <div class="rb-label">Detected Regime</div>
        <div class="rb-value">{regime}</div>
        <div class="rb-sub">{vote_count}/8 confirmations currently met</div></div>""",
                unsafe_allow_html=True)

st.write("")

# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Return", f"{metrics['total_return_pct']:+.2f}%",
          f"${metrics['final_capital'] - metrics['initial_capital']:+,.0f}")
m2.metric("Alpha vs Buy & Hold", f"{metrics['alpha_pct']:+.2f}pp",
          f"B&H {metrics['buy_hold_return_pct']:+.2f}%", delta_color="off")
m3.metric("Win Rate", f"{metrics['win_rate_pct']:.1f}%",
          f"{metrics['wins']}W / {metrics['losses']}L", delta_color="off")
m4.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")

s1, s2, s3, s4 = st.columns(4)
s1.metric("Total Trades", f"{metrics['num_trades']}")
s2.metric("Avg Trade PnL", f"{metrics['avg_trade_pct']:+.2f}%")
s3.metric("Sharpe (ann., illustrative)", f"{metrics['sharpe']:.2f}")
s4.metric("Final Capital", f"${metrics['final_capital']:,.0f}")

st.divider()

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_chart, tab_equity, tab_trades, tab_signal = st.tabs(
    ["📈 Chart & Regimes", "💰 Equity Curve", "📋 Trade Log", "🗳️ Signal Detail"]
)

with tab_chart:
    fig = charts.build_price_chart(result_df)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Background shading marks HMM-detected regime episodes — "
        "🟩 green = Bull Run (highest mean-return state), 🟥 red = Bear/Crash (lowest mean-return state). "
        "Unshaded periods are the Neutral bucket (all other states)."
    )

with tab_equity:
    eq_fig = charts.build_equity_chart(result_df, initial_capital)
    st.plotly_chart(eq_fig, width="stretch")

with tab_trades:
    if open_position is not None:
        st.info(
            f"**Open position** — entered {open_position['entry_time']:%Y-%m-%d %H:%M} "
            f"@ ${open_position['entry_price']:,.2f} · "
            f"unrealized PnL (leveraged): {open_position['unrealized_pnl_pct']:+.2f}%"
        )

    if len(trades_df) == 0:
        st.info("No closed trades yet for the current configuration.")
    else:
        cols_map = {
            "entry_time": "Entry Time", "entry_price": "Entry Price",
            "exit_time": "Exit Time", "exit_price": "Exit Price",
            "exit_reason": "Exit Reason", "pnl_pct": "PnL %",
            "pnl_dollars": "PnL $", "duration_hours": "Duration (h)",
        }
        display_trades = trades_df[list(cols_map.keys())].rename(columns=cols_map).iloc[::-1]

        def _pnl_color(val):
            color = config.REGIME_COLORS["Bull Run"] if val > 0 else config.REGIME_COLORS["Bear/Crash"]
            return f"color: {color}; font-weight: 600;"

        styled = (
            display_trades.style
            .map(_pnl_color, subset=["PnL %", "PnL $"])
            .format({
                "Entry Time": lambda t: t.strftime("%Y-%m-%d %H:%M"),
                "Exit Time": lambda t: t.strftime("%Y-%m-%d %H:%M"),
                "Entry Price": "${:,.2f}".format,
                "Exit Price": "${:,.2f}".format,
                "PnL %": "{:+.2f}%".format,
                "PnL $": "${:+,.2f}".format,
                "Duration (h)": "{:.1f}".format,
            })
        )
        st.caption(f"{len(trades_df)} closed trades (most recent first).")
        st.dataframe(styled, width="stretch", hide_index=True)

with tab_signal:
    st.subheader("Latest bar — 8-Confirmation Vote Breakdown")
    rows = [
        ("RSI < 90", f"{latest['RSI']:.1f}", "< 90"),
        ("Momentum > 1%", f"{latest['Momentum']:+.2f}%", "> 1%"),
        ("Volatility < 6%", f"{latest['VolatilityPct']:.2f}%", "< 6%"),
        ("Volume > 20-SMA", f"{latest['Volume']:,.0f}", f"> {latest['VolumeSMA']:,.0f}"),
        ("ADX > 25", f"{latest['ADX']:.1f}", "> 25"),
        ("Price > EMA50", f"${latest['Close']:,.2f}", f"> ${latest['EMA50']:,.2f}"),
        ("Price > EMA200", f"${latest['Close']:,.2f}", f"> ${latest['EMA200']:,.2f}"),
        ("MACD > Signal", f"{latest['MACD']:.2f}", f"> {latest['MACD_Signal']:.2f}"),
    ]
    vote_table = pd.DataFrame(rows, columns=["Condition", "Current Value", "Threshold"])
    vote_table["Met?"] = [("✅" if latest_votes[c] else "❌") for c, _, _ in rows]
    st.dataframe(vote_table, width="stretch", hide_index=True)
    st.caption(
        f"**{int(latest['VoteCount'])} / 8** confirmations met · "
        f"entry additionally requires Regime == **Bull Run** (currently **{latest['Regime']}**)."
    )

    st.subheader("HMM State Summary")
    stats = detector.state_stats.copy()
    stats.insert(0, "Regime Label", stats.index.map(detector.state_labels))
    stats["Mean_Return"] = stats["Mean_Return"] * 100
    stats = stats.rename(columns={"Mean_Return": "Mean Return (%)", "Volatility": "Volatility (σ)"})
    st.dataframe(
        stats.style.format({"Mean Return (%)": "{:+.3f}", "Volatility (σ)": "{:.4f}", "Count": "{:,.0f}"}),
        width="stretch",
    )
    st.caption(
        "One row per hidden state, ranked by mean return. The highest-return state is auto-labeled "
        "'Bull Run', the lowest 'Bear/Crash'; every other state collapses to 'Neutral'."
    )

st.divider()
with st.expander("ℹ️ Methodology & assumptions"):
    st.markdown(f"""
- **Regime detection** is fit **in-sample** on the full {lookback_days}-day window shown (a `GaussianHMM`
  with {n_components} components on standardized Returns / Range / Volume-Volatility) — this demonstrates
  regime identification, not a walk-forward/out-of-sample trading system.
- **Entry** requires Regime == Bull Run **and** ≥ {min_votes}/8 confirmations on the same bar.
- **Exit** is immediate on a Bear/Crash regime flip — no take-profit or trailing stop is modeled.
- **Leverage** ({leverage}x) is applied to the % price move for PnL purposes only; unrealized loss is floored
  at -100% of allocated capital per position (simplified liquidation floor, no margin calls modeled).
- **Fees** of {fee_pct:.2f}% per side are deducted from each round-trip trade; set to 0 in the sidebar for a
  frictionless comparison.
- **Cooldown** of {cooldown_hours}h is enforced after every exit before a new entry is considered.
- Not financial advice — a backtest, not a live trading system.
""")
