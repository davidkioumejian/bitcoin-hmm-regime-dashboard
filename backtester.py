"""
backtester.py
Runs the bar-by-bar simulation: enters Long when strategy.EntryGate fires,
exits immediately on a Bear/Crash regime flip, applies 2.5x leverage to PnL,
and enforces a hard 48-hour cooldown after any exit before re-entry.
Logs every closed trade and produces the equity curve + summary metrics.
"""

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

import config


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    pnl_pct: float          # leveraged, net-of-fees % return on this trade
    pnl_dollars: float
    capital_before: float
    capital_after: float
    duration_hours: float


class Backtester:
    def __init__(self,
                 initial_capital: float = config.INITIAL_CAPITAL,
                 leverage: float = config.LEVERAGE,
                 cooldown_hours: int = config.COOLDOWN_HOURS,
                 fee_pct_per_side: float = config.FEE_PCT_PER_SIDE):
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.cooldown_hours = cooldown_hours
        self.fee_frac = fee_pct_per_side / 100.0

    def run(self, df: pd.DataFrame):
        """
        df must already contain: Close, Regime, EntryGate, RegimeExit.
        Returns (df_with_equity, trades_df, metrics_dict, open_position_dict_or_None).
        """
        n = len(df)
        times = df.index
        closes = df["Close"].to_numpy()
        entry_gate = df["EntryGate"].to_numpy()
        regime_exit = df["RegimeExit"].to_numpy()

        capital = self.initial_capital
        position: Optional[dict] = None
        cooldown_until: Optional[pd.Timestamp] = None
        trades: list[Trade] = []

        equity = np.empty(n)
        in_position = np.zeros(n, dtype=bool)
        in_cooldown = np.zeros(n, dtype=bool)

        for i in range(n):
            t = times[i]
            price = closes[i]

            # 1) Hard exit rule: regime flipped to Bear/Crash while in a position.
            if position is not None and regime_exit[i]:
                gross_ret = (price / position["entry_price"] - 1.0) * self.leverage
                net_ret = gross_ret - 2 * self.fee_frac  # entry + exit fee, approximated
                net_ret = max(net_ret, -1.0)             # floor: can't lose more than 100% of allocated capital

                capital_before = capital
                capital = capital * (1.0 + net_ret)
                duration_h = (t - position["entry_time"]).total_seconds() / 3600.0

                trades.append(Trade(
                    entry_time=position["entry_time"],
                    entry_price=position["entry_price"],
                    exit_time=t,
                    exit_price=price,
                    exit_reason="Regime flip to Bear/Crash",
                    pnl_pct=net_ret * 100.0,
                    pnl_dollars=capital - capital_before,
                    capital_before=capital_before,
                    capital_after=capital,
                    duration_hours=duration_h,
                ))
                position = None
                cooldown_until = t + pd.Timedelta(hours=self.cooldown_hours)

            # 2) Entry: only when flat, not in cooldown, and the strategy gate fires.
            if position is None:
                blocked_by_cooldown = cooldown_until is not None and t < cooldown_until
                if not blocked_by_cooldown and entry_gate[i]:
                    position = {"entry_time": t, "entry_price": price}
                in_cooldown[i] = blocked_by_cooldown

            # 3) Mark-to-market equity for this bar.
            if position is not None:
                unrealized = (price / position["entry_price"] - 1.0) * self.leverage
                unrealized = max(unrealized, -1.0)
                equity[i] = capital * (1.0 + unrealized)
                in_position[i] = True
            else:
                equity[i] = capital

        out = df.copy()
        out["Equity"] = equity
        out["InPosition"] = in_position
        out["InCooldown"] = in_cooldown

        trades_df = pd.DataFrame([asdict(tr) for tr in trades])

        open_position = None
        if position is not None:
            last_price = closes[-1]
            unrealized_pct = (last_price / position["entry_price"] - 1.0) * self.leverage * 100.0
            open_position = {
                "entry_time": position["entry_time"],
                "entry_price": position["entry_price"],
                "last_price": last_price,
                "unrealized_pnl_pct": unrealized_pct,
            }

        metrics = self._compute_metrics(out, trades_df, capital)
        return out, trades_df, metrics, open_position

    def _compute_metrics(self, df: pd.DataFrame, trades_df: pd.DataFrame, final_capital: float) -> dict:
        total_return_pct = (final_capital / self.initial_capital - 1.0) * 100.0

        buy_hold_return_pct = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1.0) * 100.0
        alpha_pct = total_return_pct - buy_hold_return_pct

        num_trades = len(trades_df)
        if num_trades > 0:
            wins = int((trades_df["pnl_pct"] > 0).sum())
            win_rate_pct = wins / num_trades * 100.0
            avg_trade_pct = float(trades_df["pnl_pct"].mean())
            avg_win_pct = float(trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].mean()) if wins > 0 else 0.0
            losses = num_trades - wins
            avg_loss_pct = float(trades_df.loc[trades_df["pnl_pct"] <= 0, "pnl_pct"].mean()) if losses > 0 else 0.0
        else:
            wins = 0
            win_rate_pct = 0.0
            avg_trade_pct = 0.0
            avg_win_pct = 0.0
            avg_loss_pct = 0.0

        equity = df["Equity"]
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown_pct = float(drawdown.min() * 100.0)

        hourly_ret = equity.pct_change().dropna()
        if hourly_ret.std() > 0:
            sharpe = float((hourly_ret.mean() / hourly_ret.std()) * np.sqrt(24 * 365))
        else:
            sharpe = 0.0

        return {
            "total_return_pct": total_return_pct,
            "buy_hold_return_pct": buy_hold_return_pct,
            "alpha_pct": alpha_pct,
            "win_rate_pct": win_rate_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "num_trades": num_trades,
            "wins": wins,
            "losses": num_trades - wins,
            "avg_trade_pct": avg_trade_pct,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "sharpe": sharpe,
            "final_capital": final_capital,
            "initial_capital": self.initial_capital,
        }
