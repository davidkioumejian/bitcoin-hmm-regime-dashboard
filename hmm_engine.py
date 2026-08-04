"""
hmm_engine.py
The Core Engine: fits a GaussianHMM on (Returns, Range, VolumeVolatility) to
detect market regimes, then automatically labels the states by ranking them
on mean return so the strategy can reason about "Bull Run" / "Bear/Crash"
without hardcoding which numeric state id they map to.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

import config


@dataclass
class RegimeDetector:
    n_components: int = config.N_COMPONENTS
    covariance_type: str = config.COVARIANCE_TYPE
    n_iter: int = config.N_ITER
    random_state: int = config.RANDOM_STATE
    smoothing_hours: int = config.REGIME_SMOOTHING_HOURS

    model: GaussianHMM = field(init=False, default=None)
    scaler: StandardScaler = field(init=False, default=None)
    state_labels: dict = field(init=False, default_factory=dict)
    state_stats: pd.DataFrame = field(init=False, default=None)
    bull_state: int = field(init=False, default=None)
    bear_state: int = field(init=False, default=None)

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the HMM on df[HMM_FEATURES] and return df with State/Regime columns."""
        X = df[config.HMM_FEATURES].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = self._fit_model(X_scaled)

        raw_states = self.model.predict(X_scaled)
        states = self._smooth_states(raw_states)
        out = df.copy()
        out["State"] = states
        self._label_states(out)
        out["Regime"] = out["State"].map(self.state_labels)
        return out

    def _smooth_states(self, states: np.ndarray) -> np.ndarray:
        """
        Causal rolling-mode filter over the Viterbi-decoded state path: each bar's
        smoothed state is the most frequent raw state in the trailing `smoothing_hours`
        window (current bar included, no future data used).

        Raw hourly Viterbi output is dominated by bar-to-bar noise (empirically ~5,600
        state flips / avg 3-bar segments across a 730-day BTC-USD window) — not
        meaningful multi-hour "regimes." Smoothing turns the decoded path into
        something that actually behaves like a regime, and incidentally keeps the
        chart's shaded-segment count and the strategy's trade frequency sane.
        """
        window = self.smoothing_hours
        n = len(states)
        if window <= 1:
            return states.copy()

        one_hot = np.zeros((n, self.n_components), dtype=np.int32)
        one_hot[np.arange(n), states] = 1
        cumsum = np.vstack([np.zeros((1, self.n_components), dtype=np.int64), np.cumsum(one_hot, axis=0)])

        idx = np.arange(n)
        lo = np.clip(idx - window + 1, 0, n)
        hi = idx + 1
        counts = cumsum[hi] - cumsum[lo]
        return counts.argmax(axis=1)

    def _fit_model(self, X_scaled: np.ndarray) -> GaussianHMM:
        """Fit GaussianHMM, falling back to diagonal covariance if 'full' is
        numerically unstable (a known failure mode with limited data per state)."""
        try:
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            model.fit(X_scaled)
            if not np.all(np.isfinite(model.means_)):
                raise ValueError("Non-finite means after fit")
            return model
        except Exception:
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            model.fit(X_scaled)
            return model

    def _label_states(self, df: pd.DataFrame) -> None:
        """
        Rank hidden states by mean Returns (descending):
          - highest mean return -> 'Bull Run'
          - lowest mean return  -> 'Bear/Crash'
          - everything else     -> 'Neutral'
        Only these two extremes are decision-relevant to the strategy (entry
        gates on Bull Run, hard-exits on Bear/Crash); the remaining states are
        intentionally collapsed into one 'Neutral' bucket rather than an
        invented gradient of labels. Also stores a per-state summary table
        (mean return, volatility, count) for transparency in the UI.
        """
        grouped = df.groupby("State")["Returns"]
        stats = grouped.agg(Mean_Return="mean", Volatility="std", Count="count")
        stats = stats.sort_values("Mean_Return", ascending=False)
        self.state_stats = stats

        order = stats.index.tolist()
        n = len(order)
        labels = {}
        for rank, state in enumerate(order):
            if rank == 0:
                labels[state] = config.BULL_STATE_LABEL
            elif rank == n - 1:
                labels[state] = config.BEAR_STATE_LABEL
            else:
                labels[state] = "Neutral"

        self.state_labels = labels
        self.bull_state = order[0]
        self.bear_state = order[-1]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply an already-fitted model/scaler to (possibly new) data."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("RegimeDetector.fit() must be called before predict().")
        X_scaled = self.scaler.transform(df[config.HMM_FEATURES].values)
        raw_states = self.model.predict(X_scaled)
        states = self._smooth_states(raw_states)
        out = df.copy()
        out["State"] = states
        out["Regime"] = out["State"].map(self.state_labels)
        return out
