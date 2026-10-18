"""Trade flow imbalance calculator.

Maintains incremental exponentially decayed buy and sell volume accumulators
with log sizing and time-based half-life decay.
"""

from __future__ import annotations

import math
from typing import Sequence

from framework.base.stream.models import Trade


class TradeFlowImbalance:
    """Calculates trade flow imbalance from incoming public market trades."""

    def __init__(self, window_size: int = 500, half_life_s: float = 30.0) -> None:
        """Initialize the trade flow calculator.

        Args:
            window_size: Nominal trade horizon used for step decay rate.
            half_life_s: Time-based exponential decay half-life in seconds.
        """
        self.window_size = max(10, window_size)
        self.half_life_s = max(0.1, half_life_s)
        self._step_decay = math.exp(-0.005)
        self._time_decay_rate = math.log(0.5) / self.half_life_s

        self._delta_buys: float = 0.0
        self._delta_sells: float = 0.0
        self._last_time_ms: int = 0
        self._trade_count: int = 0

    def update_trades(self, trades: Sequence[Trade]) -> None:
        """Update decayed volume accumulators with new trades in O(1) per trade.

        Args:
            trades: Sequence of executed trades from TradeMsg.
        """
        for trade in trades:
            if trade.size <= 0.0:
                continue

            decay = self._step_decay
            if self._last_time_ms > 0 and trade.time_ms > self._last_time_ms:
                elapsed_s = (trade.time_ms - self._last_time_ms) / 1000.0
                decay *= math.exp(self._time_decay_rate * elapsed_s)

            self._delta_buys *= decay
            self._delta_sells *= decay

            vol = math.log1p(trade.size)
            if trade.is_buy:
                self._delta_buys += vol
            else:
                self._delta_sells += vol

            if trade.time_ms > self._last_time_ms:
                self._last_time_ms = trade.time_ms
            self._trade_count = min(self.window_size, self._trade_count + 1)

    def clear(self) -> None:
        """Reset all accumulators."""
        self._delta_buys = 0.0
        self._delta_sells = 0.0
        self._last_time_ms = 0
        self._trade_count = 0

    @property
    def trade_count(self) -> int:
        """Current count of trades observed (capped at window_size)."""
        return self._trade_count

    @property
    def last_time_ms(self) -> int:
        """Timestamp in milliseconds of the most recent trade."""
        return self._last_time_ms

    def imbalance(self, now_ms: int | None = None) -> float:
        """Calculate normalized trade flow imbalance in [-1.0, 1.0].

        Args:
            now_ms: Optional current timestamp in milliseconds to fade stale flow
                after a quiet period without trades.

        Returns:
            float: Imbalance in [-1.0, 1.0], where 1.0 indicates net buy aggression and
                -1.0 indicates net sell aggression.
        """
        total = self._delta_buys + self._delta_sells
        if total <= 0.0:
            return 0.0

        raw = (self._delta_buys - self._delta_sells) / total
        if now_ms is not None and self._last_time_ms > 0 and now_ms > self._last_time_ms:
            quiet_s = (now_ms - self._last_time_ms) / 1000.0
            raw *= math.exp(self._time_decay_rate * quiet_s)

        return max(-1.0, min(1.0, raw))
