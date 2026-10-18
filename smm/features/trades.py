"""Trade flow imbalance calculator.

Maintains a bounded window of recent trade executions and computes normalized
aggressive buy vs sell volume delta, applying log sizing and exponential decay.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Sequence

from framework.base.stream.models import Trade


class TradeFlowImbalance:
    """Calculates trade flow imbalance from incoming public market trades."""

    def __init__(self, window_size: int = 500) -> None:
        """Initialize the trade flow calculator.

        Args:
            window_size: Maximum number of recent trades retained in memory.
        """
        self.window_size = max(10, window_size)
        # Store tuples of (size, is_buy)
        self._trades: deque[tuple[float, bool]] = deque(maxlen=self.window_size)

    def update_trades(self, trades: Sequence[Trade]) -> None:
        """Append new trades to the sliding window.

        Args:
            trades: Sequence of executed trades from TradeMsg.
        """
        for trade in trades:
            if trade.size > 0.0:
                self._trades.append((trade.size, trade.is_buy))

    def clear(self) -> None:
        """Clear all stored trades."""
        self._trades.clear()

    @property
    def trade_count(self) -> int:
        """Current count of trades in the window."""
        return len(self._trades)

    def imbalance(self) -> float:
        """Calculate normalized trade flow imbalance in [-1.0, 1.0].

        Recent trades are weighted exponentially more heavily. Each trade volume is
        scaled by ln(1 + size) to dampen extreme whale outliers.

        Returns:
            float: Imbalance in [-1.0, 1.0], where 1.0 indicates net buy aggression and
                -1.0 indicates net sell aggression.
        """
        n = len(self._trades)
        if n == 0:
            return 0.0

        # Weights: index 0 (oldest) has smallest weight, index n-1 (newest) has weight 1.0
        # decay per step: exp(-0.01 * (n - 1 - i))
        delta_buys = 0.0
        delta_sells = 0.0

        for i, (size, is_buy) in enumerate(self._trades):
            steps_from_newest = n - 1 - i
            weight = math.exp(-0.005 * steps_from_newest)
            weighted_vol = math.log1p(size) * weight

            if is_buy:
                delta_buys += weighted_vol
            else:
                delta_sells += weighted_vol

        total = delta_buys + delta_sells
        if total <= 0.0:
            return 0.0

        return max(-1.0, min(1.0, (delta_buys - delta_sells) / total))
