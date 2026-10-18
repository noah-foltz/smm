"""Orderbook imbalance calculators.

Calculates top-of-book (BBA) volume imbalance and multi-depth orderbook
imbalance across basis-point depth horizons with exponential decay weights.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Sequence

from framework.base.stream.models import OrderbookLevel

DEFAULT_DEPTHS_BPS: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0)


@lru_cache(maxsize=16)
def _depth_weights(num_depths: int) -> tuple[float, ...]:
    """Precompute normalized exponential decay weights for depth tiers."""
    raw_weights = [math.exp(-0.5 * i) for i in range(num_depths)]
    sum_weights = sum(raw_weights)
    return tuple(w / sum_weights for w in raw_weights)

def bba_imbalance(
    bids: Sequence[OrderbookLevel],
    asks: Sequence[OrderbookLevel],
) -> float:
    """Calculate normalized best bid/ask (BBA) quantity imbalance.

    In the framework stream model, bids are ordered in strictly increasing price order
    (best bid is bids[-1]), and asks are ordered in strictly increasing price order
    (best ask is asks[0]).

    Args:
        bids: Bid levels.
        asks: Ask levels.

    Returns:
        float: Normalized imbalance in [-1.0, 1.0], where 1.0 indicates complete bid
            dominance (buying pressure) and -1.0 indicates complete ask dominance.
    """
    if not bids or not asks:
        return 0.0

    best_bid_size = bids[-1].size
    best_ask_size = asks[0].size
    total = best_bid_size + best_ask_size
    if total <= 0.0:
        return 0.0

    return (best_bid_size - best_ask_size) / total


def orderbook_imbalance(
    bids: Sequence[OrderbookLevel],
    asks: Sequence[OrderbookLevel],
    depths_bps: Sequence[float] = DEFAULT_DEPTHS_BPS,
) -> float:
    """Calculate multi-depth geometrically weighted orderbook imbalance.

    Depths are specified in basis points (BPS) and converted to decimals (d * 1e-4).
    For each depth tier, total bid volume within [best_bid * (1 - d), best_bid] and
    total ask volume within [best_ask, best_ask * (1 + d)] are summed, and the normalized
    imbalance (bid_vol - ask_vol) / (bid_vol + ask_vol) is weighted with exponential decay.

    Args:
        bids: Bid levels in strictly increasing price order.
        asks: Ask levels in strictly increasing price order.
        depths_bps: Price depths in basis points.

    Returns:
        float: Aggregated imbalance in [-1.0, 1.0].
    """
    if not bids or not asks or not depths_bps:
        return 0.0

    best_bid_p = bids[-1].price
    best_ask_p = asks[0].price
    if best_bid_p <= 0.0 or best_ask_p <= 0.0:
        return 0.0

    weights = _depth_weights(len(depths_bps))
    weighted_imbalance = 0.0

    bid_idx = len(bids) - 1
    ask_idx = 0
    num_asks = len(asks)
    bid_vol = 0.0
    ask_vol = 0.0
    prev_depth_bps = -math.inf

    for i, depth_bps in enumerate(depths_bps):
        if depth_bps < prev_depth_bps:
            bid_idx = len(bids) - 1
            ask_idx = 0
            bid_vol = 0.0
            ask_vol = 0.0
        prev_depth_bps = depth_bps

        depth_dec = depth_bps * 1e-4
        min_bid = best_bid_p * (1.0 - depth_dec)
        max_ask = best_ask_p * (1.0 + depth_dec)

        while bid_idx >= 0 and bids[bid_idx].price >= min_bid:
            bid_vol += bids[bid_idx].size
            bid_idx -= 1

        while ask_idx < num_asks and asks[ask_idx].price <= max_ask:
            ask_vol += asks[ask_idx].size
            ask_idx += 1

        total_vol = bid_vol + ask_vol
        tier_imb = (bid_vol - ask_vol) / total_vol if total_vol > 0.0 else 0.0
        weighted_imbalance += tier_imb * weights[i]
    return max(-1.0, min(1.0, weighted_imbalance))
