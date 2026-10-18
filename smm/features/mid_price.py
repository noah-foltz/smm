"""Weighted mid-price and fair value calculators."""

from __future__ import annotations

import math
from typing import Sequence

from framework.base.stream.models import OrderbookLevel


def weighted_mid_price(
    bids: Sequence[OrderbookLevel],
    asks: Sequence[OrderbookLevel],
    fallback_mid: float = 0.0,
) -> float:
    """Calculate the volume-weighted mid price (WMID) from top-of-book levels.

    WMID leans toward the opposite side of higher volume (e.g. if best bid size is
    significantly higher than best ask size, WMID tilts higher toward the ask):
        WMID = (best_bid_price * best_ask_size + best_ask_price * best_bid_size) / (best_bid_size + best_ask_size)

    Args:
        bids: Bid levels (best bid is bids[-1]).
        asks: Ask levels (best ask is asks[0]).
        fallback_mid: Fallback mid-price if book data is insufficient.

    Returns:
        float: Volume-weighted mid price.
    """
    if not bids or not asks:
        return fallback_mid

    best_bid = bids[-1]
    best_ask = asks[0]
    total_qty = best_bid.size + best_ask.size

    if total_qty <= 0.0 or best_bid.size <= 0.0 or best_ask.size <= 0.0:
        return (best_bid.price + best_ask.price) / 2.0

    return (best_bid.price * best_ask.size + best_ask.price * best_bid.size) / total_qty


def volume_adjusted_mid_price(
    bids: Sequence[OrderbookLevel],
    asks: Sequence[OrderbookLevel],
    depth: int = 10,
    fallback_mid: float = 0.0,
) -> float:
    """Calculate Volume-Adjusted Mid Price (VAMP) over top-N orderbook depth.

    Computes the arithmetic average of bid VWAP and ask VWAP over the top-N levels.

    Args:
        bids: Bid levels (best bids at the end).
        asks: Ask levels (best asks at the start).
        depth: Number of levels from top-of-book to consider per side.
        fallback_mid: Fallback price if book data is insufficient.

    Returns:
        float: Volume-adjusted mid price.
    """
    if not bids or not asks or depth <= 0:
        return fallback_mid

    top_bids = bids[-depth:] if len(bids) > depth else bids
    top_asks = asks[:depth] if len(asks) > depth else asks

    bid_qty = sum(lvl.size for lvl in top_bids)
    ask_qty = sum(lvl.size for lvl in top_asks)

    if bid_qty <= 0.0 or ask_qty <= 0.0:
        return weighted_mid_price(bids, asks, fallback_mid)

    bid_vwap = sum(lvl.price * lvl.size for lvl in top_bids) / bid_qty
    ask_vwap = sum(lvl.price * lvl.size for lvl in top_asks) / ask_qty

    return (bid_vwap + ask_vwap) / 2.0


def log_price_difference(follow: float, base: float) -> float:
    """Calculate logarithmic percentage price difference between two prices.

    A positive value indicates that follow price is higher than base price, while
    a negative value indicates that follow price is lower than base price.

    Args:
        follow: Target/reference price (e.g. lead exchange mid or mark price).
        base: Local price being compared against.

    Returns:
        float: Log price difference scaled by 100.
    """
    if follow <= 0.0 or base <= 0.0:
        return 0.0
    return math.log(follow / base) * 100.0
