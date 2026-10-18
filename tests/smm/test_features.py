"""Unit tests for microstructural feature extractors.

Tests cover:
- Top-of-book (BBA) volume imbalance
- Multi-depth orderbook imbalance with exponential decay weights
- Trade flow imbalance with aggressive delta tracking
- Volume-weighted mid price (WMID) and Volume-Adjusted Mid Price (VAMP)
- Log price difference calculations
"""

from __future__ import annotations

import math

from framework.base.stream.models import OrderbookLevel, Trade
from smm.features import (
    TradeFlowImbalance,
    bba_imbalance,
    log_price_difference,
    orderbook_imbalance,
    volume_adjusted_mid_price,
    weighted_mid_price,
)


class TestBbaImbalance:
    """Tests for top-of-book (BBA) imbalance."""

    def test_bid_dominance(self) -> None:
        """Heavier bid size produces positive imbalance."""
        bids = (OrderbookLevel(price=100.0, size=10.0),)
        asks = (OrderbookLevel(price=101.0, size=2.0),)
        imb = bba_imbalance(bids, asks)
        assert math.isclose(imb, (10.0 - 2.0) / 12.0)
        assert imb > 0.0

    def test_ask_dominance(self) -> None:
        """Heavier ask size produces negative imbalance."""
        bids = (OrderbookLevel(price=100.0, size=3.0),)
        asks = (OrderbookLevel(price=101.0, size=9.0),)
        imb = bba_imbalance(bids, asks)
        assert math.isclose(imb, (3.0 - 9.0) / 12.0)
        assert imb < 0.0

    def test_symmetric_sizes(self) -> None:
        """Equal bid and ask sizes produce 0.0 imbalance."""
        bids = (OrderbookLevel(price=100.0, size=5.0),)
        asks = (OrderbookLevel(price=101.0, size=5.0),)
        assert bba_imbalance(bids, asks) == 0.0

    def test_empty_levels(self) -> None:
        """Empty book levels return 0.0 safely."""
        assert bba_imbalance((), ()) == 0.0
        assert bba_imbalance((OrderbookLevel(price=100.0, size=1.0),), ()) == 0.0
        assert bba_imbalance((), (OrderbookLevel(price=101.0, size=1.0),)) == 0.0


class TestOrderbookImbalance:
    """Tests for multi-depth orderbook imbalance."""

    def test_multi_depth_bid_pressure(self) -> None:
        """Multi-level book with heavier bids returns positive imbalance."""
        bids = (
            OrderbookLevel(price=98.0, size=10.0),
            OrderbookLevel(price=99.0, size=8.0),
            OrderbookLevel(price=100.0, size=5.0),
        )
        asks = (
            OrderbookLevel(price=101.0, size=2.0),
            OrderbookLevel(price=102.0, size=2.0),
            OrderbookLevel(price=103.0, size=1.0),
        )
        imb = orderbook_imbalance(bids, asks, depths_bps=(50.0, 100.0, 200.0, 500.0))
        assert 0.0 < imb <= 1.0

    def test_multi_depth_ask_pressure(self) -> None:
        """Multi-level book with heavier asks returns negative imbalance."""
        bids = (
            OrderbookLevel(price=98.0, size=1.0),
            OrderbookLevel(price=99.0, size=1.0),
            OrderbookLevel(price=100.0, size=1.0),
        )
        asks = (
            OrderbookLevel(price=101.0, size=5.0),
            OrderbookLevel(price=102.0, size=10.0),
            OrderbookLevel(price=103.0, size=20.0),
        )
        imb = orderbook_imbalance(bids, asks, depths_bps=(50.0, 100.0, 200.0, 500.0))
        assert -1.0 <= imb < 0.0

    def test_symmetric_depth(self) -> None:
        """Symmetric book returns zero imbalance."""
        bids = (
            OrderbookLevel(price=98.0, size=4.0),
            OrderbookLevel(price=99.0, size=3.0),
            OrderbookLevel(price=100.0, size=2.0),
        )
        asks = (
            OrderbookLevel(price=101.0, size=2.0),
            OrderbookLevel(price=102.0, size=3.0),
            OrderbookLevel(price=103.0, size=4.0),
        )
        imb = orderbook_imbalance(bids, asks, depths_bps=(100.0, 200.0, 500.0))
        assert math.isclose(imb, 0.0, abs_tol=1e-5)

    def test_depth_scaling_bps(self) -> None:
        """Orders outside depth thresholds are not counted in tighter tiers."""
        # best_bid = 100.0, best_ask = 100.5
        # 10 bps from best_bid is 100.0 * (1 - 0.001) = 99.9.
        # So a bid at 95.0 is excluded from 10 bps tier.
        bids = (
            OrderbookLevel(price=95.0, size=100.0),
            OrderbookLevel(price=100.0, size=1.0),
        )
        asks = (
            OrderbookLevel(price=100.5, size=1.0),
            OrderbookLevel(price=105.0, size=100.0),
        )
        # Tight 10 bps window: only 100.0 (size 1) and 100.5 (size 1) qualify -> imb = 0.0
        imb_tight = orderbook_imbalance(bids, asks, depths_bps=(10.0,))
        assert math.isclose(imb_tight, 0.0, abs_tol=1e-5)


class TestTradeFlowImbalance:
    """Tests for trade flow delta tracking."""

    def test_all_buys_returns_one(self) -> None:
        """100% buy flow yields 1.0 imbalance."""
        flow = TradeFlowImbalance(window_size=20)
        trades = (
            Trade(time_ms=100, price=100.0, size=1.0, is_buy=True),
            Trade(time_ms=200, price=100.5, size=2.5, is_buy=True),
        )
        flow.update_trades(trades)
        assert flow.imbalance() == 1.0

    def test_all_sells_returns_minus_one(self) -> None:
        """100% sell flow yields -1.0 imbalance."""
        flow = TradeFlowImbalance(window_size=20)
        trades = (
            Trade(time_ms=100, price=100.0, size=1.0, is_buy=False),
            Trade(time_ms=200, price=99.5, size=2.5, is_buy=False),
        )
        flow.update_trades(trades)
        assert flow.imbalance() == -1.0

    def test_empty_returns_zero(self) -> None:
        """Empty trade flow yields 0.0."""
        flow = TradeFlowImbalance()
        assert flow.imbalance() == 0.0

    def test_recency_decay(self) -> None:
        """Newer trades outweigh older trades of the same size."""
        flow = TradeFlowImbalance(window_size=10)
        # First 5 trades are sells, next 5 trades are buys
        old_sells = tuple(
            Trade(time_ms=100 + i, price=100.0, size=1.0, is_buy=False)
            for i in range(5)
        )
        new_buys = tuple(
            Trade(time_ms=200 + i, price=100.0, size=1.0, is_buy=True)
            for i in range(5)
        )
        flow.update_trades(old_sells)
        flow.update_trades(new_buys)
        # Because buys are more recent, imbalance should be positive
        assert flow.imbalance() > 0.0

    def test_time_decay_after_quiet_period(self) -> None:
        """Trade flow imbalance fades toward zero after a quiet period."""
        flow = TradeFlowImbalance(window_size=10, half_life_s=10.0)
        flow.update_trades((Trade(time_ms=1_000, price=100.0, size=5.0, is_buy=True),))
        assert flow.imbalance(now_ms=1_000) == 1.0
        # After one half-life (10 seconds = 10,000 ms), imbalance halves to 0.5
        assert math.isclose(flow.imbalance(now_ms=11_000), 0.5, abs_tol=1e-5)
        # After 100 seconds, imbalance is near 0.0
        assert flow.imbalance(now_ms=101_000) < 0.001

class TestMidPriceFeatures:
    """Tests for weighted mid-price, VAMP, and log price difference."""

    def test_wmid_tilts_toward_ask_when_bids_heavy(self) -> None:
        """When bid volume outweighs ask volume, WMID tilts higher toward ask."""
        bids = (OrderbookLevel(price=100.0, size=10.0),)
        asks = (OrderbookLevel(price=102.0, size=2.0),)
        # Arithmetic mid is 101.0
        wmid = weighted_mid_price(bids, asks)
        # WMID = (100 * 2 + 102 * 10) / 12 = (200 + 1020) / 12 = 1220 / 12 = 101.6667
        assert math.isclose(wmid, 1220.0 / 12.0)
        assert wmid > 101.0

    def test_wmid_symmetric(self) -> None:
        """Equal bid and ask volumes give arithmetic mid."""
        bids = (OrderbookLevel(price=100.0, size=5.0),)
        asks = (OrderbookLevel(price=102.0, size=5.0),)
        assert math.isclose(weighted_mid_price(bids, asks), 101.0)

    def test_vamp_calculation(self) -> None:
        """VAMP computes average VWAP across top depth levels."""
        bids = (
            OrderbookLevel(price=98.0, size=2.0),
            OrderbookLevel(price=100.0, size=8.0),
        )
        asks = (
            OrderbookLevel(price=102.0, size=4.0),
            OrderbookLevel(price=104.0, size=6.0),
        )
        vamp = volume_adjusted_mid_price(bids, asks, depth=2)
        bid_vwap = (98.0 * 2.0 + 100.0 * 8.0) / 10.0  # 99.6
        ask_vwap = (102.0 * 4.0 + 104.0 * 6.0) / 10.0  # 103.2
        assert math.isclose(vamp, (bid_vwap + ask_vwap) / 2.0)

    def test_log_price_difference(self) -> None:
        """Log price difference reflects direction and magnitude accurately."""
        # 49000 follow vs 48980 base: follow is higher -> positive
        pos_diff = log_price_difference(49000.0, 48980.0)
        assert pos_diff > 0.0
        assert math.isclose(pos_diff, math.log(49000.0 / 48980.0) * 100.0)

        # 48980 follow vs 49000 base: follow is lower -> negative
        neg_diff = log_price_difference(48980.0, 49000.0)
        assert neg_diff < 0.0
        assert math.isclose(neg_diff, math.log(48980.0 / 49000.0) * 100.0)

        # Equal prices
        assert log_price_difference(50000.0, 50000.0) == 0.0
