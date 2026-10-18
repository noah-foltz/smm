"""Unit tests for micro-feature skewing, WMID centering, and cross-venue pricing."""

from __future__ import annotations

import math
import pytest

from framework.base.common import (
    Asset,
    Instrument,
    InstrumentType,
    Symbol,
    Venue,
)
from framework.base.stream.models import (
    OrderbookLevel,
    OrderbookMsg,
    Trade,
    TradeMsg,
)
from mm_toolbox.rounding import Rounder, RounderConfig
from smm.config import (
    CrossVenueConfig,
    PlainConfig,
    PricingConfig,
    VolatilityConfig,
)
from smm.traders.base.volatility import VolatilityEstimator
from smm.traders.plain.pricing import PlainPricingEngine
from tests.smm.support import make_instrument, make_msg, make_position

TICK = 0.01
LOT = 0.001
MID = 100.0


def _build_skew_pricing(
    use_wmid: bool = False,
    skew_intensity: float = 0.5,
    levels: int = 3,
    base_spread_bps: float = 10.0,
    max_inventory_quote: float = 1000.0,
    cross_instrument: Instrument | None = None,
) -> PlainPricingEngine:
    instrument = make_instrument(TICK, LOT)
    config = PricingConfig(
        levels=levels,
        base_spread_bps=base_spread_bps,
        max_inventory_quote=max_inventory_quote,
        inventory_spread_ladder=[],
        use_wmid=use_wmid,
        skew_intensity=skew_intensity,
        bba_imbalance_weight=0.3,
        orderbook_imbalance_weight=0.3,
        trade_flow_weight=0.2,
        cross_venue_weight=0.2,
        inventory_skew_weight=0.5,
    )
    volatility = VolatilityEstimator(VolatilityConfig())
    rounder = Rounder(RounderConfig.default(tick_size=TICK, lot_size=LOT))
    return PlainPricingEngine(
        instrument=instrument,
        config=config,
        plain_config=PlainConfig(),
        rounder=rounder,
        volatility=volatility,
        cross_instrument=cross_instrument,
    )


def _make_cross_instrument() -> Instrument:
    return Instrument(
        venue=Venue.BINANCE_USDM,
        base=Asset("BTC"),
        quote=Asset("USDT"),
        symbol=Symbol("BTCUSDT"),
        code=0,
        instrument_type=InstrumentType.PERPETUAL,
        tick_size=0.01,
        lot_size=0.001,
    )


def _make_book_msg(
    instrument: Instrument,
    bids: tuple[OrderbookLevel, ...],
    asks: tuple[OrderbookLevel, ...],
) -> OrderbookMsg:
    return make_msg(
        OrderbookMsg,
        instrument,
        bids=bids,
        asks=asks,
        is_bbo=False,
    )


class TestSkewPricingEngine:
    """Tests for skew-adjusted plain pricing."""

    def test_wmid_centering(self) -> None:
        """Engine centers quotes around WMID when enabled and bids are heavier."""
        engine = _build_skew_pricing(use_wmid=True, skew_intensity=0.0)
        # Arithmetic mid is 100.0. Heavy bids (size 9 vs 1) pull WMID to 100.8
        bids = (OrderbookLevel(price=99.0, size=9.0),)
        asks = (OrderbookLevel(price=101.0, size=1.0),)
        engine.consume_msg(_make_book_msg(engine.instrument, bids, asks))

        assert math.isclose(engine.mid_price, 100.0)
        assert math.isclose(engine.wmid_price, (99.0 * 1.0 + 101.0 * 9.0) / 10.0)  # 100.8
        desired = engine.generate_desired_state()
        # Quotes should be placed relative to WMID 100.8, not 100.0
        assert desired.bids[0].price > 100.0
        assert desired.asks[0].price > 100.8

    def test_asymmetric_spread_bullish_skew(self) -> None:
        """Strong buy book & trade flow imbalance narrows bid spread and widens ask spread."""
        engine = _build_skew_pricing(use_wmid=False, skew_intensity=0.5)
        # Heavy bids in book
        bids = (OrderbookLevel(price=99.0, size=10.0),)
        asks = (OrderbookLevel(price=101.0, size=1.0),)
        engine.consume_msg(_make_book_msg(engine.instrument, bids, asks))

        # Heavy aggressive buy trades
        trades_msg = make_msg(
            TradeMsg,
            engine.instrument,
            trades=(
                Trade(time_ms=1000, price=100.5, size=5.0, is_buy=True),
                Trade(time_ms=1001, price=100.6, size=5.0, is_buy=True),
            ),
        )
        engine.consume_msg(trades_msg)

        desired = engine.generate_desired_state()
        assert len(desired.bids) > 0
        assert len(desired.asks) > 0

        # Calculate distance from mid for level 0
        bid_distance = engine.mid_price - desired.bids[0].price
        ask_distance = desired.asks[0].price - engine.mid_price

        # With positive (bullish) skew, bid spread narrows (tighter to mid)
        # and ask spread widens (further from mid)
        assert bid_distance < ask_distance

    def test_asymmetric_spread_bearish_skew(self) -> None:
        """Heavy ask pressure and sell trades widen bid spread and narrow ask spread."""
        engine = _build_skew_pricing(use_wmid=False, skew_intensity=0.5)
        # Heavy asks in book
        bids = (OrderbookLevel(price=99.0, size=1.0),)
        asks = (OrderbookLevel(price=101.0, size=10.0),)
        engine.consume_msg(_make_book_msg(engine.instrument, bids, asks))

        # Heavy aggressive sell trades
        trades_msg = make_msg(
            TradeMsg,
            engine.instrument,
            trades=(
                Trade(time_ms=1000, price=99.5, size=5.0, is_buy=False),
                Trade(time_ms=1001, price=99.4, size=5.0, is_buy=False),
            ),
        )
        engine.consume_msg(trades_msg)

        desired = engine.generate_desired_state()
        bid_distance = engine.mid_price - desired.bids[0].price
        ask_distance = desired.asks[0].price - engine.mid_price

        # With negative (bearish) skew, ask spread narrows (tighter to mid to sell)
        # and bid spread widens (further from mid to protect against downside)
        assert ask_distance < bid_distance

    def test_inventory_skew_offsets_quotes(self) -> None:
        """Long inventory tilts skew negative (widening bids and pulling asks closer)."""
        engine = _build_skew_pricing(use_wmid=False, skew_intensity=0.5)
        # Neutral symmetric book
        bids = (OrderbookLevel(price=99.0, size=5.0),)
        asks = (OrderbookLevel(price=101.0, size=5.0),)
        engine.consume_msg(_make_book_msg(engine.instrument, bids, asks))

        # Significant long position (5 units * 100 mid = 500 quote out of 1000 max -> 50% util)
        pos_msg = make_position(
            engine.instrument,
            size=5.0,
            is_long=True,
            price=100.0,
        )
        engine.consume_msg(pos_msg)

        desired = engine.generate_desired_state()
        bid_distance = engine.mid_price - desired.bids[0].price
        ask_distance = desired.asks[0].price - engine.mid_price

        # Long inventory -> ask is tighter to offload, bid is wider
        assert ask_distance < bid_distance

    def test_cross_venue_lead_lag_skew(self) -> None:
        """Cross venue lead price shifts local quoting skew."""
        cross_inst = _make_cross_instrument()
        engine = _build_skew_pricing(
            use_wmid=False, skew_intensity=0.5, cross_instrument=cross_inst
        )

        # Local book at 100.0
        bids = (OrderbookLevel(price=99.0, size=5.0),)
        asks = (OrderbookLevel(price=101.0, size=5.0),)
        engine.consume_msg(_make_book_msg(engine.instrument, bids, asks))

        # Cross venue book leading higher at 101.0 mid
        cross_bids = (OrderbookLevel(price=100.5, size=5.0),)
        cross_asks = (OrderbookLevel(price=101.5, size=5.0),)
        engine.consume_msg(_make_book_msg(cross_inst, cross_bids, cross_asks))

        assert engine.cross_mid_price == 101.0

        desired = engine.generate_desired_state()
        bid_distance = engine.mid_price - desired.bids[0].price
        ask_distance = desired.asks[0].price - engine.mid_price

        # Cross venue is higher -> positive skew -> tighter bids, wider asks
        assert bid_distance < ask_distance


class TestConfigValidation:
    """Tests for new configuration fields and validation."""

    def test_pricing_config_validation(self) -> None:
        """PricingConfig validates skew_intensity and weight bounds."""
        with pytest.raises(ValueError, match="skew_intensity must be within"):
            PricingConfig(
                levels=5,
                base_spread_bps=10.0,
                max_inventory_quote=100.0,
                inventory_spread_ladder=[],
                skew_intensity=1.5,
            )

        with pytest.raises(ValueError, match="bba_imbalance_weight must be >= 0"):
            PricingConfig(
                levels=5,
                base_spread_bps=10.0,
                max_inventory_quote=100.0,
                inventory_spread_ladder=[],
                bba_imbalance_weight=-0.1,
            )

    def test_cross_venue_config_validation(self) -> None:
        """CrossVenueConfig validates symbol when enabled."""
        with pytest.raises(ValueError, match="symbol must be non-empty when enabled"):
            CrossVenueConfig(enabled=True, symbol="  ")

        # Disabled with empty symbol is fine
        cfg = CrossVenueConfig(enabled=False, symbol="")
        assert not cfg.enabled
