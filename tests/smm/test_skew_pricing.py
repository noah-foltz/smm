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

    def test_ticker_msg_mark_price_fallback(self) -> None:
        """TickerMsg with mark_price provides fallback mid without AttributeError."""
        from framework.base.stream.models import TickerMsg
        engine = _build_skew_pricing()
        ticker_msg = make_msg(
            TickerMsg,
            engine.instrument,
            mark_price=105.5,
            index_price=105.4,
            funding_rate=0.0001,
            funding_period_min=480,
            next_funding_time_ms=1700000000000,
            open_interest=1000.0,
            avg_volume_24h=50000.0,
            price_chg_24h_pct=2.5,
        )
        engine.consume_msg(ticker_msg)
        assert engine.mid_price == 105.5

    def test_is_bbo_preserves_depth_imbalance(self) -> None:
        """BBO messages with 1 level per side do not overwrite multi-depth imbalance."""
        engine = _build_skew_pricing()
        # Send full book with heavy depth bids
        bids = (
            OrderbookLevel(price=98.0, size=20.0),
            OrderbookLevel(price=99.0, size=15.0),
            OrderbookLevel(price=100.0, size=10.0),
        )
        asks = (
            OrderbookLevel(price=101.0, size=1.0),
            OrderbookLevel(price=102.0, size=1.0),
            OrderbookLevel(price=103.0, size=1.0),
        )
        full_book = make_msg(
            OrderbookMsg,
            engine.instrument,
            bids=bids,
            asks=asks,
            is_bbo=False,
        )
        engine.consume_msg(full_book)
        depth_before = engine.depth_imbalance_value
        assert depth_before > 0.5

        # Send subsequent 1-level BBO message
        bbo_msg = make_msg(
            OrderbookMsg,
            engine.instrument,
            bids=(OrderbookLevel(price=100.0, size=2.0),),
            asks=(OrderbookLevel(price=101.0, size=2.0),),
            is_bbo=True,
        )
        engine.consume_msg(bbo_msg)

        # BBA updated to 0.0, but depth imbalance was preserved
        assert engine.bba_imbalance_value == 0.0
        assert engine.depth_imbalance_value == depth_before

    def test_symmetric_full_book_not_overwritten_by_bbo(self) -> None:
        """Finding 1: A balanced full book (depth imbalance = 0.0) is not overwritten by a subsequent BBO tick."""
        engine = _build_skew_pricing()
        # Symmetric full book
        bids = (
            OrderbookLevel(price=98.0, size=5.0),
            OrderbookLevel(price=99.0, size=5.0),
            OrderbookLevel(price=100.0, size=5.0),
        )
        asks = (
            OrderbookLevel(price=101.0, size=5.0),
            OrderbookLevel(price=102.0, size=5.0),
            OrderbookLevel(price=103.0, size=5.0),
        )
        full_book = make_msg(
            OrderbookMsg,
            engine.instrument,
            bids=bids,
            asks=asks,
            is_bbo=False,
        )
        engine.consume_msg(full_book)
        assert engine.depth_imbalance_value == 0.0
        assert engine._has_full_depth is True

        # Next comes a heavily skewed 1-level BBO message (bids 100 vs asks 1)
        bbo_msg = make_msg(
            OrderbookMsg,
            engine.instrument,
            bids=(OrderbookLevel(price=100.0, size=100.0),),
            asks=(OrderbookLevel(price=101.0, size=1.0),),
            is_bbo=True,
        )
        engine.consume_msg(bbo_msg)
        assert engine.bba_imbalance_value > 0.9
        # depth imbalance remains 0.0 because full book was already observed!
        assert engine.depth_imbalance_value == 0.0

    def test_initial_bbo_not_latched(self) -> None:
        """Finding 1: Consecutive BBO messages before any full book arrives keep updating depth imbalance."""
        engine = _build_skew_pricing()
        # First BBO message
        bbo_1 = make_msg(
            OrderbookMsg,
            engine.instrument,
            bids=(OrderbookLevel(price=100.0, size=10.0),),
            asks=(OrderbookLevel(price=101.0, size=2.0),),
            is_bbo=True,
        )
        engine.consume_msg(bbo_1)
        assert engine.depth_imbalance_value > 0.0

        # Second BBO message with reversed pressure before any full book arrived
        bbo_2 = make_msg(
            OrderbookMsg,
            engine.instrument,
            bids=(OrderbookLevel(price=100.0, size=2.0),),
            asks=(OrderbookLevel(price=101.0, size=10.0),),
            is_bbo=True,
        )
        engine.consume_msg(bbo_2)
        # Should NOT be latched/frozen at the first BBO's positive value
        assert engine.depth_imbalance_value < 0.0

    def test_cross_venue_staleness_drop(self) -> None:
        """Finding 2: Stale cross-venue mid price is ignored in skew calculation."""
        cross_inst = _make_cross_instrument()
        engine = _build_skew_pricing(
            use_wmid=False, skew_intensity=0.5, cross_instrument=cross_inst
        )
        # Local book at 100.0, recv_ns = 1_000_000_000
        bids = (OrderbookLevel(price=99.0, size=5.0),)
        asks = (OrderbookLevel(price=101.0, size=5.0),)
        engine.consume_msg(
            make_msg(OrderbookMsg, engine.instrument, bids=bids, asks=asks, recv_ns=1_000_000_000, is_bbo=False)
        )

        # Cross venue book at 101.0 mid, recv_ns = 1_000_000_000
        cross_bids = (OrderbookLevel(price=100.5, size=5.0),)
        cross_asks = (OrderbookLevel(price=101.5, size=5.0),)
        engine.consume_msg(
            make_msg(OrderbookMsg, cross_inst, bids=cross_bids, asks=cross_asks, recv_ns=1_000_000_000, is_bbo=False)
        )
        # When fresh, cross venue produces positive skew
        fresh_state = engine.generate_desired_state()
        assert (engine.mid_price - fresh_state.bids[0].price) < (fresh_state.asks[0].price - engine.mid_price)

        # Now simulate 10 seconds passing on the primary venue without any cross-venue update
        engine.consume_msg(
            make_msg(OrderbookMsg, engine.instrument, bids=bids, asks=asks, recv_ns=11_000_000_000, is_bbo=False)
        )
        stale_state = engine.generate_desired_state()
        # Since cross quote is now 10 seconds old (> 5.0s max age), cross spread drops to 0.0
        # With symmetric local book and 0 trade flow, spreads should be symmetric!
        stale_bid_dist = engine.mid_price - stale_state.bids[0].price
        stale_ask_dist = stale_state.asks[0].price - engine.mid_price
        assert math.isclose(stale_bid_dist, stale_ask_dist, abs_tol=1e-5)

    def test_zero_size_wmid_fallback(self) -> None:
        """Zero size BBO levels safely fall back to arithmetic mid."""
        from smm.features import weighted_mid_price
        bids = (OrderbookLevel(price=100.0, size=0.0),)
        asks = (OrderbookLevel(price=102.0, size=5.0),)
        assert weighted_mid_price(bids, asks) == 101.0

    def test_volume_adjusted_mid_price_depth_zero(self) -> None:
        """VAMP with depth <= 0 safely returns fallback mid."""
        from smm.features import volume_adjusted_mid_price
        bids = (OrderbookLevel(price=100.0, size=5.0),)
        asks = (OrderbookLevel(price=102.0, size=5.0),)
        assert volume_adjusted_mid_price(bids, asks, depth=0, fallback_mid=101.0) == 101.0
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

    def test_cross_venue_matching_primary_raises(self) -> None:
        """Finding 3: AppConfig rejects cross_venue matching primary venue and symbol."""
        from smm.config import AppConfig, CoreConfig, TraderId
        with pytest.raises(ValueError, match="cannot match primary"):
            AppConfig(
                core=CoreConfig(venue=Venue.BYBIT, symbol="SOLUSDT", trader=TraderId.PLAIN),
                pricing=PricingConfig(
                    levels=5,
                    base_spread_bps=10.0,
                    max_inventory_quote=100.0,
                    inventory_spread_ladder=[],
                ),
                cross_venue=CrossVenueConfig(enabled=True, venue=Venue.BYBIT, symbol="SOLUSDT"),
            )

    def test_cross_venue_default_symbol_raises_when_enabled(self) -> None:
        """CrossVenueConfig has no default symbol so enabling without symbol raises."""
        with pytest.raises(ValueError, match="symbol must be non-empty when enabled"):
            CrossVenueConfig(enabled=True)

    def test_mismatched_cross_asset_pair_raises(self) -> None:
        """PlainPricingEngine rejects cross_instrument with different base/quote asset."""
        sol_cross = Instrument(
            venue=Venue.BINANCE_USDM,
            base=Asset("SOL"),
            quote=Asset("USDT"),
            symbol=Symbol("SOLUSDT"),
            code=0,
            instrument_type=InstrumentType.PERPETUAL,
            tick_size=0.01,
            lot_size=0.001,
        )
        with pytest.raises(ValueError, match="must match primary instrument"):
            _build_skew_pricing(cross_instrument=sol_cross)

    def test_stinky_trader_does_not_support_cross_venue(self) -> None:
        """PlainTrader opts into cross_venue while StinkyTrader opts out."""
        from smm.traders.plain.trader import PlainTrader
        from smm.traders.stinky.trader import StinkyTrader

        assert PlainTrader.supports_cross_venue is True
        assert StinkyTrader.supports_cross_venue is False
