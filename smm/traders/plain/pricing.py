"""Plain pricing engine for mid-price following quotes.

Usage: instantiated by PlainTrader to generate desired quotes.
Components: mid-price tracking, volatility spreads, inventory adjustments.
"""

from framework.base.common import ClientOrderId, Instrument
from framework.base.stream.models import (
    DataMsg,
    OrderbookMsg,
    PositionMsg,
    TickerMsg,
    TradeMsg,
)
from mm_toolbox.rounding import Rounder

from smm.config import PlainConfig, PricingConfig
from smm.features import (
    TradeFlowImbalance,
    bba_imbalance,
    log_price_difference,
    orderbook_imbalance,
    weighted_mid_price,
)
from smm.traders.base.pricing import BasePricingEngine
from smm.traders.base.types import DesiredOrder, DesiredState
from smm.traders.base.volatility import VolatilityEstimator
class PlainPricingEngine(BasePricingEngine):
    """Pricing engine that builds symmetric quotes around the mid price.

    Attributes:
        _plain_config (PlainConfig): Plain-specific settings.
        _rounder (Rounder): Price/size rounder.
        _volatility (VolatilityEstimator): Volatility estimator.
    """

    def __init__(
        self,
        instrument: Instrument,
        config: PricingConfig,
        plain_config: PlainConfig,
        rounder: Rounder,
        volatility: VolatilityEstimator,
        cross_instrument: Instrument | None = None,
    ) -> None:
        """Initialize the plain pricing engine.

        Args:
            instrument (Instrument): Instrument being traded.
            config (PricingConfig): Pricing configuration settings.
            plain_config (PlainConfig): Plain-specific configuration.
            rounder (Rounder): Price and size rounder.
            volatility (VolatilityEstimator): Volatility estimator for spreads.
            cross_instrument (Instrument | None): Optional secondary reference instrument.
        """
        super().__init__(instrument=instrument, config=config)
        self._plain_config = plain_config
        self._rounder = rounder
        self._volatility = volatility
        self._cross_instrument = cross_instrument

        self._mid_price: float = 0.0
        self._position_size: float = 0.0
        self._wmid_price: float = 0.0
        self._cross_mid_price: float = 0.0
        self._cross_update_ns: int = 0
        self._last_primary_time_ns: int = 0
        self._has_full_depth: bool = False
        self._bba_imbalance: float = 0.0
        self._depth_imbalance: float = 0.0
        self._trade_flow = TradeFlowImbalance()

    @property
    def mid_price(self) -> float:
        """Return the current arithmetic mid price."""
        return self._mid_price

    @property
    def wmid_price(self) -> float:
        """Return the current volume-weighted mid price."""
        return self._wmid_price

    @property
    def bba_imbalance_value(self) -> float:
        """Return the top-of-book BBA volume imbalance."""
        return self._bba_imbalance

    @property
    def depth_imbalance_value(self) -> float:
        """Return the multi-depth orderbook imbalance."""
        return self._depth_imbalance

    @property
    def trade_flow_imbalance_value(self) -> float:
        """Return the recent trade flow delta imbalance."""
        return self._trade_flow.imbalance()

    @property
    def cross_mid_price(self) -> float:
        """Return the secondary cross-venue mid price."""
        return self._cross_mid_price

    def consume_msg(self, msg: DataMsg) -> None:
        """Consume a pricing-relevant stream message.

        Args:
            msg (DataMsg): Stream message consumed by pricing.
        """
        match msg:
            case TradeMsg():
                if msg.instrument == self.instrument:
                    self._last_primary_time_ns = msg.moments.recv_time_ns
                    self._volatility.update_msg(msg)
                    self._trade_flow.update_trades(msg.trades)
            case OrderbookMsg():
                if msg.instrument == self.instrument:
                    self._last_primary_time_ns = msg.moments.recv_time_ns
                    if not msg.bids or not msg.asks:
                        return
                    best_bid = msg.bids[-1].price
                    best_ask = msg.asks[0].price
                    self._mid_price = (best_bid + best_ask) / 2.0
                    self._bba_imbalance = bba_imbalance(msg.bids, msg.asks)
                    if not msg.is_bbo:
                        self._has_full_depth = True
                        self._depth_imbalance = orderbook_imbalance(msg.bids, msg.asks)
                    elif not self._has_full_depth:
                        self._depth_imbalance = orderbook_imbalance(msg.bids, msg.asks)
                    self._wmid_price = weighted_mid_price(
                        msg.bids, msg.asks, fallback_mid=self._mid_price
                    )
                elif (
                    self._cross_instrument is not None
                    and msg.instrument == self._cross_instrument
                ):
                    if msg.bids and msg.asks:
                        self._cross_mid_price = (
                            msg.bids[-1].price + msg.asks[0].price
                        ) / 2.0
                        self._cross_update_ns = msg.moments.recv_time_ns
            case TickerMsg():
                if msg.instrument == self.instrument:
                    self._last_primary_time_ns = msg.moments.recv_time_ns
                    if self._mid_price <= 0.0 and msg.mark_price > 0.0:
                        self._mid_price = msg.mark_price
                elif (
                    self._cross_instrument is not None
                    and msg.instrument == self._cross_instrument
                ):
                    if self._cross_mid_price <= 0.0 and msg.mark_price > 0.0:
                        self._cross_mid_price = msg.mark_price
                        self._cross_update_ns = msg.moments.recv_time_ns
            case PositionMsg():
                if msg.instrument == self.instrument:
                    self._position_size = msg.size if msg.is_long else -msg.size
    def generate_desired_state(self) -> DesiredState:
        """Generate the desired state for plain quoting.

        Returns:
            DesiredState: Desired state with bid and ask orders.
        """
        self._volatility.ensure_fresh()
        desired = DesiredState.empty(self.instrument)

        reference_mid = (
            self._wmid_price
            if (self.config.use_wmid and self._wmid_price > 0.0)
            else self._mid_price
        )
        if reference_mid <= 0.0:
            return desired

        levels = self.config.levels
        quote_per_level = self.config.max_inventory_quote / (2.0 * levels)
        base_spread_bps = max(
            self.config.base_spread_bps, self._volatility.spread_bps()
        )
        spread_multiplier = self._inventory_spread_multiplier()
        spread_bps = base_spread_bps * spread_multiplier

        # Calculate combined market feature and inventory skew
        total_skew = self._calculate_skew()

        # Apply asymmetric spread skewing when enabled
        if self.config.skew_intensity > 0.0:
            bid_spread_bps = max(
                0.1, spread_bps * (1.0 - self.config.skew_intensity * total_skew)
            )
            ask_spread_bps = max(
                0.1, spread_bps * (1.0 + self.config.skew_intensity * total_skew)
            )
        else:
            bid_spread_bps = spread_bps
            ask_spread_bps = spread_bps

        inventory_util = self._inventory_utilization()
        reduce_buy = self._position_size > 0.0
        reduce_sell = self._position_size < 0.0

        for level in range(levels):
            level_mult = level + 1
            bid_level_spread = level_mult * (bid_spread_bps / 10_000.0)
            ask_level_spread = level_mult * (ask_spread_bps / 10_000.0)

            bid_price = self._rounder.bid(reference_mid * (1.0 - bid_level_spread))
            ask_price = self._rounder.ask(reference_mid * (1.0 + ask_level_spread))

            size_quote = quote_per_level
            size_base = self._rounder.size(size_quote / reference_mid)
            bid_size = size_base
            ask_size = size_base
            if level == 0 and inventory_util > 0.0:
                scale = max(0.0, 1.0 - inventory_util)
                if reduce_buy:
                    bid_size = self._rounder.size(size_base * scale)
                elif reduce_sell:
                    ask_size = self._rounder.size(size_base * scale)

            if bid_size > 0.0:
                desired.bids.append(
                    DesiredOrder(
                        price=bid_price,
                        is_buy=True,
                        size=bid_size,
                        is_maker=True,
                        reduce_only=False,
                        client_order_id=ClientOrderId(
                            f"{self._plain_config.cloid_prefix}{level:02d}B"
                        ),
                    )
                )
            if ask_size > 0.0:
                desired.asks.append(
                    DesiredOrder(
                        price=ask_price,
                        is_buy=False,
                        size=ask_size,
                        is_maker=True,
                        reduce_only=False,
                        client_order_id=ClientOrderId(
                            f"{self._plain_config.cloid_prefix}{level:02d}S"
                        ),
                    )
                )

        return desired

    def _calculate_skew(self) -> float:
        """Calculate aggregate skew from micro-features and inventory."""
        trade_imb = self._trade_flow.imbalance()
        cross_spread = 0.0
        is_cross_fresh = False
        if self._cross_mid_price > 0.0 and self._cross_update_ns > 0:
            ref_time_ns = self._last_primary_time_ns
            if ref_time_ns <= 0:
                from mm_toolbox.time import time_ns

                ref_time_ns = time_ns()
            age_s = max(0.0, (ref_time_ns - self._cross_update_ns) / 1e9)
            if age_s <= self.config.cross_venue_max_age_s:
                is_cross_fresh = True

        if is_cross_fresh and self._mid_price > 0.0:
            diff = log_price_difference(self._cross_mid_price, self._mid_price)
            # Normalize 50 bps (0.50%) deviation to approximately [-1.0, 1.0]
            cross_spread = max(-1.0, min(1.0, diff / 0.5))

        active_cross_weight = (
            self.config.cross_venue_weight if is_cross_fresh else 0.0
        )
        total_signal_weight = (
            self.config.bba_imbalance_weight
            + self.config.orderbook_imbalance_weight
            + self.config.trade_flow_weight
            + active_cross_weight
        )
        if total_signal_weight > 0.0:
            signal_skew = (
                self._bba_imbalance * self.config.bba_imbalance_weight
                + self._depth_imbalance * self.config.orderbook_imbalance_weight
                + trade_imb * self.config.trade_flow_weight
                + cross_spread * active_cross_weight
            ) / total_signal_weight
        else:
            signal_skew = 0.0

        inv_util = self._inventory_utilization()
        inv_direction = (
            -1.0 if self._position_size > 0.0 else (1.0 if self._position_size < 0.0 else 0.0)
        )
        inv_skew = inv_direction * inv_util

        inv_w = self.config.inventory_skew_weight
        total_skew = signal_skew * (1.0 - inv_w) + inv_skew * inv_w
        return max(-1.0, min(1.0, total_skew))

    def _inventory_utilization(self) -> float:
        """Calculate inventory utilization as a fraction of max.

        Returns:
            float: Utilization in [0, 1].
        """
        if self._mid_price <= 0.0:
            return 0.0
        inventory_quote = abs(self._position_size) * self._mid_price
        return min(1.0, inventory_quote / self.config.max_inventory_quote)

    def _inventory_spread_multiplier(self) -> float:
        """Return the spread multiplier based on inventory ladder.

        Returns:
            float: Spread multiplier for current inventory utilization.
        """
        util = self._inventory_utilization()
        multiplier = 1.0
        for entry_multiplier, threshold in self.config.inventory_spread_ladder:
            if util >= threshold:
                multiplier = entry_multiplier
        return multiplier
