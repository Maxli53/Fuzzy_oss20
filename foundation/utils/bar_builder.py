"""
Bar Construction Utilities

Implements all advanced bar types documented in Data_policy.md:
- VolumeBar: Volume threshold-based bars
- DollarBar: Dollar volume threshold-based bars
- ImbalanceBar: Order flow imbalance-based bars
- TickBar: Tick count-based bars
- RangeBar: Price range-based bars
- RenkoBar: Renko brick construction

Each bar builder follows the same pattern:
1. Accumulate ticks until threshold condition is met
2. Construct bar with OHLCV + metadata
3. Reset accumulator for next bar
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
import numpy as np
from collections import deque

from foundation.models.market import (
    TickData, OHLCVBar, TimeBar, VolumeBar, DollarBar, ImbalanceBar,
    TickBar, RangeBar, RenkoBar
)
from foundation.models.enums import TradeSign, BarType, TimeInterval
from foundation.models.metadata import SymbolDayMetadata


class BarAccumulator:
    """Base accumulator for bar construction"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ticks: List[TickData] = []
        self.first_tick: Optional[TickData] = None
        self.last_tick: Optional[TickData] = None

        # OHLCV tracking
        self.open_price: Optional[Decimal] = None
        self.high_price: Optional[Decimal] = None
        self.low_price: Optional[Decimal] = None
        self.close_price: Optional[Decimal] = None
        self.volume: Decimal = Decimal('0')

        # Metadata tracking
        self.buy_volume: Decimal = Decimal('0')
        self.sell_volume: Decimal = Decimal('0')
        self.neutral_volume: Decimal = Decimal('0')
        self.buy_dollar_volume: Decimal = Decimal('0')
        self.sell_dollar_volume: Decimal = Decimal('0')
        self.spreads: List[Decimal] = []
        self.tick_sizes: List[Decimal] = []

    def add_tick(self, tick: TickData) -> None:
        """Add tick to accumulator and update OHLCV + metadata"""
        self.ticks.append(tick)

        if self.first_tick is None:
            self.first_tick = tick
            self.open_price = tick.price
            self.high_price = tick.price
            self.low_price = tick.price

        self.last_tick = tick
        self.close_price = tick.price

        # Update high/low
        if tick.price > self.high_price:
            self.high_price = tick.price
        if tick.price < self.low_price:
            self.low_price = tick.price

        # Update volume
        self.volume += tick.size

        # Update trade direction volumes
        if tick.trade_sign == TradeSign.BUY:
            self.buy_volume += tick.size
            self.buy_dollar_volume += tick.price * tick.size
        elif tick.trade_sign == TradeSign.SELL:
            self.sell_volume += tick.size
            self.sell_dollar_volume += tick.price * tick.size
        else:
            self.neutral_volume += tick.size

        # Track spreads and tick sizes
        if tick.spread is not None:
            self.spreads.append(tick.spread)
        if len(self.ticks) > 1:
            price_change = abs(tick.price - self.ticks[-2].price)
            if price_change > 0:
                self.tick_sizes.append(price_change)

    def get_dollar_volume(self) -> Decimal:
        """Calculate total dollar volume"""
        return self.buy_dollar_volume + self.sell_dollar_volume

    def get_avg_spread_bps(self) -> Optional[float]:
        """Calculate average spread in basis points"""
        if not self.spreads or self.close_price is None:
            return None
        avg_spread = sum(self.spreads) / len(self.spreads)
        return float(avg_spread / self.close_price * 10000)

    def get_spread_volatility(self) -> Optional[float]:
        """Calculate spread volatility"""
        if len(self.spreads) < 2:
            return None
        spreads_array = np.array([float(s) for s in self.spreads])
        return float(np.std(spreads_array))

    def get_avg_tick_size(self) -> Optional[Decimal]:
        """Calculate average tick size"""
        if not self.tick_sizes:
            return None
        return sum(self.tick_sizes) / len(self.tick_sizes)

    def get_vwap(self) -> Optional[Decimal]:
        """Calculate Volume Weighted Average Price"""
        if not self.ticks:
            return None
        total_pv = sum(tick.price * tick.size for tick in self.ticks)
        return total_pv / self.volume if self.volume > 0 else None

    def get_effective_spread_mean(self) -> Optional[float]:
        """Calculate mean effective spread"""
        if not self.ticks or len(self.ticks) < 2:
            return None
        effective_spreads = []
        for tick in self.ticks:
            if tick.midpoint is not None:
                effective_spread = 2 * abs(tick.price - tick.midpoint)
                effective_spreads.append(float(effective_spread))
        return sum(effective_spreads) / len(effective_spreads) if effective_spreads else None

    def get_liquidity_score(self) -> float:
        """Calculate composite liquidity score (0-100)"""
        score = 50.0  # Base score

        # Volume component (0-25 points)
        if self.volume > 0:
            volume_score = min(25.0, float(self.volume) / 1000.0)
            score += volume_score

        # Spread component (0-25 points)
        avg_spread_bps = self.get_avg_spread_bps()
        if avg_spread_bps is not None:
            spread_score = max(0.0, 25.0 - avg_spread_bps / 10.0)
            score += spread_score

        return min(100.0, max(0.0, score))

    def reset(self) -> None:
        """Reset accumulator for next bar"""
        self.ticks.clear()
        self.first_tick = None
        self.last_tick = None
        self.open_price = None
        self.high_price = None
        self.low_price = None
        self.close_price = None
        self.volume = Decimal('0')
        self.buy_volume = Decimal('0')
        self.sell_volume = Decimal('0')
        self.neutral_volume = Decimal('0')
        self.buy_dollar_volume = Decimal('0')
        self.sell_dollar_volume = Decimal('0')
        self.spreads.clear()
        self.tick_sizes.clear()


class TimeBarBuilder:
    """Time-based bar construction (1-min, 5-min, etc.)"""

    def __init__(self, symbol: str, interval_seconds: int):
        self.symbol = symbol
        self.interval_seconds = interval_seconds
        self.accumulator = BarAccumulator(symbol)
        self.current_bar_start: Optional[datetime] = None
        self.current_bar_end: Optional[datetime] = None

    def add_tick(self, tick: TickData) -> Optional[TimeBar]:
        """Add tick and return bar if time interval complete"""
        # Initialize bar boundaries on first tick
        if self.current_bar_start is None:
            self._initialize_bar_boundaries(tick.timestamp)

        # Check if tick belongs to next bar interval
        if tick.timestamp >= self.current_bar_end:
            # Complete current bar if it has data
            bar = None
            if self.accumulator.ticks:
                bar = self._create_bar(is_complete=True)
                self.accumulator.reset()

            # Move to next interval
            self._advance_to_interval(tick.timestamp)

            # Add tick to new interval
            self.accumulator.add_tick(tick)

            return bar
        else:
            # Add tick to current interval
            self.accumulator.add_tick(tick)
            return None

    def force_close(self) -> Optional[TimeBar]:
        """Force close current bar (e.g., at market close)"""
        if not self.accumulator.ticks:
            return None
        bar = self._create_bar(is_complete=False)
        self.accumulator.reset()
        return bar

    def _initialize_bar_boundaries(self, timestamp: datetime) -> None:
        """Initialize bar start/end times based on first tick"""
        # Round down to nearest interval
        total_seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        interval_start_seconds = (total_seconds // self.interval_seconds) * self.interval_seconds

        hours = interval_start_seconds // 3600
        minutes = (interval_start_seconds % 3600) // 60
        seconds = interval_start_seconds % 60

        self.current_bar_start = timestamp.replace(
            hour=hours,
            minute=minutes,
            second=seconds,
            microsecond=0
        )

        from datetime import timedelta
        self.current_bar_end = self.current_bar_start + timedelta(seconds=self.interval_seconds)

    def _advance_to_interval(self, timestamp: datetime) -> None:
        """Advance to the interval containing the given timestamp"""
        while timestamp >= self.current_bar_end:
            from datetime import timedelta
            self.current_bar_start = self.current_bar_end
            self.current_bar_end = self.current_bar_start + timedelta(seconds=self.interval_seconds)

    def _get_interval_enum(self) -> TimeInterval:
        """Convert interval seconds to TimeInterval enum"""
        interval_map = {
            60: TimeInterval.MINUTE_1,
            300: TimeInterval.MINUTE_5,
            900: TimeInterval.MINUTE_15,
            1800: TimeInterval.MINUTE_30,
            3600: TimeInterval.HOUR_1,
            14400: TimeInterval.HOUR_4,
        }
        return interval_map.get(self.interval_seconds, TimeInterval.MINUTE_1)

    def _create_bar(self, is_complete: bool) -> TimeBar:
        """Create TimeBar from current accumulator state"""
        # Count gaps (no-trade periods) - simplified
        gaps = 0
        if len(self.accumulator.ticks) > 1:
            for i in range(1, len(self.accumulator.ticks)):
                time_diff = (self.accumulator.ticks[i].timestamp -
                           self.accumulator.ticks[i-1].timestamp).total_seconds()
                if time_diff > 10:  # More than 10 seconds between trades
                    gaps += 1

        return TimeBar(
            symbol=self.symbol,
            timestamp=self.accumulator.last_tick.timestamp,
            open=self.accumulator.open_price,
            high=self.accumulator.high_price,
            low=self.accumulator.low_price,
            close=self.accumulator.close_price,
            volume=self.accumulator.volume,
            dollar_volume=self.accumulator.get_dollar_volume(),
            trade_count=len(self.accumulator.ticks),
            buy_volume=self.accumulator.buy_volume,
            sell_volume=self.accumulator.sell_volume,
            interval=self._get_interval_enum(),
            tick_count=len(self.accumulator.ticks),
            interval_seconds=self.interval_seconds,
            bar_start_time=self.current_bar_start,
            bar_end_time=self.current_bar_end,
            is_complete=is_complete,
            gaps=gaps,
            avg_spread_bps=self.accumulator.get_avg_spread_bps(),
            spread_volatility=self.accumulator.get_spread_volatility(),
            buy_dollar_volume=self.accumulator.buy_dollar_volume,
            vwap=self.accumulator.get_vwap(),
            avg_tick_size=self.accumulator.get_avg_tick_size(),
            effective_spread_mean=self.accumulator.get_effective_spread_mean(),
            liquidity_score=self.accumulator.get_liquidity_score()
        )


class VolumeBarBuilder:
    """Volume threshold-based bar construction"""

    def __init__(self, symbol: str, volume_threshold: Decimal):
        self.symbol = symbol
        self.volume_threshold = volume_threshold
        self.accumulator = BarAccumulator(symbol)

    def add_tick(self, tick: TickData) -> Optional[VolumeBar]:
        """Add tick and return bar if volume threshold reached"""
        self.accumulator.add_tick(tick)

        if self.accumulator.volume >= self.volume_threshold:
            bar = self._create_bar()
            self.accumulator.reset()
            return bar
        return None

    def _create_bar(self) -> VolumeBar:
        """Create VolumeBar from current accumulator state"""
        return VolumeBar(
            symbol=self.symbol,
            timestamp=self.accumulator.last_tick.timestamp,
            open=self.accumulator.open_price,
            high=self.accumulator.high_price,
            low=self.accumulator.low_price,
            close=self.accumulator.close_price,
            volume=self.accumulator.volume,
            dollar_volume=self.accumulator.get_dollar_volume(),
            trade_count=len(self.accumulator.ticks),
            buy_volume=self.accumulator.buy_volume,
            sell_volume=self.accumulator.sell_volume,
            volume_threshold=self.volume_threshold,
            interval=TimeInterval.TICK,
            tick_count=len(self.accumulator.ticks),
            avg_spread_bps=self.accumulator.get_avg_spread_bps(),
            spread_volatility=self.accumulator.get_spread_volatility(),
            buy_dollar_volume=self.accumulator.buy_dollar_volume,
            vwap=self.accumulator.get_vwap(),
            avg_tick_size=self.accumulator.get_avg_tick_size(),
            effective_spread_mean=self.accumulator.get_effective_spread_mean(),
            liquidity_score=self.accumulator.get_liquidity_score()
        )


class DollarBarBuilder:
    """Dollar volume threshold-based bar construction"""

    def __init__(self, symbol: str, dollar_threshold: Decimal):
        self.symbol = symbol
        self.dollar_threshold = dollar_threshold
        self.accumulator = BarAccumulator(symbol)

    def add_tick(self, tick: TickData) -> Optional[DollarBar]:
        """Add tick and return bar if dollar volume threshold reached"""
        self.accumulator.add_tick(tick)

        if self.accumulator.get_dollar_volume() >= self.dollar_threshold:
            bar = self._create_bar()
            self.accumulator.reset()
            return bar
        return None

    def _create_bar(self) -> DollarBar:
        """Create DollarBar from current accumulator state"""
        return DollarBar(
            symbol=self.symbol,
            timestamp=self.accumulator.last_tick.timestamp,
            open=self.accumulator.open_price,
            high=self.accumulator.high_price,
            low=self.accumulator.low_price,
            close=self.accumulator.close_price,
            volume=self.accumulator.volume,
            dollar_volume=self.accumulator.get_dollar_volume(),
            trade_count=len(self.accumulator.ticks),
            buy_volume=self.accumulator.buy_volume,
            sell_volume=self.accumulator.sell_volume,
            dollar_threshold=self.dollar_threshold,
            interval=TimeInterval.TICK,
            tick_count=len(self.accumulator.ticks),
            avg_spread_bps=self.accumulator.get_avg_spread_bps(),
            spread_volatility=self.accumulator.get_spread_volatility(),
            buy_dollar_volume=self.accumulator.buy_dollar_volume,
            vwap=self.accumulator.get_vwap(),
            avg_tick_size=self.accumulator.get_avg_tick_size(),
            effective_spread_mean=self.accumulator.get_effective_spread_mean(),
            liquidity_score=self.accumulator.get_liquidity_score()
        )


class ImbalanceBarBuilder:
    """Order flow imbalance-based bar construction"""

    def __init__(self, symbol: str, imbalance_threshold: Decimal):
        self.symbol = symbol
        self.imbalance_threshold = imbalance_threshold
        self.accumulator = BarAccumulator(symbol)

    def add_tick(self, tick: TickData) -> Optional[ImbalanceBar]:
        """Add tick and return bar if imbalance threshold reached"""
        self.accumulator.add_tick(tick)

        imbalance = abs(self.accumulator.buy_volume - self.accumulator.sell_volume)
        if imbalance >= self.imbalance_threshold:
            bar = self._create_bar()
            self.accumulator.reset()
            return bar
        return None

    def _create_bar(self) -> ImbalanceBar:
        """Create ImbalanceBar from current accumulator state"""
        imbalance_ratio = float(
            (self.accumulator.buy_volume - self.accumulator.sell_volume) /
            self.accumulator.volume if self.accumulator.volume > 0 else 0
        )

        return ImbalanceBar(
            symbol=self.symbol,
            timestamp=self.accumulator.last_tick.timestamp,
            open=self.accumulator.open_price,
            high=self.accumulator.high_price,
            low=self.accumulator.low_price,
            close=self.accumulator.close_price,
            volume=self.accumulator.volume,
            dollar_volume=self.accumulator.get_dollar_volume(),
            trade_count=len(self.accumulator.ticks),
            buy_volume=self.accumulator.buy_volume,
            sell_volume=self.accumulator.sell_volume,
            imbalance_threshold=self.imbalance_threshold,
            imbalance_ratio=imbalance_ratio,
            interval=TimeInterval.TICK,
            tick_count=len(self.accumulator.ticks),
            avg_spread_bps=self.accumulator.get_avg_spread_bps(),
            spread_volatility=self.accumulator.get_spread_volatility(),
            buy_dollar_volume=self.accumulator.buy_dollar_volume,
            vwap=self.accumulator.get_vwap(),
            avg_tick_size=self.accumulator.get_avg_tick_size(),
            effective_spread_mean=self.accumulator.get_effective_spread_mean(),
            liquidity_score=self.accumulator.get_liquidity_score()
        )


class TickBarBuilder:
    """Tick count-based bar construction"""

    def __init__(self, symbol: str, tick_threshold: int):
        self.symbol = symbol
        self.tick_threshold = tick_threshold
        self.accumulator = BarAccumulator(symbol)

    def add_tick(self, tick: TickData) -> Optional[TickBar]:
        """Add tick and return bar if tick count threshold reached"""
        self.accumulator.add_tick(tick)

        if len(self.accumulator.ticks) >= self.tick_threshold:
            bar = self._create_bar()
            self.accumulator.reset()
            return bar
        return None

    def _create_bar(self) -> TickBar:
        """Create TickBar from current accumulator state"""
        return TickBar(
            symbol=self.symbol,
            timestamp=self.accumulator.last_tick.timestamp,
            open=self.accumulator.open_price,
            high=self.accumulator.high_price,
            low=self.accumulator.low_price,
            close=self.accumulator.close_price,
            volume=self.accumulator.volume,
            dollar_volume=self.accumulator.get_dollar_volume(),
            trade_count=len(self.accumulator.ticks),
            buy_volume=self.accumulator.buy_volume,
            sell_volume=self.accumulator.sell_volume,
            tick_threshold=self.tick_threshold,
            interval=TimeInterval.TICK,
            tick_count=len(self.accumulator.ticks),
            avg_spread_bps=self.accumulator.get_avg_spread_bps(),
            spread_volatility=self.accumulator.get_spread_volatility(),
            buy_dollar_volume=self.accumulator.buy_dollar_volume,
            vwap=self.accumulator.get_vwap(),
            avg_tick_size=self.accumulator.get_avg_tick_size(),
            effective_spread_mean=self.accumulator.get_effective_spread_mean(),
            liquidity_score=self.accumulator.get_liquidity_score()
        )


class RangeBarBuilder:
    """Price range-based bar construction"""

    def __init__(self, symbol: str, price_range: Decimal):
        self.symbol = symbol
        self.price_range = price_range
        self.accumulator = BarAccumulator(symbol)

    def add_tick(self, tick: TickData) -> Optional[RangeBar]:
        """Add tick and return bar if price range threshold reached"""
        self.accumulator.add_tick(tick)

        price_range = self.accumulator.high_price - self.accumulator.low_price
        if price_range >= self.price_range:
            bar = self._create_bar()
            self.accumulator.reset()
            return bar
        return None

    def _create_bar(self) -> RangeBar:
        """Create RangeBar from current accumulator state"""
        return RangeBar(
            symbol=self.symbol,
            timestamp=self.accumulator.last_tick.timestamp,
            open=self.accumulator.open_price,
            high=self.accumulator.high_price,
            low=self.accumulator.low_price,
            close=self.accumulator.close_price,
            volume=self.accumulator.volume,
            dollar_volume=self.accumulator.get_dollar_volume(),
            trade_count=len(self.accumulator.ticks),
            buy_volume=self.accumulator.buy_volume,
            sell_volume=self.accumulator.sell_volume,
            price_range=self.price_range,
            range_threshold=self.price_range,  # RangeBar expects this field
            interval=TimeInterval.TICK,
            tick_count=len(self.accumulator.ticks),
            avg_spread_bps=self.accumulator.get_avg_spread_bps(),
            spread_volatility=self.accumulator.get_spread_volatility(),
            buy_dollar_volume=self.accumulator.buy_dollar_volume,
            vwap=self.accumulator.get_vwap(),
            avg_tick_size=self.accumulator.get_avg_tick_size(),
            effective_spread_mean=self.accumulator.get_effective_spread_mean(),
            liquidity_score=self.accumulator.get_liquidity_score()
        )


class RenkoBarBuilder:
    """Renko brick construction"""

    def __init__(self, symbol: str, brick_size: Decimal):
        self.symbol = symbol
        self.brick_size = brick_size
        self.current_brick_high: Optional[Decimal] = None
        self.current_brick_low: Optional[Decimal] = None
        self.accumulator = BarAccumulator(symbol)

    def add_tick(self, tick: TickData) -> Optional[RenkoBar]:
        """Add tick and return bar if new brick is formed"""
        self.accumulator.add_tick(tick)

        # Initialize brick levels on first tick
        if self.current_brick_high is None:
            base_price = tick.price.quantize(self.brick_size, rounding=ROUND_HALF_UP)
            self.current_brick_high = base_price + self.brick_size
            self.current_brick_low = base_price

        # Check for new brick formation
        if tick.price >= self.current_brick_high:
            # Upward brick
            bar = self._create_bar(is_bullish=True)
            self._advance_brick_up()
            self.accumulator.reset()
            return bar
        elif tick.price <= self.current_brick_low:
            # Downward brick
            bar = self._create_bar(is_bullish=False)
            self._advance_brick_down()
            self.accumulator.reset()
            return bar

        return None

    def _create_bar(self, is_bullish: bool) -> RenkoBar:
        """Create RenkoBar from current accumulator state"""
        if is_bullish:
            open_price = self.current_brick_low
            close_price = self.current_brick_high
            brick_direction = 1
        else:
            open_price = self.current_brick_high
            close_price = self.current_brick_low
            brick_direction = -1

        return RenkoBar(
            symbol=self.symbol,
            timestamp=self.accumulator.last_tick.timestamp,
            open=open_price,
            high=max(open_price, close_price),
            low=min(open_price, close_price),
            close=close_price,
            volume=self.accumulator.volume,
            dollar_volume=self.accumulator.get_dollar_volume(),
            trade_count=len(self.accumulator.ticks),
            buy_volume=self.accumulator.buy_volume,
            sell_volume=self.accumulator.sell_volume,
            brick_size=self.brick_size,
            brick_direction=brick_direction,
            interval=TimeInterval.MINUTE_1,  # Default 1-minute interval
            tick_count=len(self.accumulator.ticks),
            avg_spread_bps=self.accumulator.get_avg_spread_bps(),
            spread_volatility=self.accumulator.get_spread_volatility(),
            buy_dollar_volume=self.accumulator.buy_dollar_volume,
            vwap=self.accumulator.get_vwap(),
            avg_tick_size=self.accumulator.get_avg_tick_size(),
            effective_spread_mean=self.accumulator.get_effective_spread_mean(),
            liquidity_score=self.accumulator.get_liquidity_score()
        )

    def _advance_brick_up(self) -> None:
        """Advance brick levels upward"""
        self.current_brick_low = self.current_brick_high
        self.current_brick_high += self.brick_size

    def _advance_brick_down(self) -> None:
        """Advance brick levels downward"""
        self.current_brick_high = self.current_brick_low
        self.current_brick_low -= self.brick_size


class BarBuilderFactory:
    """Factory for creating bar builders"""

    @staticmethod
    def create_builder(bar_type: BarType, symbol: str, **kwargs) -> Any:
        """Create appropriate bar builder based on type"""
        builders = {
            BarType.TIME: TimeBarBuilder,
            BarType.VOLUME: VolumeBarBuilder,
            BarType.DOLLAR: DollarBarBuilder,
            BarType.IMBALANCE: ImbalanceBarBuilder,
            BarType.TICK: TickBarBuilder,
            BarType.RANGE: RangeBarBuilder,
            BarType.RENKO: RenkoBarBuilder
        }

        builder_class = builders.get(bar_type)
        if not builder_class:
            raise ValueError(f"Unsupported bar type: {bar_type}")

        return builder_class(symbol, **kwargs)


class MultiBarProcessor:
    """Process multiple bar types simultaneously"""

    def __init__(self, symbol: str, bar_configs: Dict[BarType, Dict[str, Any]]):
        self.symbol = symbol
        self.builders = {}

        for bar_type, config in bar_configs.items():
            self.builders[bar_type] = BarBuilderFactory.create_builder(
                bar_type, symbol, **config
            )

    def process_tick(self, tick: TickData) -> Dict[BarType, Any]:
        """Process tick through all builders and return any completed bars"""
        completed_bars = {}

        for bar_type, builder in self.builders.items():
            bar = builder.add_tick(tick)
            if bar is not None:
                completed_bars[bar_type] = bar

        return completed_bars