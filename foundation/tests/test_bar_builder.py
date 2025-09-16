"""
Comprehensive tests for bar construction utilities

Tests all bar builders in foundation/utils/bar_builder.py:
- BarAccumulator: Base accumulator functionality
- VolumeBarBuilder: Volume threshold bars
- DollarBarBuilder: Dollar volume threshold bars
- ImbalanceBarBuilder: Order flow imbalance bars
- TickBarBuilder: Tick count bars
- RangeBarBuilder: Price range bars
- RenkoBarBuilder: Renko brick construction
- BarBuilderFactory: Factory pattern
- MultiBarProcessor: Multiple bar types simultaneously
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import List

from foundation.models.market import (
    TickData, VolumeBar, DollarBar, ImbalanceBar,
    TickBar, RangeBar, RenkoBar
)
from foundation.models.enums import TradeSign, MarketCenter, TradeCondition, BarType
from foundation.utils.bar_builder import (
    BarAccumulator, VolumeBarBuilder, DollarBarBuilder, ImbalanceBarBuilder,
    TickBarBuilder, RangeBarBuilder, RenkoBarBuilder, BarBuilderFactory,
    MultiBarProcessor
)


class TestBarAccumulator:
    """Test BarAccumulator base functionality"""

    def create_test_tick(self, price: float, size: int, trade_sign: TradeSign = TradeSign.BUY) -> TickData:
        """Create test tick data"""
        return TickData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            price=Decimal(str(price)),
            size=Decimal(str(size)),
            bid=Decimal(str(price - 0.01)),
            ask=Decimal(str(price + 0.01)),
            bid_size=Decimal("500"),
            ask_size=Decimal("300"),
            trade_sign=trade_sign,
            market_center=MarketCenter.NASDAQ,
            conditions=[TradeCondition.REGULAR],
            sequence_number=12345
        )

    def test_accumulator_initialization(self):
        """Test BarAccumulator initialization"""
        accumulator = BarAccumulator("AAPL")

        assert accumulator.symbol == "AAPL"
        assert accumulator.ticks == []
        assert accumulator.first_tick is None
        assert accumulator.last_tick is None
        assert accumulator.volume == Decimal('0')
        assert accumulator.buy_volume == Decimal('0')
        assert accumulator.sell_volume == Decimal('0')

    def test_accumulator_add_first_tick(self):
        """Test adding first tick to accumulator"""
        accumulator = BarAccumulator("AAPL")
        tick = self.create_test_tick(150.25, 100)

        accumulator.add_tick(tick)

        assert accumulator.first_tick == tick
        assert accumulator.last_tick == tick
        assert accumulator.open_price == Decimal("150.25")
        assert accumulator.high_price == Decimal("150.25")
        assert accumulator.low_price == Decimal("150.25")
        assert accumulator.close_price == Decimal("150.25")
        assert accumulator.volume == Decimal("100")
        assert accumulator.buy_volume == Decimal("100")

    def test_accumulator_add_multiple_ticks(self):
        """Test adding multiple ticks to accumulator"""
        accumulator = BarAccumulator("AAPL")

        # Add buy tick
        tick1 = self.create_test_tick(150.25, 100, TradeSign.BUY)
        accumulator.add_tick(tick1)

        # Add sell tick with higher price
        tick2 = self.create_test_tick(150.50, 200, TradeSign.SELL)
        accumulator.add_tick(tick2)

        # Add neutral tick with lower price
        tick3 = self.create_test_tick(150.10, 50, TradeSign.NEUTRAL)
        accumulator.add_tick(tick3)

        assert accumulator.open_price == Decimal("150.25")
        assert accumulator.high_price == Decimal("150.50")
        assert accumulator.low_price == Decimal("150.10")
        assert accumulator.close_price == Decimal("150.10")
        assert accumulator.volume == Decimal("350")
        assert accumulator.buy_volume == Decimal("100")
        assert accumulator.sell_volume == Decimal("200")
        assert accumulator.neutral_volume == Decimal("50")

    def test_accumulator_dollar_volume_calculation(self):
        """Test dollar volume calculation"""
        accumulator = BarAccumulator("AAPL")

        tick1 = self.create_test_tick(150.00, 100, TradeSign.BUY)
        tick2 = self.create_test_tick(150.50, 200, TradeSign.SELL)
        accumulator.add_tick(tick1)
        accumulator.add_tick(tick2)

        expected_buy_dollar = Decimal("150.00") * Decimal("100")  # 15000
        expected_sell_dollar = Decimal("150.50") * Decimal("200")  # 30100
        expected_total_dollar = expected_buy_dollar + expected_sell_dollar  # 45100

        assert accumulator.buy_dollar_volume == expected_buy_dollar
        assert accumulator.sell_dollar_volume == expected_sell_dollar
        assert accumulator.get_dollar_volume() == expected_total_dollar

    def test_accumulator_vwap_calculation(self):
        """Test VWAP calculation"""
        accumulator = BarAccumulator("AAPL")

        tick1 = self.create_test_tick(150.00, 100)
        tick2 = self.create_test_tick(150.50, 200)
        accumulator.add_tick(tick1)
        accumulator.add_tick(tick2)

        # VWAP = (150.00*100 + 150.50*200) / (100+200) = 45100 / 300 = 150.333...
        expected_vwap = Decimal("45100") / Decimal("300")
        assert accumulator.get_vwap() == expected_vwap

    def test_accumulator_liquidity_score(self):
        """Test liquidity score calculation"""
        accumulator = BarAccumulator("AAPL")

        tick = self.create_test_tick(150.00, 1000)  # Large volume
        accumulator.add_tick(tick)

        score = accumulator.get_liquidity_score()
        assert 0.0 <= score <= 100.0
        assert score > 50.0  # Should be above base score due to volume

    def test_accumulator_reset(self):
        """Test accumulator reset functionality"""
        accumulator = BarAccumulator("AAPL")

        tick = self.create_test_tick(150.25, 100)
        accumulator.add_tick(tick)

        # Verify data exists
        assert len(accumulator.ticks) == 1
        assert accumulator.volume > 0

        # Reset
        accumulator.reset()

        # Verify reset
        assert accumulator.ticks == []
        assert accumulator.first_tick is None
        assert accumulator.last_tick is None
        assert accumulator.volume == Decimal('0')
        assert accumulator.buy_volume == Decimal('0')


class TestVolumeBarBuilder:
    """Test VolumeBarBuilder"""

    def create_test_tick(self, price: float, size: int, trade_sign: TradeSign = TradeSign.BUY) -> TickData:
        """Create test tick data"""
        return TickData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            price=Decimal(str(price)),
            size=Decimal(str(size)),
            bid=Decimal(str(price - 0.01)),
            ask=Decimal(str(price + 0.01)),
            bid_size=Decimal("500"),
            ask_size=Decimal("300"),
            trade_sign=trade_sign,
            market_center=MarketCenter.NASDAQ,
            conditions=[TradeCondition.REGULAR],
            sequence_number=12345
        )

    def test_volume_bar_builder_initialization(self):
        """Test VolumeBarBuilder initialization"""
        builder = VolumeBarBuilder("AAPL", Decimal("1000"))

        assert builder.symbol == "AAPL"
        assert builder.volume_threshold == Decimal("1000")
        assert isinstance(builder.accumulator, BarAccumulator)

    def test_volume_bar_creation(self):
        """Test volume bar creation when threshold reached"""
        builder = VolumeBarBuilder("AAPL", Decimal("300"))

        # Add ticks that don't reach threshold
        tick1 = self.create_test_tick(150.00, 100)
        bar = builder.add_tick(tick1)
        assert bar is None  # Threshold not reached

        tick2 = self.create_test_tick(150.25, 150)
        bar = builder.add_tick(tick2)
        assert bar is None  # Still not reached (250 total)

        # Add tick that reaches threshold
        tick3 = self.create_test_tick(150.50, 100)
        bar = builder.add_tick(tick3)

        assert bar is not None
        assert isinstance(bar, VolumeBar)
        assert bar.symbol == "AAPL"
        assert bar.volume >= builder.volume_threshold
        assert bar.volume_threshold == Decimal("300")
        assert bar.open == Decimal("150.00")
        assert bar.close == Decimal("150.50")

    def test_volume_bar_accumulator_reset(self):
        """Test accumulator resets after bar creation"""
        builder = VolumeBarBuilder("AAPL", Decimal("200"))

        # Fill accumulator
        tick1 = self.create_test_tick(150.00, 100)
        tick2 = self.create_test_tick(150.25, 150)
        builder.add_tick(tick1)
        bar = builder.add_tick(tick2)

        assert bar is not None
        # Accumulator should be reset
        assert len(builder.accumulator.ticks) == 0
        assert builder.accumulator.volume == Decimal('0')


class TestDollarBarBuilder:
    """Test DollarBarBuilder"""

    def create_test_tick(self, price: float, size: int, trade_sign: TradeSign = TradeSign.BUY) -> TickData:
        """Create test tick data"""
        return TickData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            price=Decimal(str(price)),
            size=Decimal(str(size)),
            bid=Decimal(str(price - 0.01)),
            ask=Decimal(str(price + 0.01)),
            bid_size=Decimal("500"),
            ask_size=Decimal("300"),
            trade_sign=trade_sign,
            market_center=MarketCenter.NASDAQ,
            conditions=[TradeCondition.REGULAR],
            sequence_number=12345
        )

    def test_dollar_bar_creation(self):
        """Test dollar bar creation when threshold reached"""
        builder = DollarBarBuilder("AAPL", Decimal("30000"))

        # Add ticks that don't reach threshold
        tick1 = self.create_test_tick(150.00, 100)  # $15,000
        bar = builder.add_tick(tick1)
        assert bar is None

        tick2 = self.create_test_tick(150.25, 50)   # $7,512.50
        bar = builder.add_tick(tick2)
        assert bar is None  # Total: $22,512.50

        # Add tick that reaches threshold
        tick3 = self.create_test_tick(150.50, 100)  # $15,050
        bar = builder.add_tick(tick3)  # Total: $37,562.50

        assert bar is not None
        assert isinstance(bar, DollarBar)
        assert bar.dollar_volume >= builder.dollar_threshold
        assert bar.dollar_threshold == Decimal("30000")


class TestImbalanceBarBuilder:
    """Test ImbalanceBarBuilder"""

    def create_test_tick(self, price: float, size: int, trade_sign: TradeSign) -> TickData:
        """Create test tick data"""
        return TickData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            price=Decimal(str(price)),
            size=Decimal(str(size)),
            bid=Decimal(str(price - 0.01)),
            ask=Decimal(str(price + 0.01)),
            bid_size=Decimal("500"),
            ask_size=Decimal("300"),
            trade_sign=trade_sign,
            market_center=MarketCenter.NASDAQ,
            conditions=[TradeCondition.REGULAR],
            sequence_number=12345
        )

    def test_imbalance_bar_creation(self):
        """Test imbalance bar creation when threshold reached"""
        builder = ImbalanceBarBuilder("AAPL", Decimal("150"))

        # Add balanced trades (no imbalance)
        tick1 = self.create_test_tick(150.00, 100, TradeSign.BUY)   # Buy: 100
        tick2 = self.create_test_tick(150.25, 100, TradeSign.SELL)  # Sell: 100, Imbalance: 0
        builder.add_tick(tick1)
        bar = builder.add_tick(tick2)
        assert bar is None

        # Add more buy volume to create imbalance
        tick3 = self.create_test_tick(150.50, 200, TradeSign.BUY)   # Buy: 300, Sell: 100, Imbalance: 200
        bar = builder.add_tick(tick3)

        assert bar is not None
        assert isinstance(bar, ImbalanceBar)
        assert abs(bar.buy_volume - bar.sell_volume) >= builder.imbalance_threshold
        assert bar.imbalance_threshold == Decimal("150")
        assert bar.imbalance_ratio > 0  # More buys than sells


class TestTickBarBuilder:
    """Test TickBarBuilder"""

    def create_test_tick(self, price: float, size: int = 100) -> TickData:
        """Create test tick data"""
        return TickData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            price=Decimal(str(price)),
            size=Decimal(str(size)),
            bid=Decimal(str(price - 0.01)),
            ask=Decimal(str(price + 0.01)),
            bid_size=Decimal("500"),
            ask_size=Decimal("300"),
            trade_sign=TradeSign.BUY,
            market_center=MarketCenter.NASDAQ,
            conditions=[TradeCondition.REGULAR],
            sequence_number=12345
        )

    def test_tick_bar_creation(self):
        """Test tick bar creation when threshold reached"""
        builder = TickBarBuilder("AAPL", 3)

        # Add ticks one by one
        tick1 = self.create_test_tick(150.00)
        bar = builder.add_tick(tick1)
        assert bar is None  # 1 tick

        tick2 = self.create_test_tick(150.25)
        bar = builder.add_tick(tick2)
        assert bar is None  # 2 ticks

        tick3 = self.create_test_tick(150.50)
        bar = builder.add_tick(tick3)  # 3 ticks - threshold reached

        assert bar is not None
        assert isinstance(bar, TickBar)
        assert bar.trade_count >= builder.tick_threshold
        assert bar.tick_threshold == 3


class TestRangeBarBuilder:
    """Test RangeBarBuilder"""

    def create_test_tick(self, price: float, size: int = 100) -> TickData:
        """Create test tick data"""
        return TickData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            price=Decimal(str(price)),
            size=Decimal(str(size)),
            bid=Decimal(str(price - 0.01)),
            ask=Decimal(str(price + 0.01)),
            bid_size=Decimal("500"),
            ask_size=Decimal("300"),
            trade_sign=TradeSign.BUY,
            market_center=MarketCenter.NASDAQ,
            conditions=[TradeCondition.REGULAR],
            sequence_number=12345
        )

    def test_range_bar_creation(self):
        """Test range bar creation when threshold reached"""
        builder = RangeBarBuilder("AAPL", Decimal("0.50"))

        # Add ticks with small price range
        tick1 = self.create_test_tick(150.00)
        bar = builder.add_tick(tick1)
        assert bar is None  # Range: 0

        tick2 = self.create_test_tick(150.25)
        bar = builder.add_tick(tick2)
        assert bar is None  # Range: 0.25

        # Add tick that creates sufficient range
        tick3 = self.create_test_tick(150.60)
        bar = builder.add_tick(tick3)  # Range: 0.60 (150.60 - 150.00)

        assert bar is not None
        assert isinstance(bar, RangeBar)
        assert (bar.high - bar.low) >= builder.price_range
        assert bar.price_range == Decimal("0.50")


class TestRenkoBarBuilder:
    """Test RenkoBarBuilder"""

    def create_test_tick(self, price: float, size: int = 100) -> TickData:
        """Create test tick data"""
        return TickData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            price=Decimal(str(price)),
            size=Decimal(str(size)),
            bid=Decimal(str(price - 0.01)),
            ask=Decimal(str(price + 0.01)),
            bid_size=Decimal("500"),
            ask_size=Decimal("300"),
            trade_sign=TradeSign.BUY,
            market_center=MarketCenter.NASDAQ,
            conditions=[TradeCondition.REGULAR],
            sequence_number=12345
        )

    def test_renko_bar_upward_brick(self):
        """Test upward Renko brick formation"""
        builder = RenkoBarBuilder("AAPL", Decimal("0.50"))

        # First tick establishes base
        tick1 = self.create_test_tick(150.00)
        bar = builder.add_tick(tick1)
        assert bar is None

        # Small moves don't create bricks
        tick2 = self.create_test_tick(150.25)
        bar = builder.add_tick(tick2)
        assert bar is None

        # Move that creates upward brick
        tick3 = self.create_test_tick(150.75)  # Crosses upper brick threshold
        bar = builder.add_tick(tick3)

        assert bar is not None
        assert isinstance(bar, RenkoBar)
        assert bar.is_bullish is True
        assert bar.brick_size == Decimal("0.50")
        assert (bar.close - bar.open) == bar.brick_size

    def test_renko_bar_downward_brick(self):
        """Test downward Renko brick formation"""
        builder = RenkoBarBuilder("AAPL", Decimal("0.50"))

        # Establish base at higher price
        tick1 = self.create_test_tick(150.50)
        builder.add_tick(tick1)

        # Move down to create downward brick
        tick2 = self.create_test_tick(149.75)  # Crosses lower brick threshold
        bar = builder.add_tick(tick2)

        assert bar is not None
        assert isinstance(bar, RenkoBar)
        assert bar.is_bullish is False
        assert bar.brick_size == Decimal("0.50")


class TestBarBuilderFactory:
    """Test BarBuilderFactory"""

    def test_factory_volume_builder(self):
        """Test factory creates VolumeBarBuilder"""
        builder = BarBuilderFactory.create_builder(
            BarType.VOLUME, "AAPL", volume_threshold=Decimal("1000")
        )
        assert isinstance(builder, VolumeBarBuilder)
        assert builder.volume_threshold == Decimal("1000")

    def test_factory_dollar_builder(self):
        """Test factory creates DollarBarBuilder"""
        builder = BarBuilderFactory.create_builder(
            BarType.DOLLAR, "AAPL", dollar_threshold=Decimal("50000")
        )
        assert isinstance(builder, DollarBarBuilder)
        assert builder.dollar_threshold == Decimal("50000")

    def test_factory_tick_builder(self):
        """Test factory creates TickBarBuilder"""
        builder = BarBuilderFactory.create_builder(
            BarType.TICK, "AAPL", tick_threshold=100
        )
        assert isinstance(builder, TickBarBuilder)
        assert builder.tick_threshold == 100

    def test_factory_unsupported_type(self):
        """Test factory raises error for unsupported bar type"""
        with pytest.raises(ValueError, match="Unsupported bar type"):
            BarBuilderFactory.create_builder(BarType.TIME, "AAPL")


class TestMultiBarProcessor:
    """Test MultiBarProcessor"""

    def create_test_tick(self, price: float, size: int, trade_sign: TradeSign = TradeSign.BUY) -> TickData:
        """Create test tick data"""
        return TickData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            price=Decimal(str(price)),
            size=Decimal(str(size)),
            bid=Decimal(str(price - 0.01)),
            ask=Decimal(str(price + 0.01)),
            bid_size=Decimal("500"),
            ask_size=Decimal("300"),
            trade_sign=trade_sign,
            market_center=MarketCenter.NASDAQ,
            conditions=[TradeCondition.REGULAR],
            sequence_number=12345
        )

    def test_multi_bar_processor_initialization(self):
        """Test MultiBarProcessor initialization"""
        configs = {
            BarType.VOLUME: {"volume_threshold": Decimal("1000")},
            BarType.TICK: {"tick_threshold": 10},
            BarType.DOLLAR: {"dollar_threshold": Decimal("150000")}
        }

        processor = MultiBarProcessor("AAPL", configs)

        assert len(processor.builders) == 3
        assert BarType.VOLUME in processor.builders
        assert BarType.TICK in processor.builders
        assert BarType.DOLLAR in processor.builders

    def test_multi_bar_processor_tick_processing(self):
        """Test processing ticks through multiple builders"""
        configs = {
            BarType.VOLUME: {"volume_threshold": Decimal("250")},
            BarType.TICK: {"tick_threshold": 2}
        }

        processor = MultiBarProcessor("AAPL", configs)

        # Process first tick
        tick1 = self.create_test_tick(150.00, 100)
        bars1 = processor.process_tick(tick1)
        assert len(bars1) == 0  # No bars completed yet

        # Process second tick - should complete tick bar but not volume bar
        tick2 = self.create_test_tick(150.25, 100)
        bars2 = processor.process_tick(tick2)

        assert len(bars2) == 1
        assert BarType.TICK in bars2
        assert isinstance(bars2[BarType.TICK], TickBar)

        # Process third tick - should complete volume bar
        tick3 = self.create_test_tick(150.50, 100)  # Total volume: 300
        bars3 = processor.process_tick(tick3)

        assert len(bars3) == 1
        assert BarType.VOLUME in bars3
        assert isinstance(bars3[BarType.VOLUME], VolumeBar)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])