"""
Comprehensive tests for foundation models

Tests all Pydantic models defined in foundation/models/:
- Base models: BaseFoundationModel, TimestampedModel, ValidatedModel
- Market models: TickData, OHLCVBar, and all advanced bar types
- Metadata models: SymbolDayMetadata and supporting models
- Enumerations: TradeSign, TimeInterval, etc.
"""

import pytest
import uuid
from decimal import Decimal
from datetime import datetime, timezone
from typing import Optional

from foundation.models.base import BaseFoundationModel, TimestampedModel, ValidatedModel
from foundation.models.market import (
    TickData, OHLCVBar, VolumeBar, DollarBar, ImbalanceBar,
    TickBar, RangeBar, RenkoBar, OrderBookSnapshot, MarketSession
)
from foundation.models.metadata import (
    SymbolDayMetadata, SpreadStatistics, TradeClassification,
    LiquidityProfile, ExecutionQuality, MarketRegime, ToxicityMetrics
)
from foundation.models.enums import (
    TradeSign, TimeInterval, ParticipantType, ExchangeCode,
    VolatilityRegime, LiquidityState, TrendState, MicrostructureRegime,
    DataQualityLevel, BarType, TradeCondition
)


class TestBaseModels:
    """Test base foundation models"""

    def test_base_foundation_model_creation(self):
        """Test BaseFoundationModel creation and validation"""
        model = BaseFoundationModel()

        assert isinstance(model.id, uuid.UUID)
        assert isinstance(model.created_at, datetime)
        assert model.created_at.tzinfo == timezone.utc
        assert model.updated_at is None
        assert model.metadata == {}

    def test_base_foundation_model_with_custom_values(self):
        """Test BaseFoundationModel with custom values"""
        custom_id = uuid.uuid4()
        custom_metadata = {"source": "test", "quality": "high"}

        model = BaseFoundationModel(
            id=custom_id,
            metadata=custom_metadata
        )

        assert model.id == custom_id
        assert model.metadata == custom_metadata

    def test_timestamped_model_creation(self):
        """Test TimestampedModel creation"""
        timestamp = datetime.now(timezone.utc)
        model = TimestampedModel(timestamp=timestamp)

        assert model.timestamp == timestamp
        assert isinstance(model.id, uuid.UUID)

    def test_validated_model_creation(self):
        """Test ValidatedModel creation with quality tracking"""
        model = ValidatedModel()

        assert model.is_validated is False
        assert model.validation_errors == []
        assert model.quality_score == 1.0

    def test_validated_model_with_validation_data(self):
        """Test ValidatedModel with validation data"""
        errors = ["Price out of range", "Volume negative"]

        model = ValidatedModel(
            is_validated=True,
            validation_errors=errors,
            quality_score=0.85
        )

        assert model.is_validated is True
        assert model.validation_errors == errors
        assert model.quality_score == 0.85


class TestTickData:
    """Test TickData model"""

    def create_valid_tick(self, **overrides) -> TickData:
        """Create valid TickData for testing using REAL IQFeed data structure"""
        # This matches the ACTUAL IQFeed numpy structured array format
        # Based on: ('tick_id', 'date', 'time', 'last', 'last_sz', 'last_type',
        #            'mkt_ctr', 'tot_vlm', 'bid', 'ask', 'cond1', 'cond2', 'cond3', 'cond4')
        defaults = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "price": Decimal("150.25"),           # 'last' field
            "size": 100,                          # 'last_sz' field (INTEGER)
            "exchange": "Q",                      # 'last_type' field (STRING, Q=NASDAQ)
            "market_center": 11,                  # 'mkt_ctr' field (INTEGER)
            "total_volume": 125000,               # 'tot_vlm' field (INTEGER, required)
            "bid": Decimal("150.24"),             # 'bid' field (optional)
            "ask": Decimal("150.26"),             # 'ask' field (optional)
            # Pre-set computed fields to avoid recursion
            "spread": Decimal("0.02"),
            "midpoint": Decimal("150.25"),
            "dollar_volume": Decimal("15025.00"),
            "conditions": "",                     # Trade conditions (max 4 chars)
            # Pre-computed fields (calculated at ingestion)
            "trade_sign": 1,                      # INTEGER: +1 buy, -1 sell, 0 unknown
            "tick_direction": 1,                  # INTEGER: +1 uptick, -1 downtick
        }
        defaults.update(overrides)
        return TickData(**defaults)

    def test_tick_data_creation(self):
        """Test basic TickData creation"""
        tick = self.create_valid_tick()

        assert tick.symbol == "AAPL"
        assert tick.price == Decimal("150.25")
        assert tick.size == 100                    # INTEGER not Decimal
        assert tick.exchange == "Q"                # STRING exchange code
        assert tick.market_center == 11            # INTEGER market center ID
        assert tick.total_volume == 125000         # Required field
        assert tick.trade_sign == 1                # INTEGER not enum

    def test_tick_data_computed_fields(self):
        """Test TickData computed fields"""
        tick = self.create_valid_tick()

        # Test spread calculation
        assert tick.spread == Decimal("0.02")  # 150.26 - 150.24

        # Test midpoint calculation
        assert tick.midpoint == Decimal("150.25")  # (150.24 + 150.26) / 2

    def test_tick_data_validation_price_positive(self):
        """Test TickData validation - price must be positive"""
        with pytest.raises(ValueError, match="Price must be positive"):
            self.create_valid_tick(price=Decimal("-10.00"))

    def test_tick_data_validation_size_positive(self):
        """Test TickData validation - size must be positive"""
        with pytest.raises(ValueError, match="Input should be greater than 0"):
            self.create_valid_tick(size=-100)

    def test_tick_data_validation_bid_ask_consistency(self):
        """Test TickData validation - bid <= ask"""
        with pytest.raises(ValueError, match="Bid price cannot exceed ask price"):
            self.create_valid_tick(bid=Decimal("150.30"), ask=Decimal("150.20"))

    def test_tick_data_validation_symbol_format(self):
        """Test TickData validation - symbol format"""
        with pytest.raises(ValueError):
            self.create_valid_tick(symbol="invalid symbol")

    def test_tick_data_with_optional_fields(self):
        """Test TickData with optional fields and pre-computed metrics"""
        tick = self.create_valid_tick(
            # Test optional bid/ask fields
            bid=Decimal("150.20"),
            ask=Decimal("150.30"),
            # Test pre-computed metrics
            spread=Decimal("0.10"),
            midpoint=Decimal("150.25"),
            spread_bps=6.67,
            dollar_volume=Decimal("15025.00")
        )

        assert tick.bid == Decimal("150.20")
        assert tick.ask == Decimal("150.30")
        assert tick.spread == Decimal("0.10")
        assert tick.midpoint == Decimal("150.25")
        assert tick.spread_bps == 6.67
        assert tick.dollar_volume == Decimal("15025.00")


class TestOHLCVBar:
    """Test OHLCVBar model"""

    def create_valid_ohlcv_bar(self, **overrides) -> OHLCVBar:
        """Create valid OHLCVBar for testing using REAL model structure"""
        defaults = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc),
            "interval": TimeInterval.MINUTE_1,       # Correct field name
            "open": Decimal("150.00"),
            "high": Decimal("150.50"),
            "low": Decimal("149.75"),
            "close": Decimal("150.25"),
            "volume": 10000,                         # INTEGER not Decimal
            "tick_count": 250,                       # Correct field name
            "dollar_volume": Decimal("1502500.00"),
            "vwap": Decimal("150.15"),               # Required field
        }
        defaults.update(overrides)
        return OHLCVBar(**defaults)

    def test_ohlcv_bar_creation(self):
        """Test basic OHLCVBar creation"""
        bar = self.create_valid_ohlcv_bar()

        assert bar.symbol == "AAPL"
        assert bar.interval == TimeInterval.MINUTE_1
        assert bar.open == Decimal("150.00")
        assert bar.high == Decimal("150.50")
        assert bar.low == Decimal("149.75")
        assert bar.close == Decimal("150.25")
        assert bar.volume == 10000                  # INTEGER not Decimal
        assert bar.tick_count == 250                # Correct field name
        assert bar.vwap == Decimal("150.15")

    def test_ohlcv_bar_price_validation(self):
        """Test OHLCVBar price validation - high >= max(open, close), low <= min(open, close)"""
        with pytest.raises(ValueError, match="High price must be >= max\\(open, close\\)"):
            self.create_valid_ohlcv_bar(
                open=Decimal("150.00"),
                high=Decimal("149.00"),  # Invalid: high < open
                close=Decimal("150.25")
            )

        with pytest.raises(ValueError, match="Low price must be <= min\\(open, close\\)"):
            self.create_valid_ohlcv_bar(
                open=Decimal("150.00"),
                low=Decimal("151.00"),  # Invalid: low > open
                close=Decimal("150.25")
            )

    def test_ohlcv_bar_volume_consistency(self):
        """Test OHLCVBar volume validation - volume must be positive"""
        with pytest.raises(ValueError, match="Input should be greater than or equal to 0"):
            self.create_valid_ohlcv_bar(volume=-1000)

    def test_ohlcv_bar_with_metadata_fields(self):
        """Test OHLCVBar with metadata fields"""
        bar = self.create_valid_ohlcv_bar(
            avg_spread_bps=5.5,
            spread_volatility=1.2,
            buy_dollar_volume=Decimal("900000.00"),
            vwap=Decimal("150.15"),
            avg_tick_size=Decimal("0.01"),
            effective_spread_mean=0.08,
            liquidity_score=75.5
        )

        assert bar.avg_spread_bps == 5.5
        assert bar.spread_volatility == 1.2
        assert bar.buy_dollar_volume == Decimal("900000.00")
        assert bar.vwap == Decimal("150.15")
        assert bar.liquidity_score == 75.5


class TestAdvancedBarTypes:
    """Test advanced bar types: VolumeBar, DollarBar, ImbalanceBar, etc."""

    def test_volume_bar_creation(self):
        """Test VolumeBar creation"""
        bar = VolumeBar(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("150.00"),
            high=Decimal("150.50"),
            low=Decimal("149.75"),
            close=Decimal("150.25"),
            volume=Decimal("10000"),
            dollar_volume=Decimal("1502500.00"),
            trade_count=250,
            buy_volume=Decimal("6000"),
            sell_volume=Decimal("4000"),
            volume_threshold=Decimal("10000")
        )

        assert bar.volume_threshold == Decimal("10000")
        assert bar.volume >= bar.volume_threshold

    def test_dollar_bar_creation(self):
        """Test DollarBar creation"""
        bar = DollarBar(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("150.00"),
            high=Decimal("150.50"),
            low=Decimal("149.75"),
            close=Decimal("150.25"),
            volume=Decimal("10000"),
            dollar_volume=Decimal("1502500.00"),
            trade_count=250,
            buy_volume=Decimal("6000"),
            sell_volume=Decimal("4000"),
            dollar_threshold=Decimal("1500000.00")
        )

        assert bar.dollar_threshold == Decimal("1500000.00")
        assert bar.dollar_volume >= bar.dollar_threshold

    def test_imbalance_bar_creation(self):
        """Test ImbalanceBar creation"""
        bar = ImbalanceBar(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("150.00"),
            high=Decimal("150.50"),
            low=Decimal("149.75"),
            close=Decimal("150.25"),
            volume=Decimal("10000"),
            dollar_volume=Decimal("1502500.00"),
            trade_count=250,
            buy_volume=Decimal("6000"),
            sell_volume=Decimal("4000"),
            imbalance_threshold=Decimal("2000"),
            imbalance_ratio=0.2
        )

        assert bar.imbalance_threshold == Decimal("2000")
        assert bar.imbalance_ratio == 0.2
        assert abs(bar.buy_volume - bar.sell_volume) >= bar.imbalance_threshold

    def test_tick_bar_creation(self):
        """Test TickBar creation"""
        bar = TickBar(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("150.00"),
            high=Decimal("150.50"),
            low=Decimal("149.75"),
            close=Decimal("150.25"),
            volume=Decimal("10000"),
            dollar_volume=Decimal("1502500.00"),
            trade_count=250,
            buy_volume=Decimal("6000"),
            sell_volume=Decimal("4000"),
            tick_threshold=250
        )

        assert bar.tick_threshold == 250
        assert bar.trade_count >= bar.tick_threshold

    def test_range_bar_creation(self):
        """Test RangeBar creation"""
        bar = RangeBar(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("150.00"),
            high=Decimal("150.50"),
            low=Decimal("149.75"),
            close=Decimal("150.25"),
            volume=Decimal("10000"),
            dollar_volume=Decimal("1502500.00"),
            trade_count=250,
            buy_volume=Decimal("6000"),
            sell_volume=Decimal("4000"),
            price_range=Decimal("0.75")
        )

        assert bar.price_range == Decimal("0.75")
        assert (bar.high - bar.low) >= bar.price_range

    def test_renko_bar_creation(self):
        """Test RenkoBar creation"""
        bar = RenkoBar(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            open=Decimal("150.00"),
            high=Decimal("150.50"),
            low=Decimal("150.00"),
            close=Decimal("150.50"),
            volume=Decimal("5000"),
            dollar_volume=Decimal("751250.00"),
            trade_count=125,
            buy_volume=Decimal("3000"),
            sell_volume=Decimal("2000"),
            brick_size=Decimal("0.50"),
            is_bullish=True
        )

        assert bar.brick_size == Decimal("0.50")
        assert bar.is_bullish is True
        assert (bar.close - bar.open) == bar.brick_size


class TestMetadataModels:
    """Test metadata models"""

    def test_spread_statistics_creation(self):
        """Test SpreadStatistics creation"""
        stats = SpreadStatistics(
            mean_bps=5.5,
            median_bps=5.0,
            std_bps=1.2,
            min_bps=3.0,
            max_bps=12.0,
            p25_bps=4.0,
            p75_bps=7.0,
            p95_bps=8.5,
            p99_bps=10.0
        )

        assert stats.mean_bps == 5.5
        assert stats.median_bps == 5.0
        assert stats.std_bps == 1.2

    def test_trade_classification_creation(self):
        """Test TradeClassification creation"""
        classification = TradeClassification(
            buy_count=1500,
            sell_count=1200,
            neutral_count=300,
            buy_volume=550000,
            sell_volume=400000,
            buy_dollar_volume=Decimal("82500000"),
            sell_dollar_volume=Decimal("60000000")
        )

        assert classification.buy_count == 1500
        assert classification.sell_count == 1200
        assert classification.buy_volume == 550000

    def test_liquidity_profile_creation(self):
        """Test LiquidityProfile creation"""
        profile = LiquidityProfile(
            quote_intensity=15.5,
            avg_trade_size=250.0,
            median_trade_size=200.0,
            trade_frequency=25.5,
            effective_tick_size=0.01,
            price_levels_count=150,
            time_between_trades_ms=2352.5,
            liquidity_score=75.5
        )

        assert profile.quote_intensity == 15.5
        assert profile.avg_trade_size == 250.0
        assert profile.liquidity_score == 75.5

    def test_symbol_day_metadata_creation(self):
        """Test SymbolDayMetadata creation"""
        now = datetime.now(timezone.utc)

        metadata = SymbolDayMetadata(
            symbol="AAPL",
            date=now,
            total_ticks=5000,
            first_tick_time=now.replace(hour=9, minute=30),
            last_tick_time=now.replace(hour=16, minute=0),
            price_open=Decimal("150.00"),
            price_high=Decimal("152.50"),
            price_low=Decimal("149.75"),
            price_close=Decimal("151.25"),
            volume_total=1500000,
            dollar_volume=Decimal("225000000"),
            vwap=Decimal("150.85"),
            spread_stats=SpreadStatistics(
                mean_bps=5.5,
                median_bps=5.0,
                std_bps=1.2,
                min_bps=3.0,
                max_bps=12.0,
                p25_bps=4.0,
                p75_bps=7.0,
                p95_bps=8.5,
                p99_bps=10.0
            ),
            trade_classification=TradeClassification(
                buy_count=1500,
                sell_count=1200,
                neutral_count=300,
                buy_volume=550000,
                sell_volume=400000,
                buy_dollar_volume=Decimal("82500000"),
                sell_dollar_volume=Decimal("60000000")
            ),
            liquidity_profile=LiquidityProfile(
                quote_intensity=15.5,
                avg_trade_size=250.0,
                median_trade_size=200.0,
                trade_frequency=25.5,
                effective_tick_size=0.01,
                price_levels_count=150,
                time_between_trades_ms=2352.5,
                liquidity_score=75.5
            ),
            execution_quality=ExecutionQuality(
                effective_spread_mean=0.05,
                effective_spread_median=0.04,
                price_improvement_rate=0.15,
                at_midpoint_rate=0.08,
                at_bid_rate=0.42,
                at_ask_rate=0.45,
                outside_quote_rate=0.02,
                odd_lot_rate=0.12,
                block_rate=0.03
            )
        )

        assert metadata.symbol == "AAPL"
        assert metadata.total_ticks == 5000
        assert metadata.volume_total == 1500000


class TestEnumerations:
    """Test enumeration models"""

    def test_trade_sign_enum(self):
        """Test TradeSign enumeration"""
        assert TradeSign.BUY == 1
        assert TradeSign.SELL == -1
        assert TradeSign.NEUTRAL == 0

        # Test string conversion
        assert str(TradeSign.BUY) == "TradeSign.BUY"

    def test_time_interval_enum(self):
        """Test TimeInterval enumeration"""
        assert TimeInterval.SECOND_1 == "1s"
        assert TimeInterval.MINUTE_1 == "1m"
        assert TimeInterval.MINUTE_5 == "5m"
        assert TimeInterval.HOUR_1 == "1h"
        assert TimeInterval.DAILY == "1d"

    def test_exchange_code_enum(self):
        """Test ExchangeCode enumeration"""
        assert ExchangeCode.NASDAQ == "Q"
        assert ExchangeCode.NYSE == "N"
        assert ExchangeCode.NYSE_ARCA == "O"
        assert ExchangeCode.IEX == "V"

    def test_volatility_regime_enum(self):
        """Test VolatilityRegime enumeration"""
        assert VolatilityRegime.LOW == "low"
        assert VolatilityRegime.NORMAL == "normal"
        assert VolatilityRegime.HIGH == "high"
        assert VolatilityRegime.EXTREME == "extreme"

    def test_data_quality_level_enum(self):
        """Test DataQualityLevel enumeration"""
        assert DataQualityLevel.HIGH == "high"
        assert DataQualityLevel.MEDIUM == "medium"
        assert DataQualityLevel.LOW == "low"
        assert DataQualityLevel.POOR == "poor"

    def test_bar_type_enum(self):
        """Test BarType enumeration"""
        assert BarType.TIME == "time"
        assert BarType.VOLUME == "volume"
        assert BarType.DOLLAR == "dollar"
        assert BarType.IMBALANCE == "imbalance"
        assert BarType.TICK == "tick"
        assert BarType.RANGE == "range"
        assert BarType.RENKO == "renko"


class TestOrderBookSnapshot:
    """Test OrderBookSnapshot model"""

    def test_order_book_snapshot_creation(self):
        """Test OrderBookSnapshot creation"""
        snapshot = OrderBookSnapshot(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            bid_prices=[Decimal("150.24"), Decimal("150.23"), Decimal("150.22")],
            bid_sizes=[Decimal("500"), Decimal("300"), Decimal("200")],
            ask_prices=[Decimal("150.26"), Decimal("150.27"), Decimal("150.28")],
            ask_sizes=[Decimal("400"), Decimal("250"), Decimal("150")],
            depth_levels=3
        )

        assert snapshot.symbol == "AAPL"
        assert snapshot.depth_levels == 3
        assert len(snapshot.bid_prices) == 3
        assert len(snapshot.ask_prices) == 3

    def test_order_book_validation_price_arrays_length(self):
        """Test OrderBookSnapshot validation - price and size arrays must have same length"""
        with pytest.raises(ValueError, match="Bid prices and sizes must have the same length"):
            OrderBookSnapshot(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                bid_prices=[Decimal("150.24"), Decimal("150.23")],  # 2 prices
                bid_sizes=[Decimal("500")],  # 1 size
                ask_prices=[Decimal("150.26")],
                ask_sizes=[Decimal("400")],
                depth_levels=1
            )

    def test_order_book_validation_price_ordering(self):
        """Test OrderBookSnapshot validation - bid prices descending, ask prices ascending"""
        with pytest.raises(ValueError, match="Bid prices must be in descending order"):
            OrderBookSnapshot(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                bid_prices=[Decimal("150.22"), Decimal("150.24")],  # Wrong order
                bid_sizes=[Decimal("500"), Decimal("300")],
                ask_prices=[Decimal("150.26")],
                ask_sizes=[Decimal("400")],
                depth_levels=2
            )


class TestMarketSession:
    """Test MarketSession model"""

    def test_market_session_creation(self):
        """Test MarketSession creation"""
        start_time = datetime.now(timezone.utc).replace(hour=9, minute=30, second=0)
        end_time = datetime.now(timezone.utc).replace(hour=16, minute=0, second=0)

        session = MarketSession(
            symbol="AAPL",
            date=datetime.now(timezone.utc),
            session_start=start_time,
            session_end=end_time,
            total_ticks=5000,
            total_volume=1500000,
            total_dollar_volume=Decimal("225000000"),
            session_open=Decimal("150.00"),
            session_high=Decimal("152.50"),
            session_low=Decimal("149.75"),
            session_close=Decimal("151.25")
        )

        assert session.symbol == "AAPL"
        assert session.total_ticks == 5000
        assert session.total_volume == 1500000

    def test_market_session_validation_time_order(self):
        """Test MarketSession validation - session_start < session_end"""
        start_time = datetime.now(timezone.utc).replace(hour=16, minute=0)
        end_time = datetime.now(timezone.utc).replace(hour=9, minute=30)  # Before start

        with pytest.raises(ValueError, match="End time must be after start time"):
            MarketSession(
                symbol="AAPL",
                date=datetime.now(timezone.utc),
                session_start=start_time,
                session_end=end_time,
                total_ticks=5000,
                total_volume=1500000,
                total_dollar_volume=Decimal("225000000"),
                session_open=Decimal("150.00"),
                session_high=Decimal("152.50"),
                session_low=Decimal("149.75"),
                session_close=Decimal("151.25")
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])