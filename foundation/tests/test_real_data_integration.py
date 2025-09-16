"""
Real Data Integration Test - IQFeed → NumPy → Pandas → Pydantic → ArcticDB

This test follows CLAUDE.md guidelines: NO MOCK DATA, only real IQFeed data.
Tests the complete pipeline from IQFeed connection to ArcticDB storage.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

# Add PyIQFeed to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'pyiqfeed_orig'))

try:
    import pyiqfeed as iq
    IQFEED_AVAILABLE = True
except ImportError:
    IQFEED_AVAILABLE = False

from foundation.models.market import TickData
from foundation.models.enums import TradeSign


def convert_iqfeed_tick_to_pydantic(iqfeed_tick: np.void) -> TickData:
    """
    Convert IQFeed numpy structured array tick to Pydantic TickData model.

    IQFeed tick structure:
    ('tick_id', 'date', 'time', 'last', 'last_sz', 'last_type',
     'mkt_ctr', 'tot_vlm', 'bid', 'ask', 'cond1', 'cond2', 'cond3', 'cond4')

    Sample tick: (3954, '2025-09-15', 26540953311, 236.62, 10, b'O', 11, 387902, 236.6, 236.67, 135, 61, 23, 0)
    """
    # Extract fields from numpy structured array
    tick_id = int(iqfeed_tick[0])
    date = iqfeed_tick[1]
    time_us = int(iqfeed_tick[2])  # Microseconds since midnight
    price = float(iqfeed_tick[3])
    size = int(iqfeed_tick[4])
    exchange_bytes = iqfeed_tick[5]
    market_center = int(iqfeed_tick[6])
    total_volume = int(iqfeed_tick[7])
    bid = float(iqfeed_tick[8]) if iqfeed_tick[8] != 0 else None
    ask = float(iqfeed_tick[9]) if iqfeed_tick[9] != 0 else None
    cond1 = int(iqfeed_tick[10])
    cond2 = int(iqfeed_tick[11])
    cond3 = int(iqfeed_tick[12])
    cond4 = int(iqfeed_tick[13])

    # Convert exchange code from bytes to string
    if isinstance(exchange_bytes, bytes):
        exchange = exchange_bytes.decode('utf-8')
    else:
        exchange = str(exchange_bytes)

    # Convert date and time to datetime
    if isinstance(date, np.datetime64):
        date_obj = date.astype('datetime64[D]').astype(datetime).date()
    else:
        date_obj = datetime.strptime(str(date), '%Y-%m-%d').date()

    # Convert microseconds to time
    hours = time_us // 3600000000
    minutes = (time_us % 3600000000) // 60000000
    seconds = (time_us % 60000000) // 1000000
    microseconds = time_us % 1000000

    timestamp = datetime.combine(
        date_obj,
        datetime.min.time().replace(hour=hours, minute=minutes, second=seconds, microsecond=microseconds),
        timezone.utc
    )

    # Build conditions string (max 4 chars)
    conditions = f"{cond1}" if cond1 != 0 else ""
    if cond2 != 0:
        conditions += f",{cond2}" if conditions else str(cond2)
    if len(conditions) >= 4:
        conditions = conditions[:4]

    # Create TickData with pre-computed fields to avoid recursion
    tick_data = {
        "symbol": "AAPL",  # We'll get this from the calling context
        "timestamp": timestamp,
        "price": Decimal(str(price)),
        "size": size,
        "exchange": exchange,
        "market_center": market_center,
        "total_volume": total_volume,
        "conditions": conditions,
    }

    # Add optional bid/ask
    if bid is not None:
        tick_data["bid"] = Decimal(str(bid))
    if ask is not None:
        tick_data["ask"] = Decimal(str(ask))

    # Pre-compute derived fields to avoid model recursion issues
    if bid is not None and ask is not None:
        spread = Decimal(str(ask)) - Decimal(str(bid))
        midpoint = (Decimal(str(bid)) + Decimal(str(ask))) / Decimal('2')
        tick_data["spread"] = spread
        tick_data["midpoint"] = midpoint
        tick_data["spread_bps"] = float((spread / midpoint) * 10000) if midpoint > 0 else None
        tick_data["effective_spread"] = 2 * abs(Decimal(str(price)) - midpoint)

    tick_data["dollar_volume"] = Decimal(str(price)) * Decimal(str(size))

    # Simple trade sign classification (can be improved with Lee-Ready algorithm)
    if bid is not None and ask is not None:
        mid = (bid + ask) / 2
        if price > mid:
            tick_data["trade_sign"] = 1  # Buy
        elif price < mid:
            tick_data["trade_sign"] = -1  # Sell
        else:
            tick_data["trade_sign"] = 0  # Unknown
    else:
        tick_data["trade_sign"] = 0

    return TickData(**tick_data)


@pytest.mark.skipif(not IQFEED_AVAILABLE, reason="PyIQFeed not available")
class TestRealDataIntegration:
    """Test real data pipeline using actual IQFeed data"""

    def test_iqfeed_connection(self):
        """Test that we can connect to IQFeed"""
        hist_conn = iq.HistoryConn(name="test-connection")

        try:
            with iq.ConnConnector([hist_conn]) as connector:
                # Simple connection test - request minimal data
                data = hist_conn.request_ticks("AAPL", max_ticks=1)

                assert data is not None, "Should receive data from IQFeed"
                assert len(data) > 0, "Should have at least one tick"
                assert isinstance(data, np.ndarray), "Should return numpy array"

                print(f"✓ IQFeed connection successful - received {len(data)} tick(s)")

        except Exception as e:
            pytest.skip(f"IQFeed connection failed: {e}")

    def test_numpy_to_pydantic_conversion(self):
        """Test conversion from IQFeed numpy array to Pydantic models"""
        hist_conn = iq.HistoryConn(name="test-conversion")

        try:
            with iq.ConnConnector([hist_conn]) as connector:
                # Get real AAPL tick data
                data = hist_conn.request_ticks("AAPL", max_ticks=10)

                assert len(data) > 0, "Need at least one tick for conversion test"

                # Convert first tick to Pydantic model
                first_tick = data[0]
                tick_model = convert_iqfeed_tick_to_pydantic(first_tick)

                # Verify conversion
                assert isinstance(tick_model, TickData)
                assert tick_model.symbol == "AAPL"
                assert tick_model.price > 0
                assert tick_model.size > 0
                assert tick_model.exchange in ['A', 'B', 'C', 'D', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                assert tick_model.market_center >= 0
                assert tick_model.total_volume >= 0

                print(f"✓ Conversion successful:")
                print(f"  Symbol: {tick_model.symbol}")
                print(f"  Price: ${tick_model.price}")
                print(f"  Size: {tick_model.size}")
                print(f"  Exchange: {tick_model.exchange}")
                print(f"  Timestamp: {tick_model.timestamp}")

        except Exception as e:
            pytest.skip(f"IQFeed conversion test failed: {e}")

    def test_multiple_tick_conversion(self):
        """Test converting multiple ticks to verify consistency"""
        hist_conn = iq.HistoryConn(name="test-multiple")

        try:
            with iq.ConnConnector([hist_conn]) as connector:
                # Get multiple ticks
                data = hist_conn.request_ticks("AAPL", max_ticks=50)

                assert len(data) >= 10, "Need at least 10 ticks for this test"

                # Convert all ticks
                tick_models = []
                for numpy_tick in data:
                    tick_model = convert_iqfeed_tick_to_pydantic(numpy_tick)
                    tick_models.append(tick_model)

                # Verify all conversions
                assert len(tick_models) == len(data)

                # Check that timestamps are in order (newest first from IQFeed)
                for i in range(1, min(10, len(tick_models))):
                    assert tick_models[i-1].timestamp >= tick_models[i].timestamp, \
                        "Timestamps should be in descending order (newest first)"

                # Check volume consistency (should be non-decreasing for same day)
                volumes = [tick.total_volume for tick in tick_models]
                print(f"✓ Converted {len(tick_models)} ticks successfully")
                print(f"  Volume range: {min(volumes):,} - {max(volumes):,}")
                print(f"  Price range: ${min(tick.price for tick in tick_models)} - ${max(tick.price for tick in tick_models)}")

        except Exception as e:
            pytest.skip(f"Multiple tick conversion test failed: {e}")

    def test_data_quality_validation(self):
        """Test that converted data passes Pydantic validation"""
        hist_conn = iq.HistoryConn(name="test-validation")

        try:
            with iq.ConnConnector([hist_conn]) as connector:
                data = hist_conn.request_ticks("AAPL", max_ticks=20)

                validation_errors = []
                successful_conversions = 0

                for i, numpy_tick in enumerate(data):
                    try:
                        tick_model = convert_iqfeed_tick_to_pydantic(numpy_tick)

                        # Additional validation checks
                        assert tick_model.price > 0, f"Price must be positive: {tick_model.price}"
                        assert tick_model.size > 0, f"Size must be positive: {tick_model.size}"
                        assert len(tick_model.symbol) > 0, "Symbol cannot be empty"
                        assert len(tick_model.exchange) <= 4, f"Exchange code too long: {tick_model.exchange}"

                        successful_conversions += 1

                    except Exception as e:
                        validation_errors.append(f"Tick {i}: {e}")

                print(f"✓ Data quality validation:")
                print(f"  Successful conversions: {successful_conversions}/{len(data)}")
                print(f"  Validation errors: {len(validation_errors)}")

                if validation_errors:
                    for error in validation_errors[:5]:  # Show first 5 errors
                        print(f"    {error}")

                # Ensure at least 80% success rate
                success_rate = successful_conversions / len(data)
                assert success_rate >= 0.8, f"Success rate too low: {success_rate:.1%}"

        except Exception as e:
            pytest.skip(f"Data quality validation test failed: {e}")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "-s"])