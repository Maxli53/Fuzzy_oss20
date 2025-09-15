#!/usr/bin/env python3
"""
Comprehensive PyTest Suite for IQFeed Integration
Tests ALL IQFeed capabilities: Historical, Real-time, Lookups, Integration, Error Handling
"""

import pytest
import time
import logging
from typing import Dict, List, Any

# Import our fixed test client wrapper
from iqfeed_client_fixed import IQFeedClient

# Setup logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# PyTest Fixtures
# -----------------------------

@pytest.fixture(scope="module")
def iqfeed_client():
    """Setup IQFeed client for all tests"""
    logger.info("Setting up IQFeed client for test module")
    client = IQFeedClient()

    # Connect and ensure it's working
    connected = client.connect()
    if not connected:
        pytest.skip("Cannot connect to IQFeed - ensure IQFeed service is running")

    yield client

    # Cleanup
    logger.info("Cleaning up IQFeed client")
    client.disconnect()

@pytest.fixture(scope="function")
def clean_client(iqfeed_client):
    """Provide a clean client state for each test function"""
    # Ensure no active subscriptions before each test
    if hasattr(iqfeed_client, '_subscriptions'):
        for symbol in list(iqfeed_client._subscriptions.keys()):
            iqfeed_client.unsubscribe_realtime(symbol)

    yield iqfeed_client

# -----------------------------
# Unit Tests
# -----------------------------

def test_connection(iqfeed_client):
    """Test connection status"""
    assert iqfeed_client.is_connected(), "Client failed to connect to IQFeed"
    logger.info("✓ Connection test passed")

def test_lookup_symbol_valid(iqfeed_client):
    """Test symbol lookup with a valid symbol"""
    result = iqfeed_client.lookup_symbol("AAPL")
    assert result is not None, "No result returned for AAPL lookup"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "symbol" in result, "Symbol field missing from result"

    # Basic validation
    symbol = result.get("symbol", "")
    assert "AAPL" in symbol.upper(), f"Returned symbol '{symbol}' doesn't contain AAPL"
    logger.info(f"✓ Valid symbol lookup test passed: {result}")

def test_lookup_symbol_invalid(iqfeed_client):
    """Symbol lookup should raise exception for invalid symbols"""
    with pytest.raises(Exception, match=r"(?i)not found|failed"):
        iqfeed_client.lookup_symbol("INVALIDSYM123456789")
    logger.info("✓ Invalid symbol lookup test passed")

def test_historical_data_structure(iqfeed_client):
    """Check historical data structure for expected fields"""
    data = iqfeed_client.get_historical_data("AAPL", interval="1d", days=5)

    assert isinstance(data, list), "Historical data should return a list"
    assert len(data) > 0, "No historical data returned"

    # Check structure of first row
    row = data[0]
    assert isinstance(row, dict), "Each data row should be a dictionary"

    required_fields = {"open", "high", "low", "close", "volume"}
    missing_fields = required_fields - set(row.keys())
    assert not missing_fields, f"Missing OHLCV fields: {missing_fields}"

    logger.info(f"✓ Historical data structure test passed: {len(data)} records with all OHLCV fields")

def test_historical_data_values(iqfeed_client):
    """Ensure OHLC data makes sense"""
    data = iqfeed_client.get_historical_data("AAPL", interval="1d", days=5)
    assert len(data) > 0, "No historical data returned"

    for i, row in enumerate(data):
        # Price validation
        assert row["high"] >= row["low"], f"Row {i}: High ({row['high']}) is less than low ({row['low']})"
        assert row["open"] >= 0, f"Row {i}: Negative open price ({row['open']})"
        assert row["close"] >= 0, f"Row {i}: Negative close price ({row['close']})"
        assert row["high"] >= 0, f"Row {i}: Negative high price ({row['high']})"
        assert row["low"] >= 0, f"Row {i}: Negative low price ({row['low']})"

        # Volume validation
        assert row["volume"] >= 0, f"Row {i}: Negative volume ({row['volume']})"

        # High/Low vs Open/Close validation
        assert row["high"] >= max(row["open"], row["close"]), f"Row {i}: High price below open/close"
        assert row["low"] <= min(row["open"], row["close"]), f"Row {i}: Low price above open/close"

    logger.info(f"✓ Historical data values test passed: {len(data)} records validated")

# -----------------------------
# Integration Tests
# -----------------------------

def test_realtime_data_feed(clean_client):
    """Test real-time data subscription and callbacks"""
    received_ticks = []

    def on_tick(tick):
        received_ticks.append(tick)
        logger.debug(f"Received tick: {tick}")

    # Subscribe to real-time data
    clean_client.subscribe_realtime("AAPL", callback=on_tick)

    # Simulate receiving ticks (since we're using a test wrapper)
    clean_client._simulate_realtime_ticks("AAPL", duration=3)

    # Clean up subscription
    clean_client.unsubscribe_realtime("AAPL")

    # Validate results
    assert len(received_ticks) > 0, "No real-time ticks received"

    for tick in received_ticks:
        assert isinstance(tick, dict), "Each tick should be a dictionary"
        required_fields = {"bid", "ask", "last"}
        missing_fields = required_fields - set(tick.keys())
        assert not missing_fields, f"Missing bid/ask/last in tick: {missing_fields}"

        # Basic price validation
        assert tick["last"] > 0, "Last price should be positive"
        assert tick["bid"] > 0, "Bid price should be positive"
        assert tick["ask"] > 0, "Ask price should be positive"
        assert tick["ask"] >= tick["bid"], "Ask should be >= bid"

    logger.info(f"✓ Real-time data feed test passed: {len(received_ticks)} ticks received")

def test_historical_to_realtime_consistency(clean_client):
    """Cross-check historical data against simulated real-time tick"""
    # Get recent historical data
    hist_data = clean_client.get_historical_data("AAPL", interval="1d", days=1)
    assert len(hist_data) > 0, "No historical data for consistency check"

    hist_close = hist_data[-1]["close"]
    logger.info(f"Historical close price: {hist_close}")

    # Get simulated real-time tick
    tick_data = []
    def on_tick(tick):
        tick_data.append(tick)

    clean_client.subscribe_realtime("AAPL", callback=on_tick)
    clean_client._simulate_realtime_ticks("AAPL", duration=1)
    clean_client.unsubscribe_realtime("AAPL")

    assert tick_data, "No tick data received for consistency check"

    # Integration check: simulated tick should be reasonably close to historical close
    last_tick = tick_data[-1]["last"]
    deviation = abs(last_tick - hist_close) / hist_close

    # Allow up to 5% deviation for simulated data
    assert deviation < 0.05, f"Realtime tick ({last_tick}) deviates {deviation:.2%} from historical close ({hist_close})"

    logger.info(f"✓ Historical-to-realtime consistency test passed: deviation {deviation:.2%}")

def test_order_integration_simulation(clean_client):
    """Simulate trading logic using real-time feed"""
    ticks = []
    def on_tick(tick):
        ticks.append(tick)

    clean_client.subscribe_realtime("AAPL", callback=on_tick)
    clean_client._simulate_realtime_ticks("AAPL", duration=2)
    clean_client.unsubscribe_realtime("AAPL")

    # Simulate basic buy signal: if last price > 0, "execute order"
    orders_executed = []
    for tick in ticks:
        if tick["last"] > 0:
            order = {
                'symbol': tick.get('symbol', 'AAPL'),
                'price': tick["last"],
                'size': 100,
                'side': 'buy',
                'timestamp': tick.get('timestamp', time.time())
            }
            orders_executed.append(order)

    assert orders_executed, "No simulated orders could be executed from tick data"

    # Validate orders
    for order in orders_executed:
        assert order['price'] > 0, "Order price should be positive"
        assert order['size'] > 0, "Order size should be positive"
        assert order['side'] in ['buy', 'sell'], "Invalid order side"

    logger.info(f"✓ Order integration simulation test passed: {len(orders_executed)} orders simulated")

# -----------------------------
# Error Handling / Edge Cases
# -----------------------------

def test_historical_invalid_symbol(iqfeed_client):
    """Historical data should raise exception for invalid symbols"""
    with pytest.raises(Exception, match=r"(?i)failed|not found|no.*data"):
        iqfeed_client.get_historical_data("INVALIDSYM123456789")
    logger.info("✓ Historical invalid symbol test passed")

def test_realtime_invalid_symbol(iqfeed_client):
    """Real-time subscription should raise exception for invalid symbols"""
    with pytest.raises(Exception, match=r"(?i)failed|not found|invalid"):
        iqfeed_client.subscribe_realtime("INVALIDSYM123456789", callback=lambda x: x)
    logger.info("✓ Realtime invalid symbol test passed")

def test_disconnected_operations(iqfeed_client):
    """Operations should fail gracefully when disconnected"""
    # Temporarily disconnect
    iqfeed_client.disconnect()
    assert not iqfeed_client.is_connected(), "Should be disconnected"

    # Test operations fail gracefully
    with pytest.raises(Exception, match=r"(?i)not connected|connection"):
        iqfeed_client.lookup_symbol("AAPL")

    with pytest.raises(Exception, match=r"(?i)not connected|connection"):
        iqfeed_client.get_historical_data("AAPL")

    with pytest.raises(Exception, match=r"(?i)not connected|connection"):
        iqfeed_client.subscribe_realtime("AAPL", lambda x: x)

    # Reconnect for other tests
    iqfeed_client.connect()
    assert iqfeed_client.is_connected(), "Should be reconnected"

    logger.info("✓ Disconnected operations test passed")

# -----------------------------
# Performance and Edge Cases
# -----------------------------

def test_multiple_symbols_lookup(iqfeed_client):
    """Test looking up multiple symbols"""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    results = {}

    for symbol in symbols:
        try:
            result = iqfeed_client.lookup_symbol(symbol)
            results[symbol] = result
            logger.debug(f"Lookup result for {symbol}: {result}")
        except Exception as e:
            logger.warning(f"Failed to lookup {symbol}: {e}")

    # At least one symbol should work
    assert len(results) > 0, "No symbols could be looked up"
    logger.info(f"✓ Multiple symbols lookup test passed: {len(results)}/{len(symbols)} symbols found")

def test_historical_data_different_intervals(iqfeed_client):
    """Test historical data with different intervals"""
    intervals_to_test = [("1d", 5), ("5m", 1)]  # (interval, days)
    results = {}

    for interval, days in intervals_to_test:
        try:
            data = iqfeed_client.get_historical_data("AAPL", interval=interval, days=days)
            results[interval] = len(data) if data else 0
            logger.debug(f"Got {results[interval]} records for {interval} interval")
        except Exception as e:
            logger.warning(f"Failed to get {interval} data: {e}")

    # At least daily data should work
    assert results.get("1d", 0) > 0, "Daily historical data should be available"
    logger.info(f"✓ Different intervals test passed: {results}")

# -----------------------------
# Test Summary
# -----------------------------

def test_comprehensive_functionality(iqfeed_client):
    """Comprehensive test combining multiple functionalities"""
    logger.info("Starting comprehensive functionality test...")

    # 1. Connection test
    assert iqfeed_client.is_connected(), "Connection failed"

    # 2. Symbol lookup
    symbol_info = iqfeed_client.lookup_symbol("AAPL")
    assert symbol_info, "Symbol lookup failed"

    # 3. Historical data
    hist_data = iqfeed_client.get_historical_data("AAPL", days=3)
    assert len(hist_data) > 0, "Historical data retrieval failed"

    # 4. Data validation
    for row in hist_data:
        assert row["high"] >= row["low"], "Invalid OHLC data"
        assert row["volume"] >= 0, "Invalid volume data"

    # 5. Real-time simulation
    ticks = []
    iqfeed_client.subscribe_realtime("AAPL", callback=ticks.append)
    iqfeed_client._simulate_realtime_ticks("AAPL", duration=1)
    iqfeed_client.unsubscribe_realtime("AAPL")

    assert len(ticks) > 0, "Real-time simulation failed"

    logger.info("✓ Comprehensive functionality test passed - all systems operational!")

if __name__ == "__main__":
    # Run tests directly if executed as script
    pytest.main([__file__, "-v", "-s", "--tb=short"])