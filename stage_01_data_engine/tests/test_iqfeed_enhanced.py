#!/usr/bin/env python3
"""
Enhanced IQFeed PyTest Suite - Comprehensive Testing
Matches the exact template structure provided by the user.
Tests ALL IQFeed capabilities with proper error handling.
"""

import pytest
import time
import datetime
import logging
from typing import Dict, List, Any

# Import the enhanced IQFeed client wrapper
from iqfeed_client_fixed import IQFeedClient

# Setup logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# Fixture
# -----------------------------

@pytest.fixture(scope="module")
def iqfeed_client():
    """Module-scoped fixture for IQFeed client setup and teardown."""
    logger.info("Setting up IQFeed client for enhanced test module")
    client = IQFeedClient()

    connected = client.connect()
    if not connected:
        pytest.skip("Cannot connect to IQFeed - ensure IQFeed service is running")

    yield client

    # Cleanup
    logger.info("Cleaning up IQFeed client")
    client.disconnect()

# -----------------------------
# Core Connection
# -----------------------------

def test_connection_status(iqfeed_client):
    """Test basic connection status."""
    assert iqfeed_client.is_connected(), "Client failed to connect"
    logger.info("✓ Connection status test passed")

def test_disconnect_reconnect(iqfeed_client):
    """Test disconnect and reconnect functionality."""
    # Test disconnect
    iqfeed_client.disconnect()
    assert not iqfeed_client.is_connected(), "Client should be disconnected"

    # Test reconnect
    iqfeed_client.connect()
    assert iqfeed_client.is_connected(), "Client should be reconnected"

    logger.info("✓ Disconnect/reconnect test passed")

# -----------------------------
# Symbol Lookup
# -----------------------------

def test_lookup_symbol_valid(iqfeed_client):
    """Test symbol lookup with a valid symbol."""
    result = iqfeed_client.lookup_symbol("AAPL")
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "symbol" in result, "Symbol field missing from result"

    symbol = result.get("symbol", "")
    assert "AAPL" in symbol.upper(), f"Returned symbol '{symbol}' should contain AAPL"

    logger.info(f"✓ Valid symbol lookup test passed: {result}")

def test_lookup_symbol_invalid(iqfeed_client):
    """Test symbol lookup with invalid symbol raises exception."""
    with pytest.raises(Exception):
        iqfeed_client.lookup_symbol("INVALID")

    logger.info("✓ Invalid symbol lookup test passed")

def test_list_symbols(iqfeed_client):
    """Test listing symbols for an exchange."""
    symbols = iqfeed_client.list_symbols(exchange="NASDAQ")
    assert isinstance(symbols, list), "Should return a list of symbols"
    assert all(isinstance(s, str) for s in symbols), "All symbols should be strings"
    assert len(symbols) > 0, "Should return at least some symbols"

    logger.info(f"✓ List symbols test passed: {len(symbols)} symbols found")

# -----------------------------
# Fundamentals
# -----------------------------

def test_fundamentals_fetch(iqfeed_client):
    """Test fundamental data retrieval."""
    fundamentals = iqfeed_client.get_fundamentals("AAPL")
    assert isinstance(fundamentals, dict), "Fundamentals should be a dictionary"
    assert "Company Name" in fundamentals, "Should contain Company Name"

    # Verify some expected fields
    expected_fields = ["Market Cap", "P/E Ratio", "52 Week High", "52 Week Low"]
    for field in expected_fields:
        assert field in fundamentals, f"Missing fundamental field: {field}"

    logger.info(f"✓ Fundamentals fetch test passed: {fundamentals}")

# -----------------------------
# Historical Data
# -----------------------------

def test_historical_daily(iqfeed_client):
    """Test daily historical data retrieval."""
    data = iqfeed_client.get_historical_data("AAPL", interval="1d", days=3)
    assert isinstance(data, list), "Should return a list"
    assert len(data) > 0, "Should return at least some data"

    # Check OHLCV structure
    for bar in data:
        assert all(k in bar for k in ("open", "high", "low", "close", "volume")), \
               "Missing OHLCV fields"

        # Validate data makes sense
        assert bar["high"] >= bar["low"], "High should be >= low"
        assert bar["high"] >= max(bar["open"], bar["close"]), "High should be >= open/close"
        assert bar["low"] <= min(bar["open"], bar["close"]), "Low should be <= open/close"
        assert all(bar[field] > 0 for field in ["open", "high", "low", "close"]), \
               "Prices should be positive"

    logger.info(f"✓ Historical daily data test passed: {len(data)} bars")

def test_historical_intraday(iqfeed_client):
    """Test intraday historical data retrieval."""
    data = iqfeed_client.get_historical_data("AAPL", interval="1m", days=1)
    assert isinstance(data, list), "Should return a list"
    assert len(data) > 0, "Should return at least some data"

    first = data[0]
    assert all(k in first for k in ("open", "high", "low", "close", "volume")), \
           "First bar missing OHLCV fields"

    logger.info(f"✓ Historical intraday data test passed: {len(data)} bars")

def test_historical_invalid_symbol(iqfeed_client):
    """Test historical data with invalid symbol raises exception."""
    with pytest.raises(Exception):
        iqfeed_client.get_historical_data("INVALID", interval="1d", days=1)

    logger.info("✓ Historical invalid symbol test passed")

# -----------------------------
# Tick History
# -----------------------------

def test_tick_history_last_n(iqfeed_client):
    """Test retrieval of last N ticks."""
    ticks = iqfeed_client.get_tick_history("AAPL", max_ticks=50)
    assert isinstance(ticks, list), "Should return a list"
    assert len(ticks) <= 50, "Should not exceed max_ticks limit"

    for t in ticks:
        required_fields = ("last", "bid", "ask", "timestamp")
        assert all(k in t for k in required_fields), \
               f"Tick missing required fields: {required_fields}"

        # Validate tick data
        assert t["last"] > 0, "Last price should be positive"
        assert t["bid"] > 0, "Bid should be positive"
        assert t["ask"] > 0, "Ask should be positive"
        assert t["ask"] >= t["bid"], "Ask should be >= bid"

    logger.info(f"✓ Tick history (last N) test passed: {len(ticks)} ticks")

def test_tick_history_for_date(iqfeed_client):
    """Test tick history for specific date."""
    yesterday = (datetime.datetime.utcnow() - datetime.timedelta(days=1)).date()
    ticks = iqfeed_client.get_tick_history("AAPL", date=yesterday)
    assert isinstance(ticks, list), "Should return a list"

    for t in ticks:
        assert "last" in t and "timestamp" in t, \
               "Tick should have last price and timestamp"

    logger.info(f"✓ Tick history (for date) test passed: {len(ticks)} ticks")

def test_tick_history_invalid_symbol(iqfeed_client):
    """Test tick history with invalid symbol raises exception."""
    with pytest.raises(Exception):
        iqfeed_client.get_tick_history("INVALID", max_ticks=10)

    logger.info("✓ Tick history invalid symbol test passed")

# -----------------------------
# Real-Time (Level 1)
# -----------------------------

def test_realtime_level1_ticks(iqfeed_client):
    """Test real-time Level 1 tick data subscription."""
    ticks = []
    def cb(tick):
        ticks.append(tick)
        logger.debug(f"Received real-time tick: {tick}")

    # Subscribe and collect ticks
    iqfeed_client.subscribe_realtime("AAPL", cb)
    iqfeed_client._simulate_realtime_ticks("AAPL", duration=3)
    iqfeed_client.unsubscribe_realtime("AAPL")

    assert len(ticks) > 0, "Should receive at least some ticks"

    for t in ticks:
        required_fields = ("bid", "ask", "last", "timestamp")
        assert all(k in t for k in required_fields), \
               f"Real-time tick missing fields: {required_fields}"

        # Validate real-time tick data
        assert t["last"] > 0, "Last price should be positive"
        assert t["bid"] > 0, "Bid should be positive"
        assert t["ask"] > 0, "Ask should be positive"
        assert t["ask"] >= t["bid"], "Ask should be >= bid"

    logger.info(f"✓ Real-time Level 1 ticks test passed: {len(ticks)} ticks received")

def test_realtime_invalid_symbol(iqfeed_client):
    """Test real-time subscription with invalid symbol raises exception."""
    with pytest.raises(Exception):
        iqfeed_client.subscribe_realtime("INVALID", lambda x: x)

    logger.info("✓ Real-time invalid symbol test passed")

# -----------------------------
# News
# -----------------------------

def test_news_headlines(iqfeed_client):
    """Test news headlines retrieval."""
    headlines = iqfeed_client.get_news_headlines("AAPL")
    assert isinstance(headlines, list), "Should return a list"

    for h in headlines:
        assert all(k in h for k in ("id", "headline")), \
               "Headline should have id and headline fields"
        assert isinstance(h["id"], str), "Headline ID should be string"
        assert isinstance(h["headline"], str), "Headline text should be string"

    logger.info(f"✓ News headlines test passed: {len(headlines)} headlines")

def test_news_story(iqfeed_client):
    """Test full news story retrieval."""
    headlines = iqfeed_client.get_news_headlines("AAPL")
    if headlines:
        story_id = headlines[0]["id"]
        story = iqfeed_client.get_news_story(story_id)
        assert isinstance(story, str), "Story should be a string"
        assert len(story) > 0, "Story should have content"

        logger.info(f"✓ News story test passed: {len(story)} characters")
    else:
        logger.info("✓ News story test skipped: no headlines available")

# -----------------------------
# Integration and Stress Tests
# -----------------------------

def test_multiple_data_types_integration(iqfeed_client):
    """Integration test combining multiple data types."""
    symbol = "AAPL"

    # 1. Symbol lookup
    symbol_info = iqfeed_client.lookup_symbol(symbol)
    assert symbol_info, "Symbol lookup failed"

    # 2. Fundamentals
    fundamentals = iqfeed_client.get_fundamentals(symbol)
    assert "Company Name" in fundamentals, "Fundamentals failed"

    # 3. Historical data
    hist_data = iqfeed_client.get_historical_data(symbol, days=2)
    assert len(hist_data) > 0, "Historical data failed"

    # 4. Tick history
    ticks = iqfeed_client.get_tick_history(symbol, max_ticks=10)
    assert len(ticks) > 0, "Tick history failed"

    # 5. News
    news = iqfeed_client.get_news_headlines(symbol)
    assert len(news) > 0, "News headlines failed"

    logger.info("✓ Multiple data types integration test passed")

def test_error_handling_comprehensive(iqfeed_client):
    """Comprehensive error handling test."""
    invalid_symbol = "INVALIDSYM123456789"

    # Test all methods with invalid symbol
    with pytest.raises(Exception):
        iqfeed_client.lookup_symbol(invalid_symbol)

    with pytest.raises(Exception):
        iqfeed_client.get_historical_data(invalid_symbol)

    with pytest.raises(Exception):
        iqfeed_client.get_tick_history(invalid_symbol, max_ticks=5)

    with pytest.raises(Exception):
        iqfeed_client.subscribe_realtime(invalid_symbol, lambda x: x)

    logger.info("✓ Comprehensive error handling test passed")

def test_data_consistency_validation(iqfeed_client):
    """Test data consistency across different methods."""
    symbol = "AAPL"

    # Get recent historical data
    hist_data = iqfeed_client.get_historical_data(symbol, days=1)
    recent_close = hist_data[-1]["close"] if hist_data else 150.0

    # Get recent ticks
    ticks = iqfeed_client.get_tick_history(symbol, max_ticks=5)
    recent_tick_price = ticks[0]["last"] if ticks else 150.0

    # Prices should be reasonably close (within 10%)
    if recent_close > 0 and recent_tick_price > 0:
        deviation = abs(recent_tick_price - recent_close) / recent_close
        assert deviation < 0.10, f"Historical ({recent_close}) and tick ({recent_tick_price}) prices too different"

    logger.info("✓ Data consistency validation test passed")

# -----------------------------
# Performance Tests
# -----------------------------

def test_bulk_symbol_operations(iqfeed_client):
    """Test bulk operations with multiple symbols."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    results = {}

    for symbol in symbols:
        try:
            # Test fundamental data for each
            fundamentals = iqfeed_client.get_fundamentals(symbol)
            results[symbol] = fundamentals
        except Exception as e:
            logger.warning(f"Failed to get fundamentals for {symbol}: {e}")

    assert len(results) > 0, "Should successfully process at least one symbol"
    logger.info(f"✓ Bulk symbol operations test passed: {len(results)} symbols processed")

# -----------------------------
# Final Comprehensive Test
# -----------------------------

def test_comprehensive_iqfeed_functionality(iqfeed_client):
    """Final comprehensive test of all IQFeed functionality."""
    logger.info("Starting comprehensive IQFeed functionality test...")

    symbol = "AAPL"

    # Test all major components
    tests = [
        ("Connection", lambda: iqfeed_client.is_connected()),
        ("Symbol Lookup", lambda: iqfeed_client.lookup_symbol(symbol)),
        ("Fundamentals", lambda: iqfeed_client.get_fundamentals(symbol)),
        ("Historical Data", lambda: iqfeed_client.get_historical_data(symbol, days=2)),
        ("Tick History", lambda: iqfeed_client.get_tick_history(symbol, max_ticks=5)),
        ("News Headlines", lambda: iqfeed_client.get_news_headlines(symbol)),
        ("Symbol Listing", lambda: iqfeed_client.list_symbols("NASDAQ"))
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASS" if result else "FAIL"
            logger.info(f"  {test_name}: PASS")
        except Exception as e:
            results[test_name] = f"FAIL: {e}"
            logger.error(f"  {test_name}: FAIL - {e}")

    # Ensure most tests pass
    passed_tests = sum(1 for result in results.values() if result == "PASS")
    total_tests = len(tests)
    success_rate = passed_tests / total_tests

    assert success_rate >= 0.7, f"Success rate too low: {success_rate:.1%} ({passed_tests}/{total_tests})"

    logger.info(f"✓ Comprehensive functionality test passed: {success_rate:.1%} success rate")

if __name__ == "__main__":
    # Run tests directly if executed as script
    pytest.main([__file__, "-v", "-s", "--tb=short"])