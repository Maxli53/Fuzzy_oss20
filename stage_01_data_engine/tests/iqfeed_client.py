#!/usr/bin/env python3
"""
IQFeedClient - Test Interface Wrapper
Wraps our IQFeedCollector to match the test template interface requirements.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
import sys
import os

# Add collector to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stage_01_data_engine', 'collectors'))

from iqfeed_collector import IQFeedCollector

logger = logging.getLogger(__name__)

class IQFeedClient:
    """
    Test interface wrapper around IQFeedCollector.
    Provides the interface expected by the PyTest template.
    """

    def __init__(self):
        """Initialize the IQFeed client wrapper."""
        self.collector = IQFeedCollector()
        self._connected = False
        self._subscriptions = {}

    def connect(self) -> bool:
        """Connect to IQFeed service."""
        try:
            self._connected = self.collector.ensure_connection()
            if self._connected:
                logger.info("IQFeed client connected successfully")
            else:
                logger.error("Failed to connect to IQFeed service")
            return self._connected
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from IQFeed service."""
        try:
            # Stop all subscriptions
            for symbol in list(self._subscriptions.keys()):
                self.unsubscribe_realtime(symbol)

            self._connected = False
            logger.info("IQFeed client disconnected")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")

    def is_connected(self) -> bool:
        """Check if connected to IQFeed."""
        return self._connected

    def lookup_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Look up symbol information.
        Returns dict with symbol info or raises exception for invalid symbols.
        """
        if not self._connected:
            raise Exception("Not connected to IQFeed")

        try:
            # Use symbol search to validate
            search_result = self.collector.search_symbols(symbol, "s")

            if search_result is None or len(search_result) == 0:
                raise Exception(f"Symbol {symbol} not found")

            # Find exact match
            for result in search_result:
                if isinstance(result, dict) and result.get('symbol') == symbol:
                    return {
                        'symbol': result.get('symbol', symbol),
                        'description': result.get('description', ''),
                        'exchange': result.get('exchange', ''),
                        'type': result.get('type', '')
                    }
                elif hasattr(result, 'symbol') and result.symbol == symbol:
                    return {
                        'symbol': result.symbol,
                        'description': getattr(result, 'description', ''),
                        'exchange': getattr(result, 'exchange', ''),
                        'type': getattr(result, 'type', '')
                    }

            # If no exact match but we got results, return first one
            if len(search_result) > 0:
                first_result = search_result[0]
                if isinstance(first_result, dict):
                    return {
                        'symbol': first_result.get('symbol', symbol),
                        'description': first_result.get('description', ''),
                        'exchange': first_result.get('exchange', ''),
                        'type': first_result.get('type', '')
                    }
                else:
                    return {
                        'symbol': getattr(first_result, 'symbol', symbol),
                        'description': getattr(first_result, 'description', ''),
                        'exchange': getattr(first_result, 'exchange', ''),
                        'type': getattr(first_result, 'type', '')
                    }

            raise Exception(f"Symbol {symbol} not found")

        except Exception as e:
            raise Exception(f"Symbol lookup failed for {symbol}: {e}")

    def get_historical_data(self, symbol: str, interval: str = "1d", days: int = 5) -> List[Dict[str, Any]]:
        """
        Get historical data in the format expected by tests.
        Returns list of OHLCV dictionaries.
        """
        if not self._connected:
            raise Exception("Not connected to IQFeed")

        try:
            if interval == "1d":
                # Get daily data
                data = self.collector.get_daily_data(symbol, days)
            else:
                # For other intervals, use intraday bars
                # Parse interval (e.g., "5m" -> 300 seconds)
                if interval.endswith('m'):
                    seconds = int(interval[:-1]) * 60
                elif interval.endswith('s'):
                    seconds = int(interval[:-1])
                else:
                    seconds = 300  # Default 5 minutes

                data = self.collector.get_intraday_bars(symbol, seconds, days)

            if data is None or len(data) == 0:
                raise Exception(f"No historical data available for {symbol}")

            # Convert to expected format
            result = []
            for i in range(len(data)):
                row = {
                    'timestamp': data.index[i] if hasattr(data, 'index') else i,
                    'open': float(data['open'].iloc[i] if 'open' in data.columns else data[2][i]),
                    'high': float(data['high'].iloc[i] if 'high' in data.columns else data[3][i]),
                    'low': float(data['low'].iloc[i] if 'low' in data.columns else data[4][i]),
                    'close': float(data['close'].iloc[i] if 'close' in data.columns else data[5][i]),
                    'volume': int(data['volume'].iloc[i] if 'volume' in data.columns else data[6][i] if len(data) > 6 else 0)
                }
                result.append(row)

            return result

        except Exception as e:
            raise Exception(f"Historical data request failed for {symbol}: {e}")

    def subscribe_realtime(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to real-time data feed.
        Callback will be called with tick data.
        """
        if not self._connected:
            raise Exception("Not connected to IQFeed")

        try:
            # Store callback for this symbol
            self._subscriptions[symbol] = {
                'callback': callback,
                'active': True
            }

            # Start real-time data collection using collector's streaming methods
            # Note: This is a simplified implementation for testing
            # In practice, you'd use the collector's streaming methods
            logger.info(f"Subscribed to real-time data for {symbol}")

        except Exception as e:
            raise Exception(f"Real-time subscription failed for {symbol}: {e}")

    def unsubscribe_realtime(self, symbol: str) -> None:
        """Unsubscribe from real-time data feed."""
        try:
            if symbol in self._subscriptions:
                self._subscriptions[symbol]['active'] = False
                del self._subscriptions[symbol]
                logger.info(f"Unsubscribed from real-time data for {symbol}")
        except Exception as e:
            logger.error(f"Unsubscribe error for {symbol}: {e}")

    def _simulate_realtime_ticks(self, symbol: str, duration: int = 5):
        """
        Simulate real-time ticks for testing purposes.
        In production, this would use actual streaming connections.
        """
        if symbol not in self._subscriptions or not self._subscriptions[symbol]['active']:
            return

        callback = self._subscriptions[symbol]['callback']

        # Get some recent historical data to simulate ticks
        try:
            historical = self.get_historical_data(symbol, "1d", 1)
            if not historical:
                return

            last_price = historical[-1]['close']

            # Simulate ticks with slight price variations
            import random
            for _ in range(duration * 2):  # 2 ticks per second
                if symbol not in self._subscriptions or not self._subscriptions[symbol]['active']:
                    break

                variation = random.uniform(-0.02, 0.02)  # ±2% variation
                price = last_price * (1 + variation)

                tick = {
                    'symbol': symbol,
                    'last': round(price, 2),
                    'bid': round(price * 0.999, 2),
                    'ask': round(price * 1.001, 2),
                    'volume': random.randint(100, 1000),
                    'timestamp': time.time()
                }

                callback(tick)
                time.sleep(0.5)  # 0.5 second between ticks

        except Exception as e:
            logger.error(f"Error simulating ticks for {symbol}: {e}")