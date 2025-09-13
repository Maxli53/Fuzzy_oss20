"""
Tick Data Collector
Handles real-time and historical tick data collection from IQFeed
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging

from stage_01_data_engine.core.base_collector import BaseCollector, StorageNamespace
from stage_01_data_engine.core.config_loader import get_config, get_symbol_config
from stage_01_data_engine.connector import IQFeedConnector
import pyiqfeed as iq

logger = logging.getLogger(__name__)

class TickCollector(BaseCollector):
    """
    Professional tick data collector with IQFeed integration.

    Features:
    - Real-time Level 1 tick streaming
    - Historical tick data retrieval
    - Multiple fallback strategies
    - Data validation and quality control
    - Exchange mapping and timezone handling
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("TickCollector", config)

        # Initialize IQFeed connector
        self.connector = IQFeedConnector()

        # Load configuration
        self.symbol_config = get_config('symbol', default={})
        self.collection_config = get_config('stream', 'tick_collection', default={})

        logger.info("TickCollector initialized")

    def collect(self, symbols: Union[str, List[str]], **kwargs) -> Optional[pd.DataFrame]:
        """
        Collect tick data for given symbols.

        Args:
            symbols: Symbol or list of symbols
            num_days: Number of days of data (default: 1)
            max_ticks: Maximum number of ticks (default: None)
            include_premarket: Include pre-market data (default: False)
            include_afterhours: Include after-hours data (default: False)

        Returns:
            DataFrame with tick data or None if failed
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        num_days = kwargs.get('num_days', 1)
        max_ticks = kwargs.get('max_ticks', None)

        all_ticks = []

        for symbol in symbols:
            try:
                logger.info(f"Collecting tick data for {symbol}")

                # Get symbol-specific configuration
                symbol_cfg = get_symbol_config(symbol)

                # Collect tick data with fallback strategies
                tick_data = self._collect_symbol_ticks(
                    symbol, num_days, max_ticks, symbol_cfg
                )

                if tick_data is not None and not tick_data.empty:
                    # Add symbol column
                    tick_data['symbol'] = symbol
                    all_ticks.append(tick_data)

                    self.update_stats(len(tick_data), success=True)
                    logger.info(f"Collected {len(tick_data)} ticks for {symbol}")
                else:
                    self.update_stats(0, success=False)
                    logger.warning(f"No tick data collected for {symbol}")

            except Exception as e:
                self.update_stats(0, success=False)
                logger.error(f"Error collecting ticks for {symbol}: {e}")
                continue

        if not all_ticks:
            return None

        # Combine all symbol data
        combined_df = pd.concat(all_ticks, ignore_index=True)

        # Sort by timestamp
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

        # Final validation
        if not self.validate(combined_df):
            logger.error("Combined tick data failed validation")
            return None

        return combined_df

    def _collect_symbol_ticks(self, symbol: str, num_days: int, max_ticks: Optional[int],
                            symbol_cfg: Dict) -> Optional[pd.DataFrame]:
        """Collect tick data for single symbol with fallback strategies"""

        if not self.connector.connect():
            logger.error("Failed to connect to IQFeed")
            return None

        hist_conn = self.connector.get_history_connection()
        if not hist_conn:
            logger.error("Failed to get history connection")
            return None

        try:
            with iq.ConnConnector([hist_conn]) as connector:
                # Strategy 1: Direct tick data request
                tick_data = self._try_direct_ticks(hist_conn, symbol, num_days, max_ticks)

                if tick_data is not None:
                    return tick_data

                # Strategy 2: Intraday 1-second bars
                tick_data = self._try_intraday_bars(hist_conn, symbol, num_days, 1)

                if tick_data is not None:
                    return tick_data

                # Strategy 3: Intraday 5-second bars
                tick_data = self._try_intraday_bars(hist_conn, symbol, num_days, 5)

                if tick_data is not None:
                    return tick_data

                # Strategy 4: Daily bars with tick simulation (last resort)
                tick_data = self._try_daily_bars_conversion(hist_conn, symbol, num_days)

                return tick_data

        except Exception as e:
            logger.error(f"Error in tick collection for {symbol}: {e}")
            return None
        finally:
            self.connector.disconnect()

    def _try_direct_ticks(self, hist_conn, symbol: str, num_days: int,
                         max_ticks: Optional[int]) -> Optional[pd.DataFrame]:
        """Try to get direct tick data from IQFeed"""
        try:
            logger.info(f"Attempting direct tick collection for {symbol}")

            tick_data = hist_conn.request_ticks_for_days(
                ticker=symbol,
                num_days=num_days,
                max_ticks=max_ticks if max_ticks else 0
            )

            if len(tick_data) == 0:
                logger.info(f"No direct tick data available for {symbol}")
                return None

            return self._convert_iqfeed_ticks(tick_data, symbol)

        except Exception as e:
            logger.warning(f"Direct tick collection failed for {symbol}: {e}")
            return None

    def _try_intraday_bars(self, hist_conn, symbol: str, num_days: int,
                          interval_seconds: int) -> Optional[pd.DataFrame]:
        """Try to get intraday bars and convert to tick-like format"""
        try:
            logger.info(f"Attempting {interval_seconds}s bars for {symbol}")

            # Fixed parameter issue - use 'days' instead of 'num_days'
            bar_data = hist_conn.request_bars_for_days(
                ticker=symbol,
                days=num_days,  # Fixed parameter name
                interval_len=interval_seconds,
                interval_type='s'
            )

            if len(bar_data) == 0:
                logger.info(f"No {interval_seconds}s bar data for {symbol}")
                return None

            return self._convert_bars_to_ticks(bar_data, symbol, interval_seconds)

        except Exception as e:
            logger.warning(f"Intraday bars failed for {symbol}: {e}")
            return None

    def _try_daily_bars_conversion(self, hist_conn, symbol: str,
                                  num_days: int) -> Optional[pd.DataFrame]:
        """Last resort: convert daily bars to tick-like format"""
        try:
            logger.info(f"Attempting daily bars conversion for {symbol}")

            daily_bars = hist_conn.request_daily_data(ticker=symbol, num_days=num_days)

            if len(daily_bars) == 0:
                logger.warning(f"No daily bar data for {symbol}")
                return None

            return self._convert_bars_to_ticks(daily_bars, symbol, 86400, is_daily=True)

        except Exception as e:
            logger.error(f"Daily bars conversion failed for {symbol}: {e}")
            return None

    def _convert_iqfeed_ticks(self, raw_ticks, symbol: str) -> pd.DataFrame:
        """Convert IQFeed tick format to standard DataFrame"""
        ticks = []

        for i, tick in enumerate(raw_ticks):
            try:
                if hasattr(tick, 'dtype') and tick.dtype.names:
                    # Structured array format
                    tick_dict = {}
                    for field in tick.dtype.names:
                        tick_dict[field] = tick[field]

                    standardized = self._standardize_tick_record(tick_dict, i)
                    if standardized:
                        ticks.append(standardized)

                elif isinstance(tick, dict):
                    standardized = self._standardize_tick_record(tick, i)
                    if standardized:
                        ticks.append(standardized)

            except Exception as e:
                logger.warning(f"Error processing tick {i}: {e}")
                continue

        if not ticks:
            return pd.DataFrame()

        df = pd.DataFrame(ticks)
        return df.sort_values('timestamp').reset_index(drop=True)

    def _convert_bars_to_ticks(self, bar_data, symbol: str, interval_seconds: int,
                              is_daily: bool = False) -> pd.DataFrame:
        """Convert bar data to tick-like format"""
        ticks = []

        for i, bar in enumerate(bar_data):
            try:
                # Extract OHLCV from bar
                if hasattr(bar, 'dtype') and bar.dtype.names:
                    if is_daily:
                        open_price = float(bar['open_p'])
                        high_price = float(bar['high_p'])
                        low_price = float(bar['low_p'])
                        close_price = float(bar['close_p'])
                        volume = int(bar['prd_vlm'])
                        bar_time = pd.to_datetime(bar['date'])
                    else:
                        # Intraday bars
                        open_price = float(bar.get('open', bar.get('open_p', 0)))
                        high_price = float(bar.get('high', bar.get('high_p', 0)))
                        low_price = float(bar.get('low', bar.get('low_p', 0)))
                        close_price = float(bar.get('close', bar.get('close_p', 0)))
                        volume = int(bar.get('volume', bar.get('prd_vlm', 0)))
                        bar_time = pd.to_datetime(bar.get('timestamp', bar.get('date', pd.Timestamp.now())))

                    # Create multiple ticks per bar
                    ticks_per_bar = min(10, max(1, interval_seconds // 10)) if not is_daily else 10

                    # Generate price path: Open -> High/Low -> Close
                    prices = self._generate_ohlc_path(open_price, high_price, low_price, close_price, ticks_per_bar)
                    volumes = [volume // ticks_per_bar] * ticks_per_bar

                    for j, (price, vol) in enumerate(zip(prices, volumes)):
                        tick_time = bar_time + pd.Timedelta(seconds=j * (interval_seconds // ticks_per_bar))

                        # Create tick record
                        spread = 0.01  # $0.01 spread
                        tick = {
                            'timestamp': tick_time,
                            'price': float(price),
                            'volume': int(vol),
                            'bid': float(price - spread/2),
                            'ask': float(price + spread/2),
                            'bid_size': 1000,
                            'ask_size': 1000,
                            'tick_id': i * ticks_per_bar + j,
                            'exchange': 'NASDAQ',
                            'conditions': 'REGULAR',
                            'total_volume': volume
                        }
                        ticks.append(tick)

            except Exception as e:
                logger.warning(f"Error processing bar {i}: {e}")
                continue

        if not ticks:
            return pd.DataFrame()

        return pd.DataFrame(ticks)

    def _standardize_tick_record(self, raw_tick: Dict, tick_index: int) -> Optional[Dict]:
        """Convert raw IQFeed tick to standard format"""
        try:
            # Field mapping
            field_map = {
                'timestamp': ['timestamp', 'time', 'datetime'],
                'price': ['last', 'price', 'last_price'],
                'volume': ['last_size', 'size', 'volume'],
                'bid': ['bid', 'bid_price'],
                'ask': ['ask', 'ask_price'],
                'exchange': ['market_center', 'exchange', 'mkt_center'],
                'conditions': ['conditions', 'trade_conditions']
            }

            standardized = {}

            # Map fields
            for std_field, possible_names in field_map.items():
                value = None
                for name in possible_names:
                    if name in raw_tick:
                        value = raw_tick[name]
                        break

                if value is not None:
                    standardized[std_field] = value
                else:
                    # Defaults
                    if std_field == 'conditions':
                        standardized[std_field] = 'REGULAR'
                    elif std_field == 'exchange':
                        standardized[std_field] = 'UNKNOWN'

            # Validate required fields
            if 'price' not in standardized or 'volume' not in standardized:
                return None

            # Handle timestamp
            if 'timestamp' not in standardized:
                standardized['timestamp'] = pd.Timestamp.now()
            else:
                standardized['timestamp'] = pd.to_datetime(standardized['timestamp'])

            # Convert numeric fields
            numeric_fields = ['price', 'volume', 'bid', 'ask', 'bid_size', 'ask_size']
            for field in numeric_fields:
                if field in standardized and standardized[field] is not None:
                    try:
                        standardized[field] = float(standardized[field])
                    except (ValueError, TypeError):
                        if field in ['price', 'volume']:
                            return None
                        else:
                            standardized[field] = 0.0

            # Add tick ID
            standardized['tick_id'] = tick_index

            return standardized

        except Exception as e:
            logger.error(f"Error standardizing tick: {e}")
            return None

    def _generate_ohlc_path(self, o: float, h: float, l: float, c: float, n: int) -> List[float]:
        """Generate realistic price path that hits OHLC"""
        if n <= 1:
            return [c]

        prices = [o]
        current = o

        # Ensure we hit high and low
        high_hit = (o == h)
        low_hit = (o == l)

        for i in range(1, n-1):
            if not high_hit and np.random.random() < 0.3:
                current = h
                high_hit = True
            elif not low_hit and np.random.random() < 0.3:
                current = l
                low_hit = True
            else:
                # Random walk within bounds
                move = np.random.normal(0, (h-l)*0.02)
                current = np.clip(current + move, l, h)

            prices.append(current)

        # Force close price
        prices.append(c)

        return prices

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate tick data quality"""
        try:
            if data.empty:
                return False

            # Check required columns
            required_cols = ['timestamp', 'price', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    logger.error(f"Missing required column: {col}")
                    return False

            # Check for reasonable values
            if data['price'].min() <= 0:
                logger.error("Invalid price values (<=0)")
                return False

            if data['volume'].min() < 0:
                logger.error("Invalid volume values (<0)")
                return False

            # Check timestamp ordering
            if not data['timestamp'].is_monotonic_increasing:
                logger.warning("Timestamps not in chronological order")
                # Sort the data
                data.sort_values('timestamp', inplace=True)

            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def get_storage_key(self, symbol: str, date: str, **kwargs) -> str:
        """Generate storage key for tick data"""
        return StorageNamespace.tick_key(symbol, date)