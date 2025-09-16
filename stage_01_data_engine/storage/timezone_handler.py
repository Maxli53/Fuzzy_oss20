"""
Exchange timezone handling for accurate tick data timestamps
All timestamps are maintained in ET (Eastern Time) for US equity markets
"""
import pytz
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TimezoneHandler:
    """Handle exchange timezones correctly for accurate tick data"""

    EXCHANGE_TIMEZONES: Dict[str, str] = {
        'NYSE': 'America/New_York',
        'NASDAQ': 'America/New_York',
        'ARCA': 'America/New_York',
        'AMEX': 'America/New_York',
        'CME': 'America/Chicago',
        'CBOT': 'America/Chicago',
        'NYMEX': 'America/New_York',
        'COMEX': 'America/New_York',
        'ICE': 'America/New_York',
        'CBOE': 'America/Chicago'
    }

    SYMBOL_EXCHANGES: Dict[str, str] = {
        # Major stocks typically trade on NYSE/NASDAQ
        'AAPL': 'NASDAQ',
        'MSFT': 'NASDAQ',
        'GOOGL': 'NASDAQ',
        'AMZN': 'NASDAQ',
        'TSLA': 'NASDAQ',
        'META': 'NASDAQ',
        'NVDA': 'NASDAQ',
        'SPY': 'ARCA',
        'QQQ': 'NASDAQ',
        'IWM': 'ARCA',
        'VIX': 'CBOE'
    }

    def __init__(self):
        self.timezones = {
            name: pytz.timezone(tz_str)
            for name, tz_str in self.EXCHANGE_TIMEZONES.items()
        }
        self.et = pytz.timezone('America/New_York')  # Primary timezone for US markets

    def get_exchange_for_symbol(self, symbol: str) -> str:
        """Get primary exchange for symbol"""
        return self.SYMBOL_EXCHANGES.get(symbol, 'NYSE')  # Default to NYSE

    def normalize_timestamp(self, timestamp: pd.Timestamp,
                          symbol: str,
                          exchange: Optional[str] = None) -> pd.Timestamp:
        """
        Normalize timestamp to exchange local time

        Args:
            timestamp: Input timestamp (may be ET or naive)
            symbol: Stock symbol
            exchange: Override exchange (if known)

        Returns:
            Timestamp in exchange local time
        """
        if exchange is None:
            exchange = self.get_exchange_for_symbol(symbol)

        if exchange not in self.timezones:
            logger.warning(f"Unknown exchange {exchange} for {symbol}, using NYSE")
            exchange = 'NYSE'

        target_tz = self.timezones[exchange]

        try:
            # If timestamp is naive, assume it's already in ET
            if timestamp.tz is None:
                localized_ts = target_tz.localize(timestamp.to_pydatetime())
            else:
                # Convert from source timezone to target
                localized_ts = timestamp.tz_convert(target_tz)

            return pd.Timestamp(localized_ts)

        except Exception as e:
            logger.error(f"Error normalizing timestamp {timestamp} for {symbol}: {e}")
            return timestamp

    def normalize_dataframe(self, df: pd.DataFrame,
                          symbol: str,
                          timestamp_col: str = 'timestamp',
                          exchange: Optional[str] = None) -> pd.DataFrame:
        """
        Normalize all timestamps in DataFrame to exchange local time

        Args:
            df: DataFrame with timestamp column
            symbol: Stock symbol
            timestamp_col: Name of timestamp column
            exchange: Override exchange

        Returns:
            DataFrame with normalized timestamps
        """
        df_copy = df.copy()

        if timestamp_col not in df_copy.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found")
            return df_copy

        try:
            # Vectorized timezone conversion
            df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])

            if exchange is None:
                exchange = self.get_exchange_for_symbol(symbol)

            if exchange not in self.timezones:
                logger.warning(f"Unknown exchange {exchange} for {symbol}, using NYSE")
                exchange = 'NYSE'

            target_tz = self.timezones[exchange]

            # Handle timezone conversion
            if df_copy[timestamp_col].dt.tz is None:
                # Naive timestamps - assume exchange local time
                df_copy[timestamp_col] = df_copy[timestamp_col].dt.tz_localize(target_tz)
            else:
                # Convert to target timezone
                df_copy[timestamp_col] = df_copy[timestamp_col].dt.tz_convert(target_tz)

            logger.info(f"Normalized {len(df_copy)} timestamps for {symbol} to {exchange} timezone")
            return df_copy

        except Exception as e:
            logger.error(f"Error normalizing DataFrame timestamps for {symbol}: {e}")
            return df_copy

    def is_market_hours(self, timestamp: pd.Timestamp, symbol: str) -> bool:
        """
        Check if timestamp is within regular market hours

        Args:
            timestamp: Timestamp in exchange local time
            symbol: Stock symbol

        Returns:
            True if within market hours (9:30 AM - 4:00 PM ET)
        """
        try:
            exchange = self.get_exchange_for_symbol(symbol)

            # Normalize to exchange timezone if needed
            if timestamp.tz is None or timestamp.tz != self.timezones[exchange]:
                timestamp = self.normalize_timestamp(timestamp, symbol, exchange)

            # Check if weekday (Monday=0, Sunday=6)
            if timestamp.weekday() > 4:  # Saturday or Sunday
                return False

            # Market hours: 9:30 AM - 4:00 PM ET
            time_of_day = timestamp.time()
            market_open = pd.Timestamp('9:30:00').time()
            market_close = pd.Timestamp('16:00:00').time()

            return market_open <= time_of_day <= market_close

        except Exception as e:
            logger.error(f"Error checking market hours for {symbol} at {timestamp}: {e}")
            return True  # Default to True if error

    def get_market_sessions(self, symbol: str, date: str) -> Dict[str, pd.Timestamp]:
        """
        Get market session times for a given date

        Args:
            symbol: Stock symbol
            date: Date string (YYYY-MM-DD)

        Returns:
            Dict with 'open', 'close', 'pre_open', 'post_close' times
        """
        exchange = self.get_exchange_for_symbol(symbol)
        target_tz = self.timezones[exchange]

        try:
            base_date = pd.Timestamp(date).tz_localize(target_tz)

            sessions = {
                'pre_open': base_date.replace(hour=4, minute=0, second=0),    # 4:00 AM
                'open': base_date.replace(hour=9, minute=30, second=0),       # 9:30 AM
                'close': base_date.replace(hour=16, minute=0, second=0),      # 4:00 PM
                'post_close': base_date.replace(hour=20, minute=0, second=0)  # 8:00 PM
            }

            return sessions

        except Exception as e:
            logger.error(f"Error getting market sessions for {symbol} on {date}: {e}")
            return {}