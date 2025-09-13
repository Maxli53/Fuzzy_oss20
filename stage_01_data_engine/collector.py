"""
Data Collector - Retrieves historical data from IQFeed
"""
import pyiqfeed as iq
from datetime import datetime, date
from typing import List, Dict, Optional
import logging
import pandas as pd
import numpy as np
from .connector import IQFeedConnector

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects historical market data from IQFeed"""

    def __init__(self):
        self.connector = IQFeedConnector()

    def get_daily_bars(self, symbol: str, num_days: int = 22) -> Optional[List[Dict]]:
        """
        Get daily OHLCV bars for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            num_days: Number of trading days to retrieve

        Returns:
            List of daily bar data dictionaries or None if failed
        """
        try:
            # Connect to IQFeed
            if not self.connector.connect():
                logger.error("Failed to connect to IQFeed")
                return None

            # Get history connection
            hist_conn = self.connector.get_history_connection()
            if not hist_conn:
                logger.error("Failed to get history connection")
                return None

            logger.info(f"Requesting {num_days} days of daily data for {symbol}...")

            # Request historical data using pyiqfeed's method
            # HDT = Historical Daily Tick data request
            bars = []

            with iq.ConnConnector([hist_conn]) as connector:
                try:
                    # Request daily bars using proper pyiqfeed method
                    # Use request_daily_data for daily bars
                    daily_data = hist_conn.request_daily_data(
                        ticker=symbol,
                        num_days=num_days
                    )

                    logger.info(f"Received {len(daily_data)} daily bars for {symbol}")

                    # Debug: Check the structure of returned data
                    if len(daily_data) > 0:
                        logger.info(f"First bar type: {type(daily_data[0])}")
                        logger.info(f"First bar fields: {daily_data[0].dtype.names if hasattr(daily_data[0], 'dtype') else 'No dtype'}")

                    # Convert to our format - pyiqfeed returns numpy structured arrays
                    for bar in daily_data:
                        # Convert numpy datetime64 to Python date
                        date_value = pd.to_datetime(bar['date']).date()

                        bar_dict = {
                            'date': date_value,
                            'open': float(bar['open_p']),    # _p suffix for price fields
                            'high': float(bar['high_p']),
                            'low': float(bar['low_p']),
                            'close': float(bar['close_p']),
                            'volume': int(bar['prd_vlm'])    # prd_vlm for volume
                        }
                        bars.append(bar_dict)

                    # Sort by date (oldest first)
                    bars.sort(key=lambda x: x['date'])

                    return bars

                except Exception as e:
                    logger.error(f"Error requesting bars: {e}")
                    return None

        except Exception as e:
            logger.error(f"Error in get_daily_bars: {e}")
            return None
        finally:
            self.connector.disconnect()

    def get_closing_prices(self, symbol: str, num_days: int = 22) -> Optional[List[Dict]]:
        """
        Get just the closing prices and dates for a symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            num_days: Number of trading days to retrieve

        Returns:
            List of {date, close} dictionaries or None if failed
        """
        bars = self.get_daily_bars(symbol, num_days)
        if not bars:
            return None

        # Extract just date and close price
        closing_prices = []
        for bar in bars:
            closing_prices.append({
                'date': bar['date'],
                'close': bar['close']
            })

        return closing_prices

    def display_closing_prices(self, symbol: str, num_days: int = 22) -> bool:
        """
        Display closing prices in a formatted way

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            num_days: Number of trading days to retrieve

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Fetching {num_days} daily closing prices for {symbol}...")

        closing_prices = self.get_closing_prices(symbol, num_days)
        if not closing_prices:
            logger.error(f"Failed to retrieve closing prices for {symbol}")
            return False

        print(f"\n{symbol} Daily Closing Prices (Last {len(closing_prices)} Trading Days):")
        print("=" * 50)

        for price_data in closing_prices:
            date_str = price_data['date'].strftime('%Y-%m-%d')
            price = price_data['close']
            print(f"{date_str}: ${price:.2f}")

        print("=" * 50)
        print(f"Successfully retrieved {len(closing_prices)} closing prices")

        return True