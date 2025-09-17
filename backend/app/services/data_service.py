"""
DataService for accessing tick and bar data from ArcticDB.
Provides unified interface for all data access operations.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pytz
import structlog
from arcticdb import Arctic

# Add foundation to path for model access
sys.path.append('C:/Users/maxli/PycharmProjects/PythonProject/Fuzzy_oss20')
from foundation.models.market import TickData, TimeBar, VolumeBar, DollarBar

from app.core.config import settings

logger = structlog.get_logger()


class DataService:
    """
    Service for accessing tick and bar data from ArcticDB.
    Handles all data retrieval operations with proper error handling.
    """

    def __init__(self):
        self.et_tz = pytz.timezone(settings.TIMEZONE)
        self.arctic_store = None
        self.libraries = {}
        self._initialize_arctic()

    def _initialize_arctic(self):
        """Initialize ArcticDB connection and libraries."""
        try:
            self.arctic_store = Arctic(settings.ARCTIC_URI)

            # Initialize libraries
            library_names = [
                settings.ARCTIC_TICK_LIBRARY,
                settings.ARCTIC_BAR_LIBRARY,
                "bars_tick_bars",
                "bars_volume_bars",
                "bars_dollar_bars",
                "bars_renko",
                "bars_range",
                "metadata_tier1",
                "metadata_tier2",
                "metadata_tier3"
            ]

            for lib_name in library_names:
                try:
                    self.libraries[lib_name] = self.arctic_store[lib_name]
                except:
                    # Create library if it doesn't exist
                    self.libraries[lib_name] = self.arctic_store.create_library(lib_name)

            logger.info("ArcticDB initialized successfully", libraries=list(self.libraries.keys()))

        except Exception as e:
            logger.error("Failed to initialize ArcticDB", error=str(e))
            raise

    async def get_tick_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        asset_class: str = "equity"
    ) -> pd.DataFrame:
        """
        Retrieve tick data for a symbol within a time range.

        Args:
            symbol: Stock symbol
            start_time: Start time (will be converted to ET)
            end_time: End time (will be converted to ET)
            asset_class: Asset class (default: equity)

        Returns:
            DataFrame with tick data
        """
        try:
            # Ensure ET timezone
            if start_time.tzinfo is None:
                start_time = self.et_tz.localize(start_time)
            else:
                start_time = start_time.astimezone(self.et_tz)

            if end_time.tzinfo is None:
                end_time = self.et_tz.localize(end_time)
            else:
                end_time = end_time.astimezone(self.et_tz)

            # Build storage keys for date range
            current_date = start_time.date()
            end_date = end_time.date()

            all_data = []

            while current_date <= end_date:
                key = f"{asset_class}/{symbol}/{current_date.strftime('%Y-%m-%d')}"

                try:
                    # Read data from ArcticDB
                    df = self.libraries[settings.ARCTIC_TICK_LIBRARY].read(key).data

                    # Filter by time range
                    if not df.empty:
                        df = df[(df.index >= start_time) & (df.index <= end_time)]
                        all_data.append(df)

                except Exception as e:
                    logger.debug(f"No data for {key}: {str(e)}")

                current_date += timedelta(days=1)

            if all_data:
                result_df = pd.concat(all_data)
                logger.info(
                    "Retrieved tick data",
                    symbol=symbol,
                    rows=len(result_df),
                    start=start_time.isoformat(),
                    end=end_time.isoformat()
                )
                return result_df
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error("Failed to retrieve tick data", error=str(e), symbol=symbol)
            raise

    async def get_bar_data(
        self,
        symbol: str,
        bar_type: str,
        interval: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Retrieve bar data for a symbol.

        Args:
            symbol: Stock symbol
            bar_type: Type of bar (time, tick, volume, dollar, renko, range)
            interval: Bar interval (1m, 5m, 15m, etc.)
            start_time: Start time
            end_time: End time

        Returns:
            DataFrame with bar data
        """
        try:
            # Map bar type to library
            library_map = {
                "time": "bars_time_bars",
                "tick": "bars_tick_bars",
                "volume": "bars_volume_bars",
                "dollar": "bars_dollar_bars",
                "renko": "bars_renko",
                "range": "bars_range"
            }

            library_name = library_map.get(bar_type)
            if not library_name:
                raise ValueError(f"Invalid bar type: {bar_type}")

            # Ensure ET timezone
            if start_time.tzinfo is None:
                start_time = self.et_tz.localize(start_time)
            else:
                start_time = start_time.astimezone(self.et_tz)

            if end_time.tzinfo is None:
                end_time = self.et_tz.localize(end_time)
            else:
                end_time = end_time.astimezone(self.et_tz)

            # Build storage keys
            current_date = start_time.date()
            end_date = end_time.date()

            all_data = []

            while current_date <= end_date:
                key = f"{symbol}/{interval}/{current_date.strftime('%Y-%m-%d')}"

                try:
                    df = self.libraries[library_name].read(key).data

                    if not df.empty:
                        df = df[(df.index >= start_time) & (df.index <= end_time)]
                        all_data.append(df)

                except Exception as e:
                    logger.debug(f"No data for {key}: {str(e)}")

                current_date += timedelta(days=1)

            if all_data:
                result_df = pd.concat(all_data)
                logger.info(
                    "Retrieved bar data",
                    symbol=symbol,
                    bar_type=bar_type,
                    interval=interval,
                    rows=len(result_df)
                )
                return result_df
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error("Failed to retrieve bar data", error=str(e), symbol=symbol)
            raise

    async def get_latest_tick(
        self,
        symbol: str,
        asset_class: str = "equity"
    ) -> Optional[Dict[str, Any]]:
        """
        Get the most recent tick for a symbol.

        Args:
            symbol: Stock symbol
            asset_class: Asset class

        Returns:
            Dictionary with latest tick data or None
        """
        try:
            # Get today's date in ET
            today = datetime.now(self.et_tz).date()
            key = f"{asset_class}/{symbol}/{today.strftime('%Y-%m-%d')}"

            # Try to read today's data
            try:
                df = self.libraries[settings.ARCTIC_TICK_LIBRARY].read(key).data
                if not df.empty:
                    latest = df.iloc[-1]
                    return {
                        "symbol": symbol,
                        "timestamp": latest.name.isoformat(),
                        "price": float(latest.get("price", 0)),
                        "size": int(latest.get("size", 0)),
                        "bid": float(latest.get("bid", 0)),
                        "ask": float(latest.get("ask", 0)),
                        "exchange": latest.get("exchange", ""),
                        "conditions": latest.get("conditions", "")
                    }
            except:
                # If today's data not available, try yesterday
                yesterday = today - timedelta(days=1)
                key = f"{asset_class}/{symbol}/{yesterday.strftime('%Y-%m-%d')}"
                df = self.libraries[settings.ARCTIC_TICK_LIBRARY].read(key).data
                if not df.empty:
                    latest = df.iloc[-1]
                    return {
                        "symbol": symbol,
                        "timestamp": latest.name.isoformat(),
                        "price": float(latest.get("price", 0)),
                        "size": int(latest.get("size", 0)),
                        "bid": float(latest.get("bid", 0)),
                        "ask": float(latest.get("ask", 0)),
                        "exchange": latest.get("exchange", ""),
                        "conditions": latest.get("conditions", "")
                    }

            return None

        except Exception as e:
            logger.error("Failed to get latest tick", error=str(e), symbol=symbol)
            return None

    async def get_metadata(
        self,
        symbol: str,
        tier: int,
        date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a symbol at specified tier.

        Args:
            symbol: Stock symbol
            tier: Metadata tier (1, 2, or 3)
            date: Date for metadata (default: today)

        Returns:
            Dictionary with metadata or None
        """
        try:
            if tier not in [1, 2, 3]:
                raise ValueError(f"Invalid tier: {tier}")

            if date is None:
                date = datetime.now(self.et_tz)
            elif date.tzinfo is None:
                date = self.et_tz.localize(date)
            else:
                date = date.astimezone(self.et_tz)

            # Build metadata key
            key = f"{symbol}/1m/{date.strftime('%Y-%m-%d')}/tier{tier}"
            library_name = f"metadata_tier{tier}"

            try:
                metadata = self.libraries[library_name].read(key).data
                return metadata.to_dict() if hasattr(metadata, 'to_dict') else dict(metadata)
            except:
                logger.debug(f"No metadata for {key}")
                return None

        except Exception as e:
            logger.error("Failed to get metadata", error=str(e), symbol=symbol, tier=tier)
            return None

    async def get_available_symbols(self) -> List[str]:
        """
        Get list of all available symbols in the database.

        Returns:
            List of symbol strings
        """
        try:
            symbols = set()

            # Check tick data library for all symbols
            all_keys = self.libraries[settings.ARCTIC_TICK_LIBRARY].list_symbols()

            for key in all_keys:
                # Parse key format: asset_class/symbol/date
                parts = key.split("/")
                if len(parts) >= 2:
                    symbols.add(parts[1])

            return sorted(list(symbols))

        except Exception as e:
            logger.error("Failed to get available symbols", error=str(e))
            return []

    async def get_data_summary(
        self,
        symbol: str,
        lookback_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get summary statistics for a symbol's data.

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to analyze

        Returns:
            Dictionary with summary statistics
        """
        try:
            end_time = datetime.now(self.et_tz)
            start_time = end_time - timedelta(days=lookback_days)

            # Get tick data for analysis
            tick_data = await self.get_tick_data(symbol, start_time, end_time)

            if tick_data.empty:
                return {
                    "symbol": symbol,
                    "status": "no_data",
                    "message": f"No data found for {symbol} in last {lookback_days} days"
                }

            # Calculate summary statistics
            summary = {
                "symbol": symbol,
                "date_range": {
                    "start": tick_data.index.min().isoformat(),
                    "end": tick_data.index.max().isoformat(),
                    "days": lookback_days
                },
                "tick_statistics": {
                    "total_ticks": len(tick_data),
                    "daily_average": len(tick_data) / lookback_days,
                    "unique_days": tick_data.index.date.nunique()
                },
                "price_statistics": {
                    "min": float(tick_data["price"].min()),
                    "max": float(tick_data["price"].max()),
                    "mean": float(tick_data["price"].mean()),
                    "std": float(tick_data["price"].std()),
                    "current": float(tick_data["price"].iloc[-1])
                },
                "volume_statistics": {
                    "total_volume": int(tick_data["size"].sum()),
                    "daily_average": int(tick_data["size"].sum() / lookback_days),
                    "mean_trade_size": float(tick_data["size"].mean())
                },
                "spread_statistics": {}
            }

            # Calculate spread statistics if bid/ask available
            if "bid" in tick_data.columns and "ask" in tick_data.columns:
                spreads = tick_data["ask"] - tick_data["bid"]
                spread_bps = (spreads / tick_data["price"]) * 10000

                summary["spread_statistics"] = {
                    "mean_spread": float(spreads.mean()),
                    "mean_spread_bps": float(spread_bps.mean()),
                    "max_spread": float(spreads.max()),
                    "min_spread": float(spreads.min())
                }

            return summary

        except Exception as e:
            logger.error("Failed to get data summary", error=str(e), symbol=symbol)
            return {
                "symbol": symbol,
                "status": "error",
                "message": str(e)
            }

    async def search_ticks(
        self,
        symbol: str,
        filters: Dict[str, Any],
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Search tick data with filters.

        Args:
            symbol: Stock symbol
            filters: Dictionary of filters (price_min, price_max, size_min, etc.)
            limit: Maximum number of results

        Returns:
            DataFrame with filtered tick data
        """
        try:
            # Get date range from filters
            start_time = filters.get("start_time", datetime.now(self.et_tz) - timedelta(days=1))
            end_time = filters.get("end_time", datetime.now(self.et_tz))

            # Get tick data
            df = await self.get_tick_data(symbol, start_time, end_time)

            if df.empty:
                return df

            # Apply filters
            if "price_min" in filters:
                df = df[df["price"] >= filters["price_min"]]
            if "price_max" in filters:
                df = df[df["price"] <= filters["price_max"]]
            if "size_min" in filters:
                df = df[df["size"] >= filters["size_min"]]
            if "size_max" in filters:
                df = df[df["size"] <= filters["size_max"]]
            if "exchange" in filters:
                df = df[df["exchange"] == filters["exchange"]]

            # Apply limit
            if len(df) > limit:
                df = df.head(limit)

            logger.info(
                "Searched tick data",
                symbol=symbol,
                filters=filters,
                results=len(df)
            )

            return df

        except Exception as e:
            logger.error("Failed to search ticks", error=str(e), symbol=symbol)
            raise