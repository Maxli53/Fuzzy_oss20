#!/usr/bin/env python3
"""
IQFeed Constraints Module - Data Availability and Subscription Limitations
Implements intelligent validation and fallback based on actual IQFeed capabilities
"""

import logging
import re
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Union
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from enum import Enum
import time
import threading

logger = logging.getLogger(__name__)

class AssetClass(Enum):
    """Asset class enumeration for different historical data availability."""
    US_STOCKS = "us_stocks"
    FOREX = "forex"
    FUTURES = "futures"
    INDEXES = "indexes"
    OPTIONS = "options"
    LONDON_STOCKS = "london_stocks"
    UNKNOWN = "unknown"

@dataclass
class DataConstraint:
    """Data availability constraint definition."""
    asset_class: AssetClass
    data_type: str  # 'tick', 'intraday', 'daily'
    earliest_date: date
    max_days_tick: int
    max_days_tick_market_hours: int
    description: str

@dataclass
class RequestLimits:
    """Symbol subscription limits - IQFeed handles rate limiting natively."""
    max_simultaneous_symbols: int = 500  # IQFeed symbol subscription limit
    max_bulk_chunk_size: int = 20  # Chunking for bulk operations

class IQFeedConstraints:
    """
    IQFeed data availability constraints with market hours awareness.

    Features:
    - Market hours aware tick data limits (8 days vs 180 days)
    - Asset class specific historical depth validation
    - Rate limiting and subscription management
    - Intelligent fallback suggestions
    """

    def __init__(self, session_manager=None):
        """Initialize constraints with session manager integration."""
        self.session_manager = session_manager
        self.et_tz = ZoneInfo("America/New_York")
        self.utc_tz = ZoneInfo("UTC")

        # Initialize constraints and limits
        self.constraints = self._initialize_constraints()
        self.limits = RequestLimits()

        logger.info("IQFeedConstraints initialized with market hours awareness")

    def _initialize_constraints(self) -> Dict[AssetClass, DataConstraint]:
        """Initialize asset class specific data constraints."""
        return {
            AssetClass.US_STOCKS: DataConstraint(
                asset_class=AssetClass.US_STOCKS,
                data_type="all",
                earliest_date=date(1994, 1, 1),  # US Stocks back to 1994
                max_days_tick=180,
                max_days_tick_market_hours=8,
                description="US Stocks/Futures/Indexes back to May 2007 for detailed data"
            ),
            AssetClass.FOREX: DataConstraint(
                asset_class=AssetClass.FOREX,
                data_type="all",
                earliest_date=date(1993, 1, 1),  # Forex back to 1993
                max_days_tick=180,
                max_days_tick_market_hours=8,
                description="Forex back to February 2005 for detailed data"
            ),
            AssetClass.FUTURES: DataConstraint(
                asset_class=AssetClass.FUTURES,
                data_type="all",
                earliest_date=date(1959, 1, 1),  # US Futures back to 1959
                max_days_tick=180,
                max_days_tick_market_hours=8,
                description="E-minis back to September 2005, US Futures back to 1959"
            ),
            AssetClass.INDEXES: DataConstraint(
                asset_class=AssetClass.INDEXES,
                data_type="all",
                earliest_date=date(1929, 1, 1),  # Indexes back to 1929
                max_days_tick=180,
                max_days_tick_market_hours=8,
                description="Indexes as far back as 1929"
            ),
            AssetClass.LONDON_STOCKS: DataConstraint(
                asset_class=AssetClass.LONDON_STOCKS,
                data_type="all",
                earliest_date=date(1985, 1, 1),  # London stocks back to 1985
                max_days_tick=180,
                max_days_tick_market_hours=8,
                description="London Stocks and FTSE Indexes back to February 2016"
            ),
            AssetClass.OPTIONS: DataConstraint(
                asset_class=AssetClass.OPTIONS,
                data_type="all",
                earliest_date=date(2007, 5, 1),  # Options with stocks
                max_days_tick=180,
                max_days_tick_market_hours=8,
                description="Options data with underlying instrument constraints"
            ),
            AssetClass.UNKNOWN: DataConstraint(
                asset_class=AssetClass.UNKNOWN,
                data_type="all",
                earliest_date=date(2007, 5, 1),  # Conservative default
                max_days_tick=180,
                max_days_tick_market_hours=8,
                description="Unknown asset class - using conservative constraints"
            )
        }

    def detect_asset_class(self, symbol: str) -> AssetClass:
        """
        Detect asset class from symbol format.

        Args:
            symbol: Symbol to classify

        Returns:
            Detected asset class
        """
        symbol = symbol.upper().strip()

        # Forex patterns (currency pairs)
        if re.match(r'^[A-Z]{6}$', symbol) and any(curr in symbol for curr in
                   ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']):
            return AssetClass.FOREX

        # Futures patterns (month codes, @ prefix)
        if re.match(r'^[A-Z]+[FGHJKMNQUVXZ]\d{2}$', symbol) or symbol.startswith('@'):
            return AssetClass.FUTURES

        # Options patterns (OCC format)
        if re.match(r'^[A-Z]+\d{6}[CP]\d{8}$', symbol):
            return AssetClass.OPTIONS

        # London stocks (.L suffix)
        if symbol.endswith('.L'):
            return AssetClass.LONDON_STOCKS

        # Index patterns (common index symbols)
        index_symbols = {'.IXIC', '.DJI', '.SPX', '.RUT', '.VIX', 'SPY', 'QQQ', 'IWM'}
        if symbol in index_symbols or symbol.startswith('$'):
            return AssetClass.INDEXES

        # Default to US stocks for standard symbols
        if re.match(r'^[A-Z]{1,5}$', symbol):
            return AssetClass.US_STOCKS

        return AssetClass.UNKNOWN

    def is_market_hours(self, timestamp: Optional[datetime] = None) -> bool:
        """
        Check if timestamp is during US market hours (9:30am-4:30pm ET).

        Args:
            timestamp: UTC timestamp to check (current time if None)

        Returns:
            True if during market hours
        """
        if timestamp is None:
            timestamp = datetime.now(self.utc_tz)

        # Convert to ET
        et_time = timestamp.astimezone(self.et_tz)

        # Check if weekday
        if et_time.weekday() >= 5:  # Saturday/Sunday
            return False

        # Check if during market hours (9:30am-4:30pm ET)
        market_start = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_end = et_time.replace(hour=16, minute=30, second=0, microsecond=0)

        return market_start <= et_time <= market_end

    def get_tick_data_limit(self, symbol: str, timestamp: Optional[datetime] = None) -> int:
        """
        Get tick data limit for symbol based on market hours.

        Args:
            symbol: Symbol to check
            timestamp: Timestamp for market hours check (current time if None)

        Returns:
            Maximum days of tick data available
        """
        asset_class = self.detect_asset_class(symbol)
        constraint = self.constraints[asset_class]

        if self.is_market_hours(timestamp):
            return constraint.max_days_tick_market_hours  # 8 days during market
        else:
            return constraint.max_days_tick  # 180 days outside market hours

    def validate_tick_request(self, symbol: str, num_days: int,
                             timestamp: Optional[datetime] = None) -> Tuple[bool, int, str]:
        """
        Validate tick data request against limitations.

        Args:
            symbol: Symbol to validate
            num_days: Requested number of days
            timestamp: Timestamp for validation context

        Returns:
            (is_valid, suggested_days, message)
        """
        max_days = self.get_tick_data_limit(symbol, timestamp)

        if num_days <= max_days:
            return True, num_days, f"Request valid for {symbol}"

        # Suggest fallback
        market_context = "during market hours" if self.is_market_hours(timestamp) else "outside market hours"
        message = (f"Requested {num_days} days exceeds {max_days} day limit for {symbol} "
                  f"{market_context}. Fallback: {max_days} days")

        return False, max_days, message

    def validate_historical_depth(self, symbol: str, start_date: Union[str, date]) -> Tuple[bool, date, str]:
        """
        Validate historical data request against asset class limits.

        Args:
            symbol: Symbol to validate
            start_date: Requested start date

        Returns:
            (is_valid, suggested_start_date, message)
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()

        asset_class = self.detect_asset_class(symbol)
        constraint = self.constraints[asset_class]

        if start_date >= constraint.earliest_date:
            return True, start_date, f"Historical request valid for {symbol} ({asset_class.value})"

        # Suggest fallback to earliest available
        message = (f"Requested start date {start_date} predates earliest available data "
                  f"for {symbol} ({asset_class.value}). Fallback: {constraint.earliest_date}")

        return False, constraint.earliest_date, message

    # REMOVED: Subscription tracking - not actually used, just overhead

    # REMOVED: Rate limiting methods - PyIQFeed handles this natively
    # can_make_request() and record_request() deleted

    def get_chunking_strategy(self, symbols: List[str],
                            operation_type: str = "backfill") -> Dict[str, Union[int, List]]:
        """
        Get chunking strategy for bulk operations.

        Args:
            symbols: List of symbols to process
            operation_type: Type of operation ('backfill', 'realtime')

        Returns:
            Chunking strategy configuration
        """
        total_symbols = len(symbols)
        available_slots = self.limits.max_simultaneous_symbols  # Just use the limit directly

        if operation_type == "backfill":
            # Conservative chunking for backfills
            chunk_size = min(self.limits.max_bulk_chunk_size, available_slots, total_symbols)

            return {
                "chunk_size": chunk_size,
                "chunks": [symbols[i:i+chunk_size] for i in range(0, total_symbols, chunk_size)]
            }

        elif operation_type == "realtime":
            # More aggressive chunking for real-time
            chunk_size = min(50, available_slots, total_symbols)

            return {
                "chunk_size": chunk_size,
                "chunks": [symbols[i:i+chunk_size] for i in range(0, total_symbols, chunk_size)]
            }

        return {"error": f"Unknown operation_type: {operation_type}"}

    def get_constraint_summary(self, symbol: str) -> Dict:
        """Get comprehensive constraint summary for a symbol."""
        asset_class = self.detect_asset_class(symbol)
        constraint = self.constraints[asset_class]

        current_tick_limit = self.get_tick_data_limit(symbol)

        return {
            "symbol": symbol,
            "asset_class": asset_class.value,
            "earliest_date": constraint.earliest_date.isoformat(),
            "tick_limit_current": current_tick_limit,
            "tick_limit_market_hours": constraint.max_days_tick_market_hours,
            "tick_limit_off_hours": constraint.max_days_tick,
            "is_market_hours": self.is_market_hours(),
            "max_subscriptions": self.limits.max_simultaneous_symbols
        }