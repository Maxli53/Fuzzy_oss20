#!/usr/bin/env python3
"""
Market Session Manager - Industry-Standard Implementation
Handles market sessions, holidays, and timezone conversions following industry best practices
"""

import logging
import pandas as pd
from datetime import datetime, time, date
from typing import Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo
import pandas_market_calendars as mcal

logger = logging.getLogger(__name__)

class MarketSessionManager:
    """
    Industry-standard market session management.

    Features:
    - Separate handling for premarket/regular/afterhours sessions
    - Multi-exchange holiday calendar support
    - UTC-standard timezone handling with zoneinfo
    - Session-aware data segmentation
    """

    # Industry-standard session times (Eastern Time)
    SESSIONS = {
        'premarket': {'start': time(4, 0), 'end': time(9, 30)},
        'regular': {'start': time(9, 30), 'end': time(16, 0)},
        'afterhours': {'start': time(16, 0), 'end': time(20, 0)}
    }

    # Early close times (Eastern Time)
    EARLY_CLOSE_TIMES = {
        'equity': time(13, 0),      # 1:00 PM ET
        'options': time(13, 15)     # 1:15 PM ET
    }

    def __init__(self):
        """Initialize market session manager with industry-standard calendars."""
        self.calendars = self._initialize_calendars()
        self.et_tz = ZoneInfo("America/New_York")
        self.utc_tz = ZoneInfo("UTC")
        logger.info("MarketSessionManager initialized with industry-standard calendars")

    def _initialize_calendars(self) -> Dict:
        """Initialize industry-standard market calendars."""
        calendars = {}

        try:
            # Primary equity markets
            calendars['NYSE'] = mcal.get_calendar('NYSE')
            calendars['NASDAQ'] = mcal.get_calendar('NASDAQ')

            # Options markets (CBOE has different names)
            try:
                calendars['CBOE'] = mcal.get_calendar('CBOE_Equity_Options')
            except:
                # Fallback if CBOE_Equity_Options doesn't exist
                pass

            logger.info(f"Initialized calendars for: {list(calendars.keys())}")

        except Exception as e:
            logger.error(f"Failed to initialize some calendars: {e}")
            # Fallback to NYSE only
            calendars['NYSE'] = mcal.get_calendar('NYSE')

        return calendars

    def get_current_session(self, timestamp: Optional[datetime] = None,
                           exchange: str = 'NYSE') -> Optional[str]:
        """
        Get current market session for a timestamp.

        Args:
            timestamp: Timestamp to check (UTC). If None, uses current time
            exchange: Exchange to check ('NYSE', 'NASDAQ', 'CBOE')

        Returns:
            Session name ('premarket', 'regular', 'afterhours') or None if market closed
        """
        if timestamp is None:
            timestamp = datetime.now(self.utc_tz)

        # Convert UTC timestamp to Eastern Time
        et_time = timestamp.astimezone(self.et_tz)

        # Check if it's a trading day
        if not self.is_trading_day(et_time.date(), exchange):
            return None

        # Check if it's an early close day
        early_close = self.get_early_close_time(et_time.date(), exchange, 'equity')
        if early_close:
            # Adjust regular session end time for early close
            sessions = self.SESSIONS.copy()
            sessions['regular']['end'] = early_close
            sessions['afterhours']['start'] = early_close
        else:
            sessions = self.SESSIONS

        # Determine current session
        current_time = et_time.time()

        for session_name, session_times in sessions.items():
            if session_times['start'] <= current_time < session_times['end']:
                return session_name

        return None  # Outside trading hours

    def is_trading_day(self, check_date: Union[date, str],
                      exchange: str = 'NYSE') -> bool:
        """
        Check if a date is a trading day for the specified exchange.

        Args:
            check_date: Date to check (date object or YYYY-MM-DD string)
            exchange: Exchange to check

        Returns:
            True if trading day, False if holiday/weekend
        """
        if isinstance(check_date, str):
            check_date = datetime.strptime(check_date, '%Y-%m-%d').date()

        calendar = self.calendars.get(exchange, self.calendars['NYSE'])

        try:
            # Check if date is in trading schedule
            schedule = calendar.schedule(start_date=check_date, end_date=check_date)
            return len(schedule) > 0

        except Exception as e:
            logger.warning(f"Error checking trading day for {check_date}: {e}")
            # Fallback: weekday check (not ideal but better than crash)
            return check_date.weekday() < 5

    def get_early_close_time(self, check_date: Union[date, str],
                           exchange: str = 'NYSE',
                           market_type: str = 'equity') -> Optional[time]:
        """
        Get early close time if the date has early close.

        Args:
            check_date: Date to check
            exchange: Exchange to check
            market_type: 'equity' or 'options'

        Returns:
            Early close time or None if regular hours
        """
        if isinstance(check_date, str):
            check_date = datetime.strptime(check_date, '%Y-%m-%d').date()

        calendar = self.calendars.get(exchange, self.calendars['NYSE'])

        try:
            # Get trading schedule for the date
            schedule = calendar.schedule(start_date=check_date, end_date=check_date)

            if len(schedule) == 0:
                return None  # Not a trading day

            # Check if market closes early
            market_close = schedule.iloc[0]['market_close']
            market_close_et = market_close.astimezone(self.et_tz).time()

            # Compare with regular close time
            regular_close = self.SESSIONS['regular']['end']

            if market_close_et < regular_close:
                # Early close detected - return appropriate time
                return self.EARLY_CLOSE_TIMES.get(market_type, market_close_et)

        except Exception as e:
            logger.warning(f"Error checking early close for {check_date}: {e}")

        return None  # Regular hours

    def get_session_boundaries(self, trading_date: Union[date, str],
                              exchange: str = 'NYSE') -> Dict[str, datetime]:
        """
        Get session start/end times for a trading date in UTC.

        Args:
            trading_date: Trading date to get boundaries for
            exchange: Exchange calendar to use

        Returns:
            Dictionary with session boundaries in UTC
        """
        if isinstance(trading_date, str):
            trading_date = datetime.strptime(trading_date, '%Y-%m-%d').date()

        if not self.is_trading_day(trading_date, exchange):
            return {}

        # Build session boundaries in Eastern Time
        boundaries = {}

        # Check for early close
        early_close = self.get_early_close_time(trading_date, exchange, 'equity')
        sessions = self.SESSIONS.copy()

        if early_close:
            sessions['regular']['end'] = early_close
            sessions['afterhours']['start'] = early_close

        # Convert to UTC
        for session_name, session_times in sessions.items():
            start_et = datetime.combine(trading_date, session_times['start'])
            end_et = datetime.combine(trading_date, session_times['end'])

            # Handle after-hours extending to next day
            if session_name == 'afterhours' and session_times['end'] < session_times['start']:
                from datetime import timedelta
                end_et += timedelta(days=1)

            # Localize to Eastern Time then convert to UTC
            start_et = start_et.replace(tzinfo=self.et_tz)
            end_et = end_et.replace(tzinfo=self.et_tz)

            boundaries[session_name] = {
                'start': start_et.astimezone(self.utc_tz),
                'end': end_et.astimezone(self.utc_tz)
            }

        return boundaries

    def classify_timestamp(self, timestamp: datetime,
                          exchange: str = 'NYSE') -> Dict[str, Union[str, bool]]:
        """
        Classify a UTC timestamp with full context.

        Args:
            timestamp: UTC timestamp to classify
            exchange: Exchange to use for classification

        Returns:
            Dictionary with classification details
        """
        # Ensure UTC timezone
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self.utc_tz)
        elif timestamp.tzinfo != self.utc_tz:
            timestamp = timestamp.astimezone(self.utc_tz)

        # Convert to Eastern Time for session logic
        et_timestamp = timestamp.astimezone(self.et_tz)
        trading_date = et_timestamp.date()

        # Basic classification
        classification = {
            'timestamp_utc': timestamp,
            'timestamp_et': et_timestamp,
            'trading_date': trading_date,
            'is_trading_day': self.is_trading_day(trading_date, exchange),
            'session': self.get_current_session(timestamp, exchange),
            'exchange': exchange,
            'is_early_close': self.get_early_close_time(trading_date, exchange) is not None
        }

        # Add session-specific metadata
        if classification['session']:
            session_boundaries = self.get_session_boundaries(trading_date, exchange)
            if classification['session'] in session_boundaries:
                session_info = session_boundaries[classification['session']]
                classification['session_start_utc'] = session_info['start']
                classification['session_end_utc'] = session_info['end']

        return classification

    def get_trading_schedule(self, start_date: Union[date, str],
                           end_date: Union[date, str],
                           exchange: str = 'NYSE') -> pd.DataFrame:
        """
        Get trading schedule for date range with session details.

        Args:
            start_date: Start date
            end_date: End date
            exchange: Exchange calendar to use

        Returns:
            DataFrame with trading schedule and session information
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        calendar = self.calendars.get(exchange, self.calendars['NYSE'])

        # Get base schedule
        schedule = calendar.schedule(start_date=start_date, end_date=end_date)

        # Add session information
        enhanced_schedule = []

        for _, row in schedule.iterrows():
            trading_date = row.name.date()

            # Get session boundaries
            boundaries = self.get_session_boundaries(trading_date, exchange)

            # Create enhanced row
            enhanced_row = {
                'trading_date': trading_date,
                'market_open': row['market_open'],
                'market_close': row['market_close'],
                'is_early_close': self.get_early_close_time(trading_date, exchange) is not None,
                'exchange': exchange
            }

            # Add session boundaries
            for session_name, session_info in boundaries.items():
                enhanced_row[f'{session_name}_start'] = session_info['start']
                enhanced_row[f'{session_name}_end'] = session_info['end']

            enhanced_schedule.append(enhanced_row)

        return pd.DataFrame(enhanced_schedule)

    def get_next_trading_day(self, from_date: Union[date, str],
                           exchange: str = 'NYSE') -> Optional[date]:
        """
        Get next trading day after given date.

        Args:
            from_date: Starting date
            exchange: Exchange calendar to use

        Returns:
            Next trading date or None if error
        """
        if isinstance(from_date, str):
            from_date = datetime.strptime(from_date, '%Y-%m-%d').date()

        calendar = self.calendars.get(exchange, self.calendars['NYSE'])

        try:
            # Look ahead up to 10 days for next trading day
            from datetime import timedelta
            end_date = from_date + timedelta(days=10)

            schedule = calendar.schedule(start_date=from_date + timedelta(days=1),
                                       end_date=end_date)

            if len(schedule) > 0:
                return schedule.index[0].date()

        except Exception as e:
            logger.error(f"Error finding next trading day after {from_date}: {e}")

        return None

    def get_last_trading_day(self, from_date: Union[date, str],
                           exchange: str = 'NYSE') -> Optional[date]:
        """
        Get last trading day before given date (institutional smart fallback).

        Args:
            from_date: Starting date to look back from
            exchange: Exchange calendar to use

        Returns:
            Last trading date or None if error
        """
        if isinstance(from_date, str):
            from_date = datetime.strptime(from_date, '%Y-%m-%d').date()

        # If the date itself is a trading day, return it
        if self.is_trading_day(from_date, exchange):
            return from_date

        calendar = self.calendars.get(exchange, self.calendars['NYSE'])

        try:
            # Look back up to 30 days for previous trading day
            from datetime import timedelta
            start_date = from_date - timedelta(days=30)

            schedule = calendar.schedule(start_date=start_date, end_date=from_date - timedelta(days=1))

            if len(schedule) > 0:
                # Return the most recent trading day
                return schedule.index[-1].date()

        except Exception as e:
            logger.error(f"Error finding last trading day before {from_date}: {e}")

        return None

    def adjust_request_date(self, request_date: Union[date, str, datetime],
                          symbol: Optional[str] = None,
                          exchange: str = 'NYSE') -> Tuple[date, Dict[str, str]]:
        """
        Universal date adjustment for institutional smart fallback.

        Handles weekends, holidays, future dates, and historical limits.

        Args:
            request_date: Date to validate and potentially adjust
            symbol: Symbol for asset class constraints (optional)
            exchange: Exchange calendar to use

        Returns:
            (adjusted_date, adjustment_metadata)
        """
        # Convert input to date
        if isinstance(request_date, str):
            if len(request_date) == 8:  # YYYYMMDD format
                request_date = datetime.strptime(request_date, '%Y%m%d').date()
            else:  # YYYY-MM-DD format
                request_date = datetime.strptime(request_date, '%Y-%m-%d').date()
        elif isinstance(request_date, datetime):
            request_date = request_date.date()

        metadata = {
            'original_request_date': request_date.isoformat(),
            'adjustment_reason': 'none',
            'asset_class': 'unknown'
        }

        today = date.today()

        # Check if future date
        if request_date > today:
            adjusted_date = self.get_last_trading_day(today, exchange)
            metadata['adjustment_reason'] = 'future_date_fallback'
            metadata['adjusted_date'] = adjusted_date.isoformat() if adjusted_date else None
            logger.info(f"Future date {request_date} adjusted to last trading day: {adjusted_date}")
            return adjusted_date or today, metadata

        # Check if it's a trading day
        if self.is_trading_day(request_date, exchange):
            metadata['adjusted_date'] = request_date.isoformat()
            return request_date, metadata

        # Not a trading day - find last trading day
        adjusted_date = self.get_last_trading_day(request_date, exchange)

        if adjusted_date:
            # Determine reason
            if request_date.weekday() >= 5:  # Weekend
                metadata['adjustment_reason'] = 'weekend_fallback'
            else:  # Holiday
                metadata['adjustment_reason'] = 'holiday_fallback'

            metadata['adjusted_date'] = adjusted_date.isoformat()
            logger.info(f"Non-trading date {request_date} ({metadata['adjustment_reason']}) "
                       f"adjusted to last trading day: {adjusted_date}")
            return adjusted_date, metadata

        # Fallback if no trading day found
        logger.error(f"Could not find suitable trading day for {request_date}")
        metadata['adjustment_reason'] = 'no_trading_day_found'
        metadata['adjusted_date'] = request_date.isoformat()
        return request_date, metadata

    def format_storage_key(self, timestamp: datetime,
                          symbol: str,
                          data_type: str = 'ticks',
                          exchange: str = 'NYSE') -> str:
        """
        Generate industry-standard storage key with session information.

        Args:
            timestamp: UTC timestamp
            symbol: Symbol name
            data_type: Data type ('ticks', 'bars', 'daily')
            exchange: Exchange for session classification

        Returns:
            Storage key following industry standards
        """
        classification = self.classify_timestamp(timestamp, exchange)

        session = classification['session'] or 'closed'
        trading_date = classification['trading_date'].strftime('%Y-%m-%d')

        # Industry-standard namespace: exchange/session/datatype/symbol/date
        return f"{exchange.lower()}/{session}/{data_type}/{symbol}/{trading_date}"

    def get_liquidity_flags(self, timestamp: datetime,
                           exchange: str = 'NYSE') -> Dict[str, Union[bool, float]]:
        """
        Get liquidity characteristics for timestamp based on session.

        Args:
            timestamp: UTC timestamp
            exchange: Exchange for session classification

        Returns:
            Dictionary with liquidity flags and multipliers
        """
        classification = self.classify_timestamp(timestamp, exchange)
        session = classification['session']

        # Industry-standard liquidity characteristics
        liquidity_profiles = {
            'regular': {
                'is_extended_hours': False,
                'liquidity_warning': False,
                'spread_multiplier': 1.0,
                'volume_expectation': 1.0
            },
            'premarket': {
                'is_extended_hours': True,
                'liquidity_warning': True,
                'spread_multiplier': 2.5,  # 2-5x wider spreads
                'volume_expectation': 0.15  # ~15% of regular volume
            },
            'afterhours': {
                'is_extended_hours': True,
                'liquidity_warning': True,
                'spread_multiplier': 3.0,  # Higher spreads after hours
                'volume_expectation': 0.10  # ~10% of regular volume
            }
        }

        return liquidity_profiles.get(session, {
            'is_extended_hours': True,
            'liquidity_warning': True,
            'spread_multiplier': 1.0,
            'volume_expectation': 0.0
        })