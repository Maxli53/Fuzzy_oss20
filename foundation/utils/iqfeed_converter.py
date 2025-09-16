"""
IQFeed to Pydantic Conversion Utilities

Converts real IQFeed numpy structured arrays to foundation Pydantic models.
Uses REAL data structure, no mock data as per CLAUDE.md guidelines.

IQFeed Tick Structure (14 fields):
('tick_id', 'date', 'time', 'last', 'last_sz', 'last_type',
 'mkt_ctr', 'tot_vlm', 'bid', 'ask', 'cond1', 'cond2', 'cond3', 'cond4')

Example real tick:
(266829, '2025-09-15', 63627382149, 236.38, 15, b'O', 19, 42615908, 236.35, 236.4, 135, 23, 0, 0)
"""

import numpy as np
from decimal import Decimal
from datetime import datetime, date, time
import pytz
from typing import Optional, List, Union

from foundation.models.market import TickData
from foundation.models.enums import TradeSign


def combine_date_time(date_val: Union[str, np.datetime64, date],
                     time_val: Union[int, np.timedelta64]) -> datetime:
    """
    Combine IQFeed date and time fields into ET datetime.

    Args:
        date_val: Date from IQFeed (string, numpy datetime64, or date object)
        time_val: Time from IQFeed (int microseconds or numpy timedelta64)

    Returns:
        Combined datetime in ET (Eastern Time)
    """
    # Handle different date formats
    if isinstance(date_val, np.datetime64):
        # Convert numpy.datetime64 to date object
        date_obj = date_val.astype('datetime64[D]').astype(date)
    elif isinstance(date_val, str):
        date_obj = datetime.strptime(date_val, '%Y-%m-%d').date()
    elif isinstance(date_val, date):
        date_obj = date_val
    else:
        # Try converting to string first (in case it's a numpy string type)
        try:
            date_str = str(date_val)
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        except:
            raise ValueError(f"Unsupported date format: {type(date_val)} = {date_val}")

    # Handle different time formats
    if isinstance(time_val, np.timedelta64):
        # Convert numpy timedelta64 to microseconds
        time_microseconds = int(time_val.astype('int64'))
    elif isinstance(time_val, int):
        time_microseconds = time_val
    else:
        raise ValueError(f"Unsupported time format: {type(time_val)}")

    # Convert microseconds to time components
    total_seconds = time_microseconds / 1_000_000
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    microseconds = int(time_microseconds % 1_000_000)

    # Create time object
    time_obj = time(hour=hours, minute=minutes, second=seconds, microsecond=microseconds)

    # Combine into datetime
    dt = datetime.combine(date_obj, time_obj)

    # IQFeed times are in ET - keep them in ET
    et_tz = pytz.timezone('America/New_York')
    return et_tz.localize(dt)


def decode_exchange_code(exchange_bytes: Union[bytes, str]) -> str:
    """
    Decode IQFeed exchange code.

    Args:
        exchange_bytes: Exchange code as bytes or string

    Returns:
        Exchange code as string
    """
    if isinstance(exchange_bytes, bytes):
        return exchange_bytes.decode('utf-8').strip()
    return str(exchange_bytes).strip()


def classify_trade_direction(price: float, bid: Optional[float], ask: Optional[float]) -> int:
    """
    Simple trade direction classification (simplified Lee-Ready algorithm).

    Args:
        price: Trade price
        bid: Best bid at time of trade
        ask: Best ask at time of trade

    Returns:
        Trade sign: +1 (buy), -1 (sell), 0 (unknown)
    """
    if bid is None or ask is None:
        return 0  # Unknown

    if bid <= 0 or ask <= 0:
        return 0  # Invalid quotes

    midpoint = (bid + ask) / 2

    if price > midpoint:
        return 1  # Buy (above midpoint)
    elif price < midpoint:
        return -1  # Sell (below midpoint)
    else:
        return 0  # At midpoint (unknown)


def convert_iqfeed_tick_to_pydantic(iqfeed_tick: np.void, symbol: str, tick_sequence: int = 0) -> TickData:
    """
    Convert single IQFeed numpy tick to Pydantic TickData model.

    Args:
        iqfeed_tick: Single tick from IQFeed numpy structured array
        symbol: Trading symbol (e.g., 'AAPL')
        tick_sequence: Sequence number for trades at same timestamp

    Returns:
        TickData Pydantic model

    Example:
        >>> data = hist_conn.request_ticks("AAPL", max_ticks=1)
        >>> tick_model = convert_iqfeed_tick_to_pydantic(data[0], "AAPL")
    """
    # Extract fields from numpy structured array
    # Field indices: [0=tick_id, 1=date, 2=time, 3=last, 4=last_sz, 5=last_type,
    #                 6=mkt_ctr, 7=tot_vlm, 8=bid, 9=ask, 10=cond1, 11=cond2, 12=cond3, 13=cond4]

    tick_id = int(iqfeed_tick[0])
    date_val = iqfeed_tick[1]
    # Handle numpy.timedelta64[us] time field
    time_field = iqfeed_tick[2]
    if isinstance(time_field, np.timedelta64):
        time_us = int(time_field.astype('int64'))  # Convert to microseconds
    else:
        time_us = int(time_field)
    price = float(iqfeed_tick[3])
    size = int(iqfeed_tick[4])
    exchange_bytes = iqfeed_tick[5]
    market_center = int(iqfeed_tick[6])
    total_volume = int(iqfeed_tick[7])
    bid = float(iqfeed_tick[8]) if iqfeed_tick[8] > 0 else None
    ask = float(iqfeed_tick[9]) if iqfeed_tick[9] > 0 else None
    cond1 = int(iqfeed_tick[10])
    cond2 = int(iqfeed_tick[11])
    cond3 = int(iqfeed_tick[12])
    cond4 = int(iqfeed_tick[13])

    # Convert timestamp
    timestamp = combine_date_time(date_val, iqfeed_tick[2])

    # Decode exchange
    exchange = decode_exchange_code(exchange_bytes)

    # Build conditions string (max 4 chars per model constraint)
    conditions_parts = []
    if cond1 != 0:
        conditions_parts.append(str(cond1))
    if cond2 != 0:
        conditions_parts.append(str(cond2))
    if cond3 != 0:
        conditions_parts.append(str(cond3))
    if cond4 != 0:
        conditions_parts.append(str(cond4))

    conditions = ",".join(conditions_parts)[:4]  # Limit to 4 chars

    # Classify trade direction
    trade_sign = classify_trade_direction(price, bid, ask)

    # Pre-compute derived fields to avoid model recursion
    tick_data = {
        "symbol": symbol,
        "tick_id": tick_id,  # Add IQFeed tick ID for debugging
        "timestamp": timestamp,
        "price": Decimal(str(price)),
        "size": size,
        "exchange": exchange,
        "market_center": market_center,
        "total_volume": total_volume,
        "conditions": conditions,
        "trade_sign": trade_sign,
        "tick_sequence": tick_sequence,  # Add sequence number
    }

    # Add optional bid/ask
    if bid is not None:
        tick_data["bid"] = Decimal(str(bid))
    if ask is not None:
        tick_data["ask"] = Decimal(str(ask))

    # Pre-compute spread metrics if both bid and ask available
    if bid is not None and ask is not None:
        bid_decimal = Decimal(str(bid))
        ask_decimal = Decimal(str(ask))
        spread = ask_decimal - bid_decimal
        midpoint = (bid_decimal + ask_decimal) / Decimal('2')

        tick_data["spread"] = spread
        tick_data["midpoint"] = midpoint

        if midpoint > 0:
            tick_data["spread_bps"] = float((spread / midpoint) * 10000)
            tick_data["spread_pct"] = float(spread / midpoint)
            tick_data["effective_spread"] = 2 * abs(Decimal(str(price)) - midpoint)

    # Pre-compute dollar volume
    tick_data["dollar_volume"] = Decimal(str(price)) * Decimal(str(size))

    # Calculate price improvement (negative means worse execution)
    if trade_sign == 1 and midpoint is not None:
        # For buy trades: positive if below midpoint, negative if above
        tick_data["price_improvement"] = midpoint - Decimal(str(price))
    elif trade_sign == -1 and midpoint is not None:
        # For sell trades: positive if above midpoint, negative if below
        tick_data["price_improvement"] = Decimal(str(price)) - midpoint
    else:
        tick_data["price_improvement"] = None

    # Set block trade flag
    tick_data["is_block_trade"] = size >= 10000

    # Determine if regular trade (all conditions are 0)
    tick_data["is_regular"] = all(c == 0 for c in [cond1, cond2, cond3, cond4])

    # Set specific condition flags based on common IQFeed codes
    tick_data["is_extended_hours"] = cond1 == 135 or cond2 == 135
    tick_data["is_odd_lot"] = cond3 == 23  # Per Data_policy.md: only check cond3
    tick_data["is_intermarket_sweep"] = cond1 == 37 or cond2 == 37
    tick_data["is_derivatively_priced"] = cond2 == 61  # Per Data_policy.md
    tick_data["is_qualified"] = True  # Default per Data_policy.md

    return TickData(**tick_data)


def convert_iqfeed_ticks_to_pydantic(iqfeed_data: np.ndarray, symbol: str) -> List[TickData]:
    """
    Convert multiple IQFeed numpy ticks to Pydantic TickData models.
    Assigns sequence numbers for trades at the same timestamp.

    Args:
        iqfeed_data: Numpy structured array from IQFeed
        symbol: Trading symbol (e.g., 'AAPL')

    Returns:
        List of TickData Pydantic models with sequence numbers

    Example:
        >>> data = hist_conn.request_ticks("AAPL", max_ticks=100)
        >>> tick_models = convert_iqfeed_ticks_to_pydantic(data, "AAPL")
    """
    tick_models = []

    # Track timestamps for sequence numbering
    last_timestamp = None
    sequence = 0

    for iqfeed_tick in iqfeed_data:
        try:
            # Calculate timestamp for this tick to determine sequence number
            date_val = iqfeed_tick[1]
            time_val = iqfeed_tick[2]
            current_timestamp = combine_date_time(date_val, time_val)

            # Assign sequence number based on timestamp
            if last_timestamp is not None and current_timestamp == last_timestamp:
                sequence += 1
            else:
                sequence = 0
                last_timestamp = current_timestamp

            # Convert with sequence number
            tick_model = convert_iqfeed_tick_to_pydantic(iqfeed_tick, symbol, tick_sequence=sequence)
            tick_models.append(tick_model)
        except Exception as e:
            # Log error but continue processing other ticks
            print(f"Warning: Failed to convert tick {iqfeed_tick[0]}: {e}")
            continue

    return tick_models


def test_conversion_with_real_data(symbol: str = "AAPL", max_ticks: int = 10) -> List[TickData]:
    """
    Test conversion using real IQFeed data.

    Args:
        symbol: Symbol to fetch (default: AAPL)
        max_ticks: Maximum number of ticks to fetch

    Returns:
        List of converted TickData models
    """
    try:
        import pyiqfeed as iq

        hist_conn = iq.HistoryConn(name="test-conversion")

        with iq.ConnConnector([hist_conn]) as connector:
            # Get real data from IQFeed
            iqfeed_data = hist_conn.request_ticks(symbol, max_ticks=max_ticks)

            if iqfeed_data is None or len(iqfeed_data) == 0:
                print(f"No data received for {symbol}")
                return []

            # Convert to Pydantic models
            tick_models = convert_iqfeed_ticks_to_pydantic(iqfeed_data, symbol)

            print(f"Successfully converted {len(tick_models)}/{len(iqfeed_data)} ticks for {symbol}")

            if tick_models:
                first_tick = tick_models[0]
                print(f"First tick: {symbol} @ ${first_tick.price} ({first_tick.size} shares) on {first_tick.exchange}")

            return tick_models

    except ImportError:
        print("PyIQFeed not available")
        return []
    except Exception as e:
        print(f"Error in test conversion: {e}")
        return []


if __name__ == "__main__":
    # Test the conversion with real data
    test_conversion_with_real_data("AAPL", 5)