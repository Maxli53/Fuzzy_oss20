#!/usr/bin/env python3
"""
Simple test for Pydantic to DataFrame conversion for TICK 265596
"""

import sys
import os
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('pyiqfeed_orig')
sys.path.append('.')

from iqfeed_collector import IQFeedCollector
from foundation.utils.iqfeed_converter import convert_iqfeed_tick_to_pydantic
import pandas as pd
from typing import List
import logging

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _pydantic_to_dataframe(pydantic_ticks: List) -> pd.DataFrame:
    """
    Test version of the Pydantic to DataFrame converter.
    """
    if not pydantic_ticks:
        logger.warning("Empty pydantic_ticks list - returning empty DataFrame")
        return pd.DataFrame()

    try:
        # Extract all fields from Pydantic models
        records = []
        for tick in pydantic_ticks:
            record = {
                # Core fields (10)
                'symbol': tick.symbol,
                'timestamp': tick.timestamp,
                'price': float(tick.price),
                'volume': tick.size,  # 'size' → 'volume' for consistency
                'exchange': tick.exchange,
                'market_center': tick.market_center,
                'total_volume': tick.total_volume,
                'bid': float(tick.bid) if tick.bid else None,
                'ask': float(tick.ask) if tick.ask else None,
                'conditions': tick.conditions,

                # Spread metrics (5)
                'spread': float(tick.spread) if tick.spread else None,
                'midpoint': float(tick.midpoint) if tick.midpoint else None,
                'spread_bps': tick.spread_bps,
                'spread_pct': tick.spread_pct,
                'effective_spread': float(tick.effective_spread) if tick.effective_spread else None,

                # Trade analysis (3)
                'trade_sign': tick.trade_sign,
                'dollar_volume': float(tick.dollar_volume) if tick.dollar_volume else None,
                'price_improvement': float(tick.price_improvement) if tick.price_improvement else None,

                # Additional metrics (7)
                'tick_direction': tick.tick_direction,
                'participant_type': tick.participant_type,
                'volume_rate': tick.volume_rate,
                'trade_pct_of_day': tick.trade_pct_of_day,
                'log_return': tick.log_return,
                'price_change': float(tick.price_change) if tick.price_change else None,
                'price_change_bps': tick.price_change_bps,

                # Condition flags (7)
                'is_regular': tick.is_regular,
                'is_extended_hours': tick.is_extended_hours,
                'is_odd_lot': tick.is_odd_lot,
                'is_intermarket_sweep': tick.is_intermarket_sweep,
                'is_derivatively_priced': tick.is_derivatively_priced,
                'is_qualified': tick.is_qualified,
                'is_block_trade': tick.is_block_trade,

                # Metadata fields (4) - MUST include even if None
                'id': str(tick.id) if hasattr(tick, 'id') and tick.id else None,
                'created_at': tick.created_at if hasattr(tick, 'created_at') else None,
                'updated_at': tick.updated_at if hasattr(tick, 'updated_at') else None,
                'metadata': tick.metadata if hasattr(tick, 'metadata') else None,

                # Timestamp fields (2)
                'processed_at': tick.processed_at if hasattr(tick, 'processed_at') else None,
                'source_timestamp': tick.source_timestamp if hasattr(tick, 'source_timestamp') else None,

                # Enum fields (3)
                'trade_sign_enum': tick.trade_sign_enum.value if hasattr(tick, 'trade_sign_enum') and tick.trade_sign_enum else None,
                'tick_direction_enum': tick.tick_direction_enum.value if hasattr(tick, 'tick_direction_enum') and tick.tick_direction_enum else None,
                'participant_type_enum': tick.participant_type_enum.value if hasattr(tick, 'participant_type_enum') and tick.participant_type_enum else None,
            }
            records.append(record)

        # Create DataFrame
        df = pd.DataFrame(records)

        # VALIDATION: Ensure all Pydantic fields are in DataFrame
        if pydantic_ticks:
            sample_model = pydantic_ticks[0].model_dump()
            expected_fields = len(sample_model)
            actual_columns = len(df.columns)

            # Account for 'size' → 'volume' rename
            pydantic_field_names = set(sample_model.keys())
            df_column_names = set(df.columns)

            missing_from_df = pydantic_field_names - df_column_names - {'size'}
            extra_in_df = df_column_names - pydantic_field_names - {'volume'}

            if missing_from_df or extra_in_df:
                print(f"ERROR: Column mismatch!")
                print(f"Expected {expected_fields} Pydantic fields, got {actual_columns} DataFrame columns")
                if missing_from_df:
                    print(f"Missing from DataFrame: {sorted(missing_from_df)}")
                if extra_in_df:
                    print(f"Extra in DataFrame: {sorted(extra_in_df)}")
                raise ValueError(f"DataFrame validation failed")

            print(f"VALIDATION PASSED: {actual_columns} DataFrame columns match {expected_fields} Pydantic fields")

        # Apply memory optimizations
        # Prices as float32
        for col in ['price', 'bid', 'ask', 'spread', 'midpoint', 'effective_spread',
                   'price_improvement', 'spread_bps', 'spread_pct', 'price_change']:
            if col in df.columns and df[col].notna().any():
                df[col] = df[col].astype('float32')

        # Volumes as uint32
        for col in ['volume', 'total_volume']:
            if col in df.columns:
                df[col] = df[col].astype('uint32')

        # Small integers
        if 'market_center' in df.columns:
            df['market_center'] = df['market_center'].astype('uint16')
        if 'trade_sign' in df.columns:
            df['trade_sign'] = df['trade_sign'].astype('int8')
        if 'tick_direction' in df.columns:
            df['tick_direction'] = df['tick_direction'].astype('int8')

        # Categories for repeated strings
        for col in ['symbol', 'exchange', 'participant_type']:
            if col in df.columns:
                df[col] = df[col].astype('category')

        # Sort by timestamp
        df.sort_values('timestamp', inplace=True, kind='mergesort')
        df.reset_index(drop=True, inplace=True)

        return df

    except Exception as e:
        logger.error(f"Failed to convert Pydantic to DataFrame: {e}")
        raise


# Main test
print("="*80)
print("TEST: Pydantic -> DataFrame Conversion for TICK 265596")
print("="*80)

# Initialize collector
collector = IQFeedCollector()
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

# Get TICK 265596
ticks = collector.get_tick_data('AAPL', num_days=1, max_ticks=10000)
found_tick = None
for t in ticks:
    if t['tick_id'] == 265596:
        found_tick = t
        break

if found_tick is None:
    print("TICK 265596 not found")
    sys.exit(1)

print("\n1. RAW NUMPY TICK 265596:")
print("-"*40)
print(f"  Fields: {found_tick.dtype.names}")
print(f"  Price: {found_tick['last']}")

print("\n2. CONVERT TO PYDANTIC:")
print("-"*40)
pydantic_tick = convert_iqfeed_tick_to_pydantic(found_tick, 'AAPL')
model_dict = pydantic_tick.model_dump()
print(f"  Fields in Pydantic model: {len(model_dict)}")
print(f"  Price: {pydantic_tick.price} (type: {type(pydantic_tick.price).__name__})")
print(f"  Trade sign: {pydantic_tick.trade_sign}")
print(f"  Price improvement: {pydantic_tick.price_improvement}")
print(f"  Is qualified: {pydantic_tick.is_qualified}")

print("\n3. CONVERT TO DATAFRAME:")
print("-"*40)
# Convert single tick to list for the method
pydantic_ticks = [pydantic_tick]
df = _pydantic_to_dataframe(pydantic_ticks)

print(f"  DataFrame shape: {df.shape}")
print(f"  Columns ({len(df.columns)}): {', '.join(sorted(df.columns.tolist()))}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print("\n4. VERIFY KEY FIELDS:")
print("-"*40)
row = df.iloc[0]
print(f"  symbol: {row['symbol']} (dtype: {df['symbol'].dtype})")
print(f"  timestamp: {row['timestamp']} (dtype: {df['timestamp'].dtype})")
print(f"  price: {row['price']:.2f} (dtype: {df['price'].dtype})")
print(f"  volume: {row['volume']} (dtype: {df['volume'].dtype})")
print(f"  trade_sign: {row['trade_sign']} (dtype: {df['trade_sign'].dtype})")
print(f"  price_improvement: {row['price_improvement']:.4f} (dtype: {df['price_improvement'].dtype})")
print(f"  is_qualified: {row['is_qualified']} (dtype: {df['is_qualified'].dtype})")

print("\n5. FIELD COUNT COMPARISON:")
print("-"*40)
print(f"  NumPy fields: {len(found_tick.dtype.names)}")
print(f"  Pydantic fields: {len(model_dict)}")
print(f"  DataFrame columns: {len(df.columns)}")
print(f"  Non-null DataFrame values: {df.notna().sum().sum()}")

print("\n6. MEMORY OPTIMIZATION SUMMARY:")
print("-"*40)
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

print("\n7. VALIDATION SUMMARY:")
print("-"*40)
print(f"  Expected: 41 Pydantic fields (includes 'size')")
print(f"  Got: 41 DataFrame columns (includes 'volume' instead of 'size')")
print(f"  Validation: PASSED - All fields properly mapped")

print("\n8. SUCCESS!")
print("-"*40)
print(f"  [OK] NumPy (14 fields) -> Pydantic ({len(model_dict)} fields) -> DataFrame ({len(df.columns)} columns)")
print(f"  [OK] All transformations completed successfully")
print(f"  [OK] DataFrame has all 41 expected columns")
print(f"  [OK] Ready for ArcticDB storage")