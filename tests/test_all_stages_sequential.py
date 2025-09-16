#!/usr/bin/env python3
"""
SINGLE TEST FILE - All stages sequentially with RAW data output
Stage 1: IQFeed -> NumPy (RAW)
Stage 2: NumPy -> Pydantic
Stage 3: Pydantic -> DataFrame
Stage 4: DataFrame -> ArcticDB
Stage 5: ArcticDB -> DataFrame
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Add paths
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(parent_dir, 'pyiqfeed_orig'))
sys.path.insert(0, os.path.join(parent_dir, 'stage_01_data_engine', 'collectors'))
sys.path.insert(0, os.path.join(parent_dir, 'stage_01_data_engine', 'storage'))
sys.path.append(parent_dir)

# Direct imports to avoid circular dependency
from iqfeed_collector import IQFeedCollector
from tick_store import TickStore

try:
    from foundation.models.market import TickData
    from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic
    FOUNDATION_MODELS = True
except ImportError:
    FOUNDATION_MODELS = False
    print("WARNING: Foundation Models not available")

def print_separator(title, level=1):
    """Print section separator"""
    if level == 1:
        print(f"\n{'='*80}")
        print(f"    {title}")
        print('='*80)
    else:
        print(f"\n{'-'*60}")
        print(f"  {title}")
        print('-'*60)

def test_stage1_raw_numpy():
    """STAGE 1: IQFeed -> NumPy with RAW data display"""

    print_separator("STAGE 1: IQFeed -> NumPy (RAW DATA)")

    collector = IQFeedCollector()
    if not collector.ensure_connection():
        print("[FAIL] Cannot connect to IQFeed")
        return None

    # Get REAL AAPL data
    print("Fetching REAL AAPL tick data...")
    tick_array = collector.get_tick_data("AAPL", num_days=1, max_ticks=100)

    if tick_array is None or len(tick_array) == 0:
        print("[FAIL] No data received")
        return None

    print(f"\n[OK] Got {len(tick_array)} ticks")
    print(f"Array type: {type(tick_array)}")
    print(f"Array dtype: {tick_array.dtype}")
    print(f"Field names: {tick_array.dtype.names}")

    print_separator("RAW NUMPY DATA - First 5 Ticks", level=2)
    print("\nSHOWING ALL 14 FIELDS EXACTLY AS THEY COME FROM IQFEED:")

    for i in range(min(5, len(tick_array))):
        tick = tick_array[i]
        print(f"\n--- Tick {i+1} RAW DATA ---")
        print(f"Complete tuple: {tick}")
        print("Field breakdown:")
        for field in tick_array.dtype.names:
            print(f"  {field:12}: {tick[field]}")

    # Show raw array representation
    print_separator("RAW ARRAY REPRESENTATION", level=2)
    print("numpy.array repr() for first 3 ticks:")
    print(repr(tick_array[:3]))

    # Data quality check
    print_separator("Stage 1 Quality Check", level=2)
    print(f"Total ticks: {len(tick_array)}")
    print(f"Price range: ${tick_array['last'].min():.4f} - ${tick_array['last'].max():.4f}")
    print(f"Volume total: {tick_array['last_sz'].sum():,}")
    print(f"All prices positive: {np.all(tick_array['last'] > 0)}")

    return tick_array

def test_stage2_pydantic(tick_array):
    """STAGE 2: NumPy -> Pydantic Models"""

    print_separator("STAGE 2: NumPy -> Pydantic Models")

    if not FOUNDATION_MODELS:
        print("[SKIP] Foundation Models not available")
        return None

    print("Converting to Pydantic models...")
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, "AAPL")

    print(f"[OK] Converted {len(pydantic_ticks)} ticks")

    print_separator("PYDANTIC MODEL DATA - First 3 Ticks", level=2)

    for i in range(min(3, len(pydantic_ticks))):
        tick = pydantic_ticks[i]
        print(f"\n--- Tick {i+1} PYDANTIC MODEL ---")
        print(f"Model fields: {tick.model_dump().keys()}")
        print("\nCore fields:")
        print(f"  symbol: {tick.symbol}")
        print(f"  price: ${tick.price}")
        print(f"  size: {tick.size}")
        print(f"  timestamp: {tick.timestamp} (TZ: {tick.timestamp.tzinfo})")
        print(f"  bid/ask: ${tick.bid}/{tick.ask}")
        print("\nEnhanced fields:")
        print(f"  spread_bps: {tick.spread_bps:.2f}")
        print(f"  trade_sign: {tick.trade_sign}")
        print(f"  dollar_volume: ${tick.dollar_volume:.2f}")
        print(f"  is_block_trade: {tick.is_block_trade}")
        print(f"  participant_type: {tick.participant_type}")

    # Show field count
    print_separator("Stage 2 Quality Check", level=2)
    print(f"NumPy fields: 14")
    print(f"Pydantic fields: {len(pydantic_ticks[0].model_dump().keys())}")
    print(f"Enhancement ratio: {len(pydantic_ticks[0].model_dump().keys())/14:.1f}x")

    return pydantic_ticks

def test_stage3_dataframe(pydantic_ticks):
    """STAGE 3: Pydantic -> DataFrame"""

    print_separator("STAGE 3: Pydantic -> DataFrame")

    print("Converting to DataFrame...")

    # Manual conversion (simplified version of what TickStore will do)
    records = []
    for tick in pydantic_ticks:
        record = {
            'timestamp': tick.timestamp,
            'symbol': tick.symbol,
            'price': float(tick.price),
            'volume': tick.size,
            'bid': float(tick.bid) if tick.bid else None,
            'ask': float(tick.ask) if tick.ask else None,
            'spread_bps': float(tick.spread_bps) if tick.spread_bps else None,
            'trade_sign': tick.trade_sign,
            'dollar_volume': float(tick.dollar_volume) if tick.dollar_volume else None,
            'is_block_trade': tick.is_block_trade,
            'participant_type': str(tick.participant_type),
        }
        records.append(record)

    df = pd.DataFrame(records)

    print(f"[OK] DataFrame created")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print_separator("DATAFRAME DATA - First 5 Rows", level=2)
    print(df.head().to_string())

    print_separator("Stage 3 Quality Check", level=2)
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"Data types: {df.dtypes.value_counts().to_dict()}")

    return df

def test_stage4_arcticdb(df):
    """STAGE 4: DataFrame -> ArcticDB"""

    print_separator("STAGE 4: DataFrame -> ArcticDB Storage")

    try:
        tick_store = TickStore()

        # Store the DataFrame
        print("Storing DataFrame in ArcticDB...")
        symbol = "AAPL"
        date = datetime.now().strftime("%Y%m%d")

        result = tick_store.store_ticks(
            symbol=symbol,
            date=date,
            tick_df=df,
            metadata={'test': True, 'stage': 4},
            overwrite=True
        )

        print(f"[OK] Stored {len(df)} ticks to ArcticDB")
        print(f"Symbol: {symbol}, Date: {date}")

        return True

    except Exception as e:
        print(f"[FAIL] ArcticDB storage failed: {e}")
        return False

def test_stage5_retrieval():
    """STAGE 5: ArcticDB -> DataFrame Retrieval"""

    print_separator("STAGE 5: ArcticDB -> DataFrame Retrieval")

    try:
        tick_store = TickStore()

        # Retrieve data
        print("Retrieving from ArcticDB...")
        symbol = "AAPL"
        date = datetime.now().strftime("%Y%m%d")

        retrieved_df = tick_store.load_ticks(symbol=symbol, date_range=date)

        if retrieved_df is not None and not retrieved_df.empty:
            print(f"[OK] Retrieved {len(retrieved_df)} ticks from ArcticDB")

            print_separator("RETRIEVED DATA - First 5 Rows", level=2)
            print(retrieved_df.head().to_string())

            return retrieved_df
        else:
            print("[FAIL] No data retrieved")
            return None

    except Exception as e:
        print(f"[FAIL] Retrieval failed: {e}")
        return None

def main():
    """Run all stages sequentially"""

    print("="*80)
    print("    COMPLETE PIPELINE TEST - ALL STAGES")
    print("    Using REAL AAPL tick data")
    print("="*80)

    # Stage 1
    tick_array = test_stage1_raw_numpy()
    if tick_array is None:
        print("\nSTOPPED: Stage 1 failed")
        return

    # input("\n>>> Press Enter to continue to Stage 2...")

    # Stage 2
    if FOUNDATION_MODELS:
        pydantic_ticks = test_stage2_pydantic(tick_array)
        if pydantic_ticks is None:
            print("\nSTOPPED: Stage 2 failed")
            return
    else:
        print("\n[SKIP] Stage 2 - Foundation Models not available")
        pydantic_ticks = None

    # input("\n>>> Press Enter to continue to Stage 3...")

    # Stage 3
    if pydantic_ticks:
        df = test_stage3_dataframe(pydantic_ticks)
        if df is None:
            print("\nSTOPPED: Stage 3 failed")
            return
    else:
        # Direct conversion from NumPy to DataFrame (fallback)
        print_separator("STAGE 3: Direct NumPy -> DataFrame (Fallback)")
        df = pd.DataFrame({
            'price': tick_array['last'],
            'volume': tick_array['last_sz'],
            'bid': tick_array['bid'],
            'ask': tick_array['ask']
        })
        print(f"[OK] Basic DataFrame created: {df.shape}")

    # input("\n>>> Press Enter to continue to Stage 4...")

    # Stage 4
    if test_stage4_arcticdb(df):
        # input("\n>>> Press Enter to continue to Stage 5...")

        # Stage 5
        retrieved_df = test_stage5_retrieval()

        if retrieved_df is not None:
            print("\n" + "="*80)
            print("    COMPLETE PIPELINE TEST SUCCESSFUL!")
            print("    All 5 stages completed")
            print("="*80)
        else:
            print("\nStage 5 failed")
    else:
        print("\nStage 4 failed - skipping Stage 5")

if __name__ == "__main__":
    main()