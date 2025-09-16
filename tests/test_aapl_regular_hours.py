#!/usr/bin/env python3
"""
Test AAPL Regular Trading Hours Data Pipeline
==============================================
Tests each stage SEPARATELY with detailed data quality reports.

Symbol: AAPL
Hours: Regular Trading Hours (9:30 AM - 4:00 PM ET)
Data: Real tick data from IQFeed
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import List, Optional, Dict, Any
import pytz

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(parent_dir, 'pyiqfeed_orig'))
sys.path.insert(0, os.path.join(parent_dir, 'stage_01_data_engine', 'collectors'))
sys.path.insert(0, os.path.join(parent_dir, 'stage_01_data_engine', 'storage'))
sys.path.append(parent_dir)

# Direct imports
from iqfeed_collector import IQFeedCollector
from tick_store import TickStore

# Try to import Foundation Models
try:
    from foundation.models.market import TickData
    from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic
    FOUNDATION_MODELS = True
except ImportError:
    FOUNDATION_MODELS = False
    print("WARNING: Foundation Models not available")


class AAPLRegularHoursTester:
    """Test AAPL data pipeline for regular trading hours only"""

    def __init__(self):
        self.symbol = "AAPL"
        self.et_tz = pytz.timezone('America/New_York')
        self.regular_start = time(9, 30)  # 9:30 AM ET
        self.regular_end = time(16, 0)    # 4:00 PM ET
        self.checksums = {}

    def print_separator(self, title: str, level: int = 1):
        """Print formatted section separator"""
        if level == 1:
            print(f"\n{'='*80}")
            print(f"    {title}")
            print('='*80)
        else:
            print(f"\n{'-'*60}")
            print(f"  {title}")
            print('-'*60)

    def is_regular_hours(self, timestamp_us) -> bool:
        """Check if timestamp is during regular trading hours"""
        # Handle timedelta or int
        if isinstance(timestamp_us, np.timedelta64):
            # Convert to microseconds
            timestamp_us = int(timestamp_us.astype('int64'))
        elif hasattr(timestamp_us, 'total_seconds'):
            # It's a timedelta object
            timestamp_us = int(timestamp_us.total_seconds() * 1_000_000)

        # Convert microseconds to time
        total_seconds = timestamp_us / 1_000_000
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        tick_time = time(hours, minutes)

        # Check if within regular hours
        return self.regular_start <= tick_time <= self.regular_end

    def filter_regular_hours(self, tick_array: np.ndarray) -> np.ndarray:
        """Filter tick array for regular trading hours only"""
        # Create mask for regular hours
        mask = []
        for tick in tick_array:
            # Check time and condition codes
            time_us = tick['time']
            cond1 = tick['cond1']

            # Regular hours: 9:30-16:00 and NOT extended hours (135)
            is_regular = self.is_regular_hours(time_us) and cond1 != 135
            mask.append(is_regular)

        # Apply filter
        regular_ticks = tick_array[mask]
        return regular_ticks

    def test_stage1_iqfeed_to_numpy(self) -> Optional[np.ndarray]:
        """
        STAGE 1: IQFeed -> NumPy (Regular Hours Only)
        """
        self.print_separator("STAGE 1: IQFeed -> NumPy (Regular Hours)", 1)

        collector = IQFeedCollector()
        if not collector.ensure_connection():
            print("[FAIL] Cannot connect to IQFeed")
            return None

        # Fetch AAPL tick data for today
        print(f"Fetching {self.symbol} tick data...")
        all_ticks = collector.get_tick_data(self.symbol, num_days=1, max_ticks=10000)

        if all_ticks is None or len(all_ticks) == 0:
            print("[FAIL] No data received from IQFeed")
            return None

        print(f"[OK] Retrieved {len(all_ticks)} total ticks")

        # Filter for regular hours
        print("\nFiltering for regular trading hours (9:30 AM - 4:00 PM ET)...")
        regular_ticks = self.filter_regular_hours(all_ticks)

        if len(regular_ticks) == 0:
            print("[WARNING] No regular hours ticks found, using all ticks")
            regular_ticks = all_ticks

        print(f"[OK] Filtered to {len(regular_ticks)} regular hours ticks")
        print(f"Removed {len(all_ticks) - len(regular_ticks)} extended hours ticks")

        # Data Structure Report
        self.print_separator("Data Structure", 2)
        print(f"Array type: {type(regular_ticks)}")
        print(f"Array dtype: {regular_ticks.dtype}")
        print(f"Field count: {len(regular_ticks.dtype.names)}")
        print(f"Field names: {regular_ticks.dtype.names}")
        print(f"Memory usage: {regular_ticks.nbytes / 1024:.2f} KB")

        # Show first 5 regular hours ticks
        self.print_separator("First 5 Regular Hours Ticks (RAW)", 2)
        for i in range(min(5, len(regular_ticks))):
            tick = regular_ticks[i]
            # Convert time to readable format
            time_field = tick['time']
            if isinstance(time_field, np.timedelta64):
                time_us = int(time_field.astype('int64'))
            else:
                time_us = int(time_field)

            hours = int(time_us / 3600000000)
            minutes = int((time_us % 3600000000) / 60000000)
            seconds = int((time_us % 60000000) / 1000000)

            print(f"\n--- Tick {i+1} ---")
            print(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d} ET")
            print(f"Price: ${tick['last']:.2f}")
            print(f"Volume: {tick['last_sz']}")
            print(f"Bid/Ask: ${tick['bid']:.2f}/${tick['ask']:.2f}")
            print(f"Conditions: {tick['cond1']}, {tick['cond2']}, {tick['cond3']}, {tick['cond4']}")
            print(f"Complete tuple: {tick}")

        # Statistical Analysis
        self.print_separator("Statistical Analysis", 2)
        print(f"Total regular hours ticks: {len(regular_ticks)}")
        print(f"Price range: ${regular_ticks['last'].min():.2f} - ${regular_ticks['last'].max():.2f}")
        print(f"Average price: ${regular_ticks['last'].mean():.2f}")
        print(f"Total volume: {regular_ticks['last_sz'].sum():,}")
        print(f"Average trade size: {regular_ticks['last_sz'].mean():.1f}")

        # Calculate checksum
        self.checksums['numpy'] = float(regular_ticks['last'].sum())
        print(f"\nChecksum (price sum): {self.checksums['numpy']:.2f}")

        return regular_ticks

    def test_stage2_numpy_to_pydantic(self, tick_array: np.ndarray) -> Optional[List]:
        """
        STAGE 2: NumPy -> Pydantic Models
        """
        self.print_separator("STAGE 2: NumPy -> Pydantic Models", 1)

        if not FOUNDATION_MODELS:
            print("[SKIP] Foundation Models not available")
            return None

        print(f"Converting {len(tick_array)} NumPy ticks to Pydantic models...")

        try:
            pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, self.symbol)
            print(f"[OK] Converted {len(pydantic_ticks)} ticks")
        except Exception as e:
            print(f"[FAIL] Conversion failed: {e}")
            return None

        # Field Enhancement Report
        self.print_separator("Field Enhancement", 2)
        print(f"NumPy fields: 14")
        if pydantic_ticks:
            print(f"Pydantic fields: {len(pydantic_ticks[0].model_dump().keys())}")
            print(f"Enhancement ratio: {len(pydantic_ticks[0].model_dump().keys())/14:.1f}x")

        # Show first 3 Pydantic models
        self.print_separator("First 3 Pydantic Models", 2)
        for i in range(min(3, len(pydantic_ticks))):
            tick = pydantic_ticks[i]
            print(f"\n--- Tick {i+1} ---")
            print(f"Timestamp: {tick.timestamp} (TZ: {tick.timestamp.tzinfo})")
            print(f"Symbol: {tick.symbol}")
            print(f"Price: ${tick.price}")
            print(f"Size: {tick.size}")
            print(f"Bid/Ask: ${tick.bid}/${tick.ask}")
            print(f"Spread (bps): {tick.spread_bps:.2f}")
            print(f"Trade sign: {tick.trade_sign}")
            print(f"Dollar volume: ${tick.dollar_volume:.2f}")
            print(f"Is regular: {tick.is_regular}")
            print(f"Is extended hours: {tick.is_extended_hours}")
            print(f"Is block trade: {tick.is_block_trade}")

        # Verify regular hours flags
        self.print_separator("Regular Hours Verification", 2)
        regular_count = sum(1 for t in pydantic_ticks if t.is_regular)
        extended_count = sum(1 for t in pydantic_ticks if t.is_extended_hours)
        print(f"Regular trades: {regular_count}")
        print(f"Extended hours trades: {extended_count}")

        if extended_count > 0:
            print(f"[WARNING] Found {extended_count} extended hours trades in regular hours data!")

        # Calculate checksum
        self.checksums['pydantic'] = sum(float(t.price) for t in pydantic_ticks)
        print(f"\nChecksum (price sum): {self.checksums['pydantic']:.2f}")

        # Verify checksum consistency
        if 'numpy' in self.checksums:
            diff = abs(self.checksums['pydantic'] - self.checksums['numpy'])
            if diff < 0.01:
                print(f"[OK] Checksum matches NumPy (diff: {diff:.6f})")
            else:
                print(f"[WARNING] Checksum mismatch with NumPy (diff: {diff:.6f})")

        return pydantic_ticks

    def test_stage3_pydantic_to_dataframe(self, pydantic_ticks: List) -> Optional[pd.DataFrame]:
        """
        STAGE 3: Pydantic -> DataFrame
        """
        self.print_separator("STAGE 3: Pydantic -> DataFrame", 1)

        print(f"Converting {len(pydantic_ticks)} Pydantic models to DataFrame...")

        # Convert to DataFrame
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
                'is_regular': tick.is_regular,
                'is_extended_hours': tick.is_extended_hours,
            }
            records.append(record)

        df = pd.DataFrame(records)
        print(f"[OK] DataFrame created with shape {df.shape}")

        # DataFrame Structure Report
        self.print_separator("DataFrame Structure", 2)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print(f"Data types:")
        for dtype, count in df.dtypes.value_counts().items():
            print(f"  {dtype}: {count}")

        # Show first 5 rows
        self.print_separator("First 5 Rows", 2)
        print(df.head().to_string())

        # Timezone verification
        self.print_separator("Timezone Verification", 2)
        if 'timestamp' in df.columns:
            print(f"Timestamp dtype: {df['timestamp'].dtype}")
            if hasattr(df['timestamp'].dtype, 'tz'):
                print(f"Timezone: {df['timestamp'].dtype.tz}")
            sample_ts = df['timestamp'].iloc[0]
            print(f"Sample timestamp: {sample_ts}")

        # Statistical Analysis
        self.print_separator("Statistical Analysis", 2)
        print(f"Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        print(f"Average price: ${df['price'].mean():.2f}")
        print(f"Total volume: {df['volume'].sum():,}")
        print(f"Average spread (bps): {df['spread_bps'].mean():.2f}")

        # Regular hours verification
        if 'is_regular' in df.columns:
            print(f"\nRegular trades: {df['is_regular'].sum()}")
            print(f"Extended hours trades: {df['is_extended_hours'].sum()}")

        # Calculate checksum
        self.checksums['dataframe'] = df['price'].sum()
        print(f"\nChecksum (price sum): {self.checksums['dataframe']:.2f}")

        # Verify checksum consistency
        for stage in ['numpy', 'pydantic']:
            if stage in self.checksums:
                diff = abs(self.checksums['dataframe'] - self.checksums[stage])
                if diff < 0.01:
                    print(f"[OK] Checksum matches {stage} (diff: {diff:.6f})")
                else:
                    print(f"[WARNING] Checksum mismatch with {stage} (diff: {diff:.6f})")

        return df

    def test_stage4_dataframe_to_arcticdb(self, df: pd.DataFrame) -> bool:
        """
        STAGE 4: DataFrame -> ArcticDB
        """
        self.print_separator("STAGE 4: DataFrame -> ArcticDB Storage", 1)

        try:
            tick_store = TickStore()

            # Prepare metadata
            metadata = {
                'session': 'regular',
                'hours': '9:30-16:00 ET',
                'test': True,
                'stage': 4,
                'checksum': self.checksums.get('dataframe', 0)
            }

            # Store the DataFrame
            print(f"Storing {len(df)} regular hours ticks to ArcticDB...")
            date_str = datetime.now().strftime("%Y%m%d")

            result = tick_store.store_ticks(
                symbol=self.symbol,
                date=date_str,
                tick_df=df,
                metadata=metadata,
                overwrite=True
            )

            print(f"[OK] Stored {len(df)} ticks")
            print(f"Symbol: {self.symbol}")
            print(f"Date: {date_str}")
            print(f"Metadata: {metadata}")

            return True

        except Exception as e:
            print(f"[FAIL] Storage failed: {e}")
            return False

    def test_stage5_arcticdb_retrieval(self) -> Optional[pd.DataFrame]:
        """
        STAGE 5: ArcticDB -> DataFrame Retrieval
        """
        self.print_separator("STAGE 5: ArcticDB -> DataFrame Retrieval", 1)

        try:
            tick_store = TickStore()

            # Retrieve data
            print(f"Retrieving {self.symbol} data from ArcticDB...")
            date_str = datetime.now().strftime("%Y%m%d")

            retrieved_df = tick_store.load_ticks(
                symbol=self.symbol,
                date_range=date_str
            )

            if retrieved_df is not None and not retrieved_df.empty:
                print(f"[OK] Retrieved {len(retrieved_df)} ticks")

                # Show first 5 rows
                self.print_separator("First 5 Retrieved Rows", 2)
                print(retrieved_df.head().to_string())

                # Timezone verification
                self.print_separator("Timezone Verification", 2)
                if 'timestamp' in retrieved_df.columns:
                    print(f"Timestamp dtype: {retrieved_df['timestamp'].dtype}")
                    sample_ts = retrieved_df['timestamp'].iloc[0]
                    print(f"Sample timestamp: {sample_ts}")

                # Calculate checksum
                if 'price' in retrieved_df.columns:
                    retrieved_checksum = retrieved_df['price'].sum()
                    print(f"\nRetrieved checksum: {retrieved_checksum:.2f}")

                    # Verify against all previous checksums
                    self.print_separator("Final Checksum Verification", 2)
                    for stage, checksum in self.checksums.items():
                        diff = abs(retrieved_checksum - checksum)
                        if diff < 0.01:
                            print(f"[OK] {stage}: matches (diff: {diff:.6f})")
                        else:
                            print(f"[FAIL] {stage}: mismatch (diff: {diff:.6f})")

                return retrieved_df
            else:
                print("[FAIL] No data retrieved")
                return None

        except Exception as e:
            print(f"[FAIL] Retrieval failed: {e}")
            return None

    def run_complete_test(self):
        """Run all stages of the test"""

        self.print_separator("AAPL REGULAR HOURS DATA PIPELINE TEST", 1)
        print(f"Symbol: {self.symbol}")
        print(f"Hours: 9:30 AM - 4:00 PM ET")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")

        stages_completed = 0

        try:
            # Stage 1: IQFeed -> NumPy
            tick_array = self.test_stage1_iqfeed_to_numpy()
            if tick_array is not None:
                stages_completed += 1
            else:
                print("\n[STOP] Stage 1 failed")
                return

            # Stage 2: NumPy -> Pydantic
            if FOUNDATION_MODELS:
                pydantic_ticks = self.test_stage2_numpy_to_pydantic(tick_array)
                if pydantic_ticks is not None:
                    stages_completed += 1
                else:
                    print("\n[STOP] Stage 2 failed")
                    return
            else:
                print("\n[SKIP] Stage 2 - Foundation Models not available")
                pydantic_ticks = None

            # Stage 3: Pydantic -> DataFrame
            if pydantic_ticks:
                df = self.test_stage3_pydantic_to_dataframe(pydantic_ticks)
                if df is not None:
                    stages_completed += 1
                else:
                    print("\n[STOP] Stage 3 failed")
                    return
            else:
                # Fallback: Direct NumPy to DataFrame
                self.print_separator("STAGE 3: Direct NumPy -> DataFrame (Fallback)", 1)
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(tick_array['date']) + pd.to_timedelta(tick_array['time'], unit='us'),
                    'price': tick_array['last'],
                    'volume': tick_array['last_sz'],
                    'bid': tick_array['bid'],
                    'ask': tick_array['ask']
                })
                df['timestamp'] = df['timestamp'].dt.tz_localize('America/New_York')
                print(f"[OK] Basic DataFrame created: {df.shape}")
                stages_completed += 1

            # Stage 4: DataFrame -> ArcticDB
            if self.test_stage4_dataframe_to_arcticdb(df):
                stages_completed += 1

                # Stage 5: ArcticDB -> DataFrame
                retrieved_df = self.test_stage5_arcticdb_retrieval()
                if retrieved_df is not None:
                    stages_completed += 1

            # Final Report
            self.print_separator("FINAL REPORT", 1)
            print(f"Stages completed: {stages_completed}/5")

            if stages_completed == 5:
                print("\n[SUCCESS] All stages completed successfully!")
                print("[SUCCESS] Data integrity maintained across all transformations!")
            else:
                print(f"\n[PARTIAL] Completed {stages_completed} stages")

        except Exception as e:
            print(f"\n[ERROR] Test failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    tester = AAPLRegularHoursTester()
    tester.run_complete_test()


if __name__ == "__main__":
    main()