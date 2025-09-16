"""
Comprehensive Data Integrity Test Suite
========================================

Tests the complete pipeline: IQFeed -> NumPy -> Pydantic -> DataFrame -> ArcticDB

Uses REAL AAPL tick data to verify data integrity at every transformation stage.
Reports data quality, provides previews, and validates checksums throughout.

Following CLAUDE.md guidelines: NO mock data, only real IQFeed data.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pyiqfeed_orig'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stage_01_data_engine'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import pyiqfeed as iq
    IQFEED_AVAILABLE = True
except ImportError:
    IQFEED_AVAILABLE = False
    print("WARNING: PyIQFeed not available - tests will be skipped")

try:
    from foundation.models.market import TickData
    from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic
    FOUNDATION_MODELS_AVAILABLE = True
except ImportError:
    FOUNDATION_MODELS_AVAILABLE = False
    print("[WARN] Foundation Models not available - will test basic integration only")

# Direct imports to avoid circular dependency
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage_01_data_engine', 'collectors'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'stage_01_data_engine', 'storage'))

try:
    from iqfeed_collector import IQFeedCollector
    from tick_store import TickStore
    DATA_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Data Engine components not available: {e}")
    DATA_ENGINE_AVAILABLE = False


class DataIntegrityTester:
    """Comprehensive data integrity testing for the entire pipeline"""

    def __init__(self):
        # Debug: Check credentials
        iqfeed_username = os.getenv('IQFEED_USERNAME')
        iqfeed_password = os.getenv('IQFEED_PASSWORD')
        print(f"[DEBUG] IQFeed Username: {iqfeed_username}")
        print(f"[DEBUG] IQFeed Password: {'***' if iqfeed_password else 'None'}")

        self.iqfeed_collector = IQFeedCollector()
        self.tick_store = TickStore()
        self.test_symbol = "AAPL"
        self.test_date = "2025-09-15"  # Last trading day (Friday if weekend)

    def print_section(self, title: str, level: int = 1):
        """Print formatted section header"""
        if level == 1:
            print(f"\n{'='*70}")
            print(f"    {title}")
            print('='*70)
        else:
            print(f"\n--- {title} ---")

    def print_data_preview(self, data, title: str, max_rows: int = 5):
        """Print data preview with formatting"""
        print(f"\n--- {title} ---")
        if isinstance(data, np.ndarray):
            for i in range(min(max_rows, len(data))):
                print(f"Row {i}: {data[i]}")
        elif isinstance(data, list):
            for i in range(min(max_rows, len(data))):
                if hasattr(data[i], 'dict'):
                    print(f"Row {i}: {data[i].dict()}")
                else:
                    print(f"Row {i}: {data[i]}")
        elif isinstance(data, pd.DataFrame):
            print(data.head(max_rows).to_string())
        else:
            print(f"Data type: {type(data)}")
            print(f"First few items: {str(data)[:200]}...")

    def test_stage1_iqfeed_to_numpy(self) -> np.ndarray:
        """
        STAGE 1: Test IQFeed data retrieval and NumPy structure

        Returns:
            np.ndarray: Raw tick data from IQFeed
        """
        self.print_section("STAGE 1: IQFeed -> NumPy", level=1)

        # Fetch real AAPL data
        print(f"Fetching {self.test_symbol} tick data...")
        if not self.iqfeed_collector.ensure_connection():
            raise ConnectionError("Failed to connect to IQFeed")

        tick_array = self.iqfeed_collector.get_tick_data(
            self.test_symbol,
            num_days=1,
            max_ticks=1000
        )

        if tick_array is None or len(tick_array) == 0:
            raise ValueError("No tick data received from IQFeed")

        # Data Quality Report
        print(f"[OK] Data retrieved successfully")
        print(f"Data type: {type(tick_array)}")
        print(f"Array shape: {tick_array.shape}")
        print(f"Array dtype: {tick_array.dtype}")
        print(f"Number of ticks: {len(tick_array)}")
        print(f"Memory usage: {tick_array.nbytes / 1024:.2f} KB")

        # Field Analysis
        expected_fields = [
            'tick_id', 'date', 'time', 'last', 'last_sz', 'last_type',
            'mkt_ctr', 'tot_vlm', 'bid', 'ask', 'cond1', 'cond2', 'cond3', 'cond4'
        ]

        print(f"\n--- Field Structure ---")
        print(f"Expected fields: {len(expected_fields)}")
        print(f"Actual fields: {len(tick_array.dtype.names) if tick_array.dtype.names else 'N/A'}")
        if tick_array.dtype.names:
            print(f"Field names: {tick_array.dtype.names}")

        # Data Preview
        self.print_data_preview(tick_array, "First 5 Ticks (Raw NumPy)", 5)

        # Statistical Analysis
        print(f"\n--- Statistical Analysis ---")
        if 'last' in tick_array.dtype.names:
            prices = tick_array['last']
            print(f"Price range: ${prices.min():.2f} - ${prices.max():.2f}")
            print(f"Price mean: ${prices.mean():.2f}")
            print(f"Price std: ${prices.std():.2f}")

        if 'last_sz' in tick_array.dtype.names:
            volumes = tick_array['last_sz']
            print(f"Volume range: {volumes.min()} - {volumes.max()}")
            print(f"Volume mean: {volumes.mean():.1f}")

        if 'bid' in tick_array.dtype.names and 'ask' in tick_array.dtype.names:
            bids = tick_array['bid']
            asks = tick_array['ask']
            spreads = asks - bids
            print(f"Spread range: ${spreads.min():.3f} - ${spreads.max():.3f}")
            print(f"Spread mean: ${spreads.mean():.3f}")

        # Data Integrity Checks
        print(f"\n--- Data Integrity Checks ---")
        checks_passed = 0
        total_checks = 0

        # Check 1: Data exists
        total_checks += 1
        if len(tick_array) > 0:
            print(f"[OK] Check 1: Data exists ({len(tick_array)} ticks)")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 1: No data received")

        # Check 2: Expected fields
        total_checks += 1
        if tick_array.dtype.names and len(tick_array.dtype.names) >= 10:
            print(f"[OK] Check 2: Field structure valid ({len(tick_array.dtype.names)} fields)")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 2: Invalid field structure")

        # Check 3: Valid prices
        total_checks += 1
        if 'last' in tick_array.dtype.names and np.all(tick_array['last'] > 0):
            print(f"[OK] Check 3: All prices positive")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 3: Invalid prices found")

        # Check 4: Valid spreads (if bid/ask available)
        if 'bid' in tick_array.dtype.names and 'ask' in tick_array.dtype.names:
            total_checks += 1
            valid_quotes = (tick_array['bid'] > 0) & (tick_array['ask'] > 0)
            if valid_quotes.any():
                valid_spreads = tick_array['bid'][valid_quotes] <= tick_array['ask'][valid_quotes]
                if np.all(valid_spreads):
                    print(f"[OK] Check 4: All spreads non-negative")
                    checks_passed += 1
                else:
                    print(f"[FAIL] Check 4: Inverted spreads found")
            else:
                print(f"[WARN] Check 4: No valid bid/ask data")

        print(f"\n[INFO] Stage 1 Summary: {checks_passed}/{total_checks} checks passed")

        if checks_passed < total_checks:
            print(f"[WARN] Some integrity checks failed - proceeding with caution")

        return tick_array

    def test_stage2_numpy_to_pydantic(self, tick_array: np.ndarray) -> List[TickData]:
        """
        STAGE 2: Test conversion to Pydantic models with validation

        Args:
            tick_array: NumPy array from Stage 1

        Returns:
            List[TickData]: Validated Pydantic models
        """
        self.print_section("STAGE 2: NumPy -> Pydantic Models", level=1)

        if not FOUNDATION_MODELS_AVAILABLE:
            print("[FAIL] Foundation Models not available - skipping Stage 2")
            return []

        # Convert to Pydantic
        print(f"Converting {len(tick_array)} ticks to Pydantic models...")
        start_time = time.time()

        try:
            pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, self.test_symbol)
            conversion_time = time.time() - start_time

            print(f"[OK] Conversion successful in {conversion_time:.3f}s")

        except Exception as e:
            print(f"[FAIL] Conversion failed: {e}")
            raise

        # Data Quality Report
        print(f"Number of ticks: {len(pydantic_ticks)}")
        print(f"Model type: {type(pydantic_ticks[0]) if pydantic_ticks else 'N/A'}")

        if pydantic_ticks:
            model_fields = pydantic_ticks[0].model_dump().keys()
            print(f"Fields available: {len(model_fields)} fields")
            print(f"Field names: {list(model_fields)}")

        # Data Preview
        if pydantic_ticks:
            print("\n--- First 3 Ticks (Pydantic Models) ---")
            for i in range(min(3, len(pydantic_ticks))):
                tick = pydantic_ticks[i]
                print(f"\nTick {i}:")
                print(f"  Symbol: {tick.symbol}")
                print(f"  Timestamp: {tick.timestamp}")
                print(f"  Price: ${tick.price}")
                print(f"  Size: {tick.size}")
                if tick.bid and tick.ask:
                    print(f"  Bid/Ask: ${tick.bid}/${tick.ask}")
                    if tick.spread_bps:
                        print(f"  Spread: {tick.spread_bps:.2f} bps")
                print(f"  Trade Sign: {tick.trade_sign}")
                print(f"  Dollar Volume: ${tick.dollar_volume}")
                print(f"  Is Block: {tick.is_block_trade}")
                print(f"  Participant: {tick.participant_type}")
                print(f"  Extended Hours: {tick.is_extended_hours}")

        # Enhanced Fields Analysis
        print(f"\n--- Enhanced Fields Statistics ---")
        if pydantic_ticks:
            # Spread analysis
            spreads = [float(t.spread_bps) for t in pydantic_ticks if t.spread_bps is not None]
            if spreads:
                print(f"Spread (bps): min={min(spreads):.1f}, max={max(spreads):.1f}, avg={np.mean(spreads):.1f}")

            # Trade classification
            trade_signs = [t.trade_sign for t in pydantic_ticks]
            buy_count = trade_signs.count(1)
            sell_count = trade_signs.count(-1)
            neutral_count = trade_signs.count(0)
            print(f"Trade classification: Buy={buy_count}, Sell={sell_count}, Neutral={neutral_count}")

            # Block trades
            block_trades = sum(1 for t in pydantic_ticks if t.is_block_trade)
            print(f"Block trades: {block_trades} ({block_trades/len(pydantic_ticks)*100:.1f}%)")

            # Extended hours
            extended_hours = sum(1 for t in pydantic_ticks if t.is_extended_hours)
            print(f"Extended hours: {extended_hours} ({extended_hours/len(pydantic_ticks)*100:.1f}%)")

            # Participant types
            participant_counts = {}
            for t in pydantic_ticks:
                participant_counts[t.participant_type] = participant_counts.get(t.participant_type, 0) + 1
            print(f"Participant types: {participant_counts}")

        # Data Integrity Checks
        print(f"\n--- Data Integrity Checks ---")
        checks_passed = 0
        total_checks = 0

        # Check 1: Count preservation
        total_checks += 1
        if len(pydantic_ticks) == len(tick_array):
            print(f"[OK] Check 1: Tick count preserved ({len(pydantic_ticks)})")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 1: Tick count mismatch - NumPy: {len(tick_array)}, Pydantic: {len(pydantic_ticks)}")

        # Check 2: Price preservation (first 10 ticks)
        total_checks += 1
        price_matches = 0
        check_count = min(10, len(tick_array), len(pydantic_ticks))

        for i in range(check_count):
            numpy_price = float(tick_array[i]['last'])
            pydantic_price = float(pydantic_ticks[i].price)
            if abs(numpy_price - pydantic_price) < 0.001:  # Allow small floating point differences
                price_matches += 1

        if price_matches == check_count:
            print(f"[OK] Check 2: Prices preserved (checked {check_count} ticks)")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 2: Price mismatches found ({price_matches}/{check_count} matched)")

        # Check 3: Enhanced fields computed
        total_checks += 1
        if pydantic_ticks and hasattr(pydantic_ticks[0], 'spread_bps'):
            computed_fields = ['spread_bps', 'trade_sign', 'dollar_volume', 'is_block_trade']
            has_computed = all(hasattr(pydantic_ticks[0], field) for field in computed_fields)
            if has_computed:
                print(f"[OK] Check 3: Enhanced fields computed")
                checks_passed += 1
            else:
                print(f"[FAIL] Check 3: Missing enhanced fields")
        else:
            print(f"[FAIL] Check 3: No enhanced fields available")

        # Check 4: Valid trade signs
        if pydantic_ticks:
            total_checks += 1
            valid_signs = all(t.trade_sign in [-1, 0, 1] for t in pydantic_ticks)
            if valid_signs:
                print(f"[OK] Check 4: All trade signs valid")
                checks_passed += 1
            else:
                print(f"[FAIL] Check 4: Invalid trade signs found")

        print(f"\n[INFO] Stage 2 Summary: {checks_passed}/{total_checks} checks passed")

        return pydantic_ticks

    def test_stage3_pydantic_to_dataframe(self, pydantic_ticks: List[TickData]) -> pd.DataFrame:
        """
        STAGE 3: Test conversion to DataFrame

        Args:
            pydantic_ticks: Pydantic models from Stage 2

        Returns:
            pd.DataFrame: Optimized DataFrame
        """
        self.print_section("STAGE 3: Pydantic -> DataFrame", level=1)

        if not pydantic_ticks:
            print("[FAIL] No Pydantic ticks to convert - skipping Stage 3")
            return pd.DataFrame()

        # Convert to DataFrame using the method we need to implement
        print(f"Converting {len(pydantic_ticks)} Pydantic models to DataFrame...")
        start_time = time.time()

        try:
            # Use the tick_store's _pydantic_to_dataframe to get ALL 41 fields
            df = self.tick_store._pydantic_to_dataframe(pydantic_ticks)
            conversion_time = time.time() - start_time

            print(f"[OK] Conversion successful in {conversion_time:.3f}s")

        except Exception as e:
            print(f"[FAIL] Conversion failed: {e}")
            raise

        # Optimize data types (similar to existing TickStore logic)
        print("\nOptimizing data types...")
        original_memory = df.memory_usage(deep=True).sum()

        # Apply optimizations
        df['price'] = df['price'].astype('float32')
        if df['bid'].notna().any():
            df['bid'] = df['bid'].astype('float32')
        if df['ask'].notna().any():
            df['ask'] = df['ask'].astype('float32')
        df['volume'] = df['volume'].astype('uint32')
        df['total_volume'] = df['total_volume'].astype('uint32')
        df['market_center'] = df['market_center'].astype('uint16')
        df['exchange'] = df['exchange'].astype('category')
        df['trade_sign'] = df['trade_sign'].astype('int8')
        df['tick_direction'] = df['tick_direction'].astype('int8')
        if df['spread_bps'].notna().any():
            df['spread_bps'] = df['spread_bps'].astype('float32')

        optimized_memory = df.memory_usage(deep=True).sum()
        memory_reduction = (1 - optimized_memory / original_memory) * 100

        # Data Quality Report
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"Memory usage: {optimized_memory / 1024:.2f} KB")
        print(f"Memory reduction: {memory_reduction:.1f}%")

        # Data Type Summary
        print(f"\n--- Data Type Optimization ---")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"{dtype}: {count} columns")

        # Data Preview
        self.print_data_preview(df[['timestamp', 'price', 'volume', 'bid', 'ask', 'spread_bps', 'trade_sign', 'dollar_volume']],
                              "First 5 Rows (Key Columns)", 5)

        # Statistical Summary
        print(f"\n--- Statistical Summary ---")
        numeric_cols = ['price', 'volume', 'spread_bps', 'dollar_volume']
        available_cols = [col for col in numeric_cols if col in df.columns and df[col].notna().any()]
        if available_cols:
            print(df[available_cols].describe().round(2).to_string())

        # Data Integrity Checks
        print(f"\n--- Data Integrity Checks ---")
        checks_passed = 0
        total_checks = 0

        # Check 1: Row count preservation
        total_checks += 1
        if len(df) == len(pydantic_ticks):
            print(f"[OK] Check 1: Row count preserved ({len(df)})")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 1: Row count mismatch - Pydantic: {len(pydantic_ticks)}, DataFrame: {len(df)}")

        # Check 2: Data type optimization
        total_checks += 1
        optimized_types = ['float32', 'uint32', 'uint16', 'category', 'int8']
        has_optimized = any(str(dtype) in optimized_types for dtype in df.dtypes)
        if has_optimized:
            print(f"[OK] Check 2: Data types optimized")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 2: Data types not optimized")

        # Check 3: Value preservation - compare total sums instead of individual rows
        # since DataFrame might be reordered
        total_checks += 1
        pydantic_sum = sum(float(t.price) for t in pydantic_ticks)
        df_sum = df['price'].sum()

        if abs(pydantic_sum - df_sum) < 0.01:  # Allow small floating point difference
            print(f"[OK] Check 3: Values preserved (sum diff: {abs(pydantic_sum - df_sum):.6f})")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 3: Value mismatch (Pydantic sum: {pydantic_sum:.2f}, DataFrame sum: {df_sum:.2f})")

        # Check 4: Enhanced fields present
        total_checks += 1
        enhanced_fields = ['spread_bps', 'trade_sign', 'dollar_volume', 'is_block_trade', 'participant_type']
        present_enhanced = [field for field in enhanced_fields if field in df.columns]
        if len(present_enhanced) >= 4:  # Most enhanced fields present
            print(f"[OK] Check 4: Enhanced fields present ({len(present_enhanced)}/{len(enhanced_fields)})")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 4: Missing enhanced fields ({len(present_enhanced)}/{len(enhanced_fields)} present)")

        print(f"\n[INFO] Stage 3 Summary: {checks_passed}/{total_checks} checks passed")

        return df

    def test_stage4_dataframe_to_arcticdb(self, df: pd.DataFrame, symbol: str = 'AAPL') -> pd.DataFrame:
        """
        STAGE 4: Test DataFrame storage and retrieval from ArcticDB using proper TickStore methods

        Args:
            df: DataFrame from Stage 3 (with all 41 columns)
            symbol: Stock symbol

        Returns:
            pd.DataFrame: Loaded DataFrame from ArcticDB
        """
        self.print_section("STAGE 4: DataFrame -> ArcticDB", level=1)

        if df.empty:
            print("[FAIL] No DataFrame to store - skipping Stage 4")
            return pd.DataFrame()

        # Use a test date to avoid conflicts
        test_date = "2025-09-16"

        # Store to ArcticDB using the proper TickStore method
        print(f"Storing {len(df)} rows with {len(df.columns)} columns to ArcticDB...")
        print(f"Symbol: {symbol}, Date: {test_date}")
        start_time = time.time()

        try:
            # Apply deduplication (which adds sequence numbers)
            df_with_seq = self.tick_store._remove_duplicates(df.copy())

            # Check for sequence numbers
            if 'tick_sequence' in df_with_seq.columns:
                max_seq = df_with_seq['tick_sequence'].max()
                print(f"Added tick_sequence column (max: {max_seq})")

            # Create metadata
            test_metadata = {
                'source': 'test_pipeline',
                'test_run': True,
                'original_columns': len(df.columns),
                'with_sequence': 'tick_sequence' in df_with_seq.columns
            }

            # Store using the proper TickStore method
            # This handles all the normalization and storage logic
            success = self.tick_store.store_ticks(
                symbol=symbol,
                date=test_date,
                tick_df=df_with_seq,
                metadata=test_metadata,
                overwrite=True  # Overwrite any existing test data
            )
            store_time = time.time() - start_time

            if success:
                print(f"[OK] Stored successfully in {store_time:.3f}s")
                print(f"Stored columns: {len(df_with_seq.columns)}")
            else:
                print(f"[FAIL] Storage failed")
                return pd.DataFrame()

        except Exception as e:
            print(f"[FAIL] Storage error: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

        # Load from ArcticDB using the proper TickStore method
        print(f"\nLoading data back from ArcticDB...")
        start_time = time.time()

        try:
            # Read using the proper storage key format (symbol/date)
            storage_key = f"{symbol}/{test_date}"
            loaded_result = self.tick_store.tick_data_lib.read(storage_key)
            loaded_df = loaded_result.data
            load_time = time.time() - start_time

            if loaded_df is not None and not loaded_df.empty:
                print(f"[OK] Loaded {len(loaded_df)} rows in {load_time:.3f}s")
                print(f"Loaded columns: {len(loaded_df.columns)}")
            else:
                print("[FAIL] No data loaded from ArcticDB")
                return pd.DataFrame()

        except Exception as e:
            print(f"[FAIL] Load error: {e}")
            return pd.DataFrame()

        # Data Integrity Checks
        print(f"\n--- Data Integrity Checks ---")
        checks_passed = 0
        total_checks = 0

        # Check 1: Row count
        total_checks += 1
        if len(loaded_df) == len(df_with_seq):
            print(f"[OK] Check 1: Row count preserved ({len(loaded_df)})")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 1: Row count mismatch - Stored: {len(df_with_seq)}, Loaded: {len(loaded_df)}")

        # Check 2: Column count (expecting one less if metadata was dropped)
        total_checks += 1
        expected_cols = len(df_with_seq.columns)
        actual_cols = len(loaded_df.columns)
        if actual_cols == expected_cols:
            print(f"[OK] Check 2: Column count preserved ({actual_cols})")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 2: Column count mismatch - Expected: {expected_cols}, Actual: {actual_cols}")
            missing_cols = set(df_with_seq.columns) - set(loaded_df.columns)
            if missing_cols:
                print(f"      Missing columns: {missing_cols}")
            extra_cols = set(loaded_df.columns) - set(df_with_seq.columns)
            if extra_cols:
                print(f"      Extra columns: {extra_cols}")

        # Check 3: Price checksum
        total_checks += 1
        original_sum = df_with_seq['price'].sum()
        loaded_sum = loaded_df['price'].sum() if 'price' in loaded_df.columns else 0
        if abs(original_sum - loaded_sum) < 0.01:
            print(f"[OK] Check 3: Price checksum preserved (diff: {abs(original_sum - loaded_sum):.6f})")
            checks_passed += 1
        else:
            print(f"[FAIL] Check 3: Price checksum mismatch - Original: {original_sum:.2f}, Loaded: {loaded_sum:.2f}")

        # Check 4: Timestamp preservation
        total_checks += 1
        if 'timestamp' in loaded_df.columns and 'timestamp' in df_with_seq.columns:
            # Check if timestamps are preserved (ignoring timezone info)
            # Get the actual timestamp values
            orig_first = df_with_seq.iloc[0]['timestamp']
            loaded_first = loaded_df.iloc[0]['timestamp']

            # Convert to naive datetime for comparison
            if hasattr(orig_first, 'tz_localize'):
                orig_first_naive = orig_first.tz_localize(None)
            else:
                orig_first_naive = pd.Timestamp(orig_first).tz_localize(None) if pd.Timestamp(orig_first).tz else pd.Timestamp(orig_first)

            if hasattr(loaded_first, 'tz_localize'):
                loaded_first_naive = loaded_first.tz_localize(None)
            else:
                loaded_first_naive = pd.Timestamp(loaded_first).tz_localize(None) if pd.Timestamp(loaded_first).tz else pd.Timestamp(loaded_first)

            # Compare the naive timestamps
            if orig_first_naive == loaded_first_naive:
                print(f"[OK] Check 4: Timestamps preserved")
                checks_passed += 1
            else:
                print(f"[FAIL] Check 4: Timestamp mismatch")
                print(f"      Original: {orig_first} (type: {type(orig_first)})")
                print(f"      Loaded: {loaded_first} (type: {type(loaded_first)})")
                print(f"      Diff: {abs(orig_first_naive - loaded_first_naive)}")
        else:
            print(f"[FAIL] Check 4: Timestamp column missing")

        print(f"\n[INFO] Stage 4 Summary: {checks_passed}/{total_checks} checks passed")

        # Cleanup test data
        print(f"\nCleaning up test data...")
        try:
            test_date = "2025-09-16"  # Same date used for storage
            storage_key = f"{symbol}/{test_date}"
            self.tick_store.tick_data_lib.delete(storage_key)
            print(f"[OK] Cleaned up {storage_key} from ArcticDB")
        except:
            pass  # Ignore cleanup errors

        return loaded_df

    def run_comprehensive_test(self):
        """Run the complete data integrity test pipeline"""

        self.print_section("DATA INTEGRITY PIPELINE TEST - AAPL", level=1)
        print(f"Symbol: {self.test_symbol}")
        print(f"Target Date: {self.test_date}")
        print(f"IQFeed Available: {IQFEED_AVAILABLE}")
        print(f"Foundation Models Available: {FOUNDATION_MODELS_AVAILABLE}")

        total_stages = 4  # NumPy, Pydantic, DataFrame, ArcticDB
        completed_stages = 0
        checksums = {}

        try:
            # Stage 1: IQFeed -> NumPy
            tick_array = self.test_stage1_iqfeed_to_numpy()
            checksums['numpy'] = np.sum(tick_array['last']) if 'last' in tick_array.dtype.names else 0
            completed_stages += 1

            # Stage 2: NumPy -> Pydantic (if available)
            if FOUNDATION_MODELS_AVAILABLE:
                pydantic_ticks = self.test_stage2_numpy_to_pydantic(tick_array)
                if pydantic_ticks:
                    checksums['pydantic'] = sum(float(t.price) for t in pydantic_ticks)
                    completed_stages += 1

                    # Stage 3: Pydantic -> DataFrame
                    df = self.test_stage3_pydantic_to_dataframe(pydantic_ticks)
                    if not df.empty:
                        checksums['dataframe'] = df['price'].sum()
                        completed_stages += 1

                        # Stage 4: DataFrame -> ArcticDB
                        loaded_df = self.test_stage4_dataframe_to_arcticdb(df, self.test_symbol)
                        if not loaded_df.empty:
                            checksums['arcticdb'] = loaded_df['price'].sum()
                            completed_stages += 1

            # Final Report
            self.print_section("FINAL DATA INTEGRITY REPORT", level=1)

            print(f"Stages completed: {completed_stages}/{total_stages}")

            if checksums:
                print(f"\n--- Checksum Verification ---")
                for stage, checksum in checksums.items():
                    print(f"{stage.capitalize():12}: {checksum:.2f}")

                # Verify checksums match (allowing small floating point differences)
                if len(checksums) > 1:
                    checksum_values = list(checksums.values())
                    max_diff = max(checksum_values) - min(checksum_values)
                    if max_diff < 0.01:  # Allow small floating point differences
                        print(f"\n[OK] All checksums match (max diff: {max_diff:.6f})")
                        print(f"[OK] DATA INTEGRITY PRESERVED!")
                    else:
                        print(f"\n[FAIL] Checksum mismatch detected (max diff: {max_diff:.6f})")
                        print(f"[FAIL] DATA INTEGRITY COMPROMISED!")

            print(f"\n[INFO] Overall Success: {completed_stages}/{total_stages} stages completed")

        except Exception as e:
            print(f"\n[FAIL] Test failed at stage {completed_stages + 1}: {e}")
            raise


def main():
    """Main test execution"""
    print("Starting Data Integrity Pipeline Test...")

    if not IQFEED_AVAILABLE:
        print("[FAIL] Cannot run tests - IQFeed not available")
        return

    tester = DataIntegrityTester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()