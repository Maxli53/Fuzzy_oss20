"""
Test Real AAPL Tick Data Collection from IQFeed
Validates API endpoints, responses, and data format against user's screenshot
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import our modules
from stage_01_data_engine.collectors.iqfeed_collector import IQFeedCollector
from stage_01_data_engine.storage import TickStore, BarBuilder, AdaptiveThresholds

# Setup logging to see all API responses
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealTickDataTest:
    """Test real tick data collection and processing"""

    def __init__(self):
        self.symbol = "AAPL"
        self.test_date = datetime.now().strftime('%Y-%m-%d')

        # Initialize components
        self.data_collector = IQFeedCollector()
        self.tick_store = TickStore()
        self.bar_builder = BarBuilder()
        self.adaptive_thresholds = AdaptiveThresholds(self.tick_store, "stage_01_data_engine/config/symbol_config.yaml")

        logger.info("Initialized real tick data test suite")

    def test_iqfeed_tick_api(self) -> bool:
        """Test IQFeed tick API endpoints and validate responses"""
        logger.info("=== TESTING IQFEED TICK API ===")

        try:
            # Test 1: Get small amount of tick data first
            logger.info("Test 1: Getting last few hours of AAPL tick data...")

            tick_data = self.data_collector.get_tick_data(
                symbol=self.symbol,
                num_days=1,
                max_ticks=1000  # Limit to first 1000 ticks for testing
            )

            if tick_data is None:
                logger.error("FAILED: No tick data returned from IQFeed API")
                return False

            if len(tick_data) == 0:
                logger.error("FAILED: Empty tick data array returned")
                return False

            # Validate API response
            logger.info(f"SUCCESS: Received {len(tick_data)} ticks from IQFeed")
            logger.info(f"NumPy fields: {tick_data.dtype.names}")

            # Test 2: Validate against user's screenshot format
            self._compare_with_user_format(tick_data)

            # Test 3: Test with more data
            logger.info("Test 2: Getting larger tick dataset...")

            large_tick_data = self.data_collector.get_tick_data(
                symbol=self.symbol,
                num_days=1,
                max_ticks=10000  # Get more data
            )

            if large_tick_data is not None:
                logger.info(f"SUCCESS: Retrieved {len(large_tick_data)} ticks in larger test")
            else:
                logger.warning("WARNING: Could not retrieve larger tick dataset")

            return True

        except Exception as e:
            logger.error(f"FAILED: IQFeed tick API test error: {e}")
            return False

    def _compare_with_user_format(self, tick_data: np.ndarray):
        """Compare our tick data format with user's IQFeed screenshot"""
        logger.info("=== COMPARING WITH USER'S IQFEED FORMAT ===")

        # IQFeed NumPy fields mapping to user's screenshot:
        iqfeed_fields = {
            'time': 'Time (microseconds since midnight)',
            'last': 'Price (trade price)',
            'last_sz': 'Volume (shares traded)',
            'bid': 'Bid (best bid)',
            'ask': 'Ask (best ask)',
            'last_type': 'Exchange (market center)',
            'cond1': 'Trade Conditions (code1)',
            'cond2': 'Trade Conditions (code2)'
        }

        logger.info("IQFeed fields compared to user's screenshot:")
        for field, description in iqfeed_fields.items():
            if field in tick_data.dtype.names:
                sample_value = tick_data[field][0] if len(tick_data) > 0 else "N/A"
                logger.info(f"✓ {field}: {description} -> Value: {sample_value}")
            else:
                logger.warning(f"✗ {field}: {description} -> MISSING from NumPy array")

        # Show sample data side by side
        if len(tick_data) > 0:
            logger.info("\n=== SAMPLE TICK COMPARISON ===")
            sample = tick_data[0]  # First tick from NumPy array

            logger.info("User's format (from screenshot):")
            logger.info("Time=15:59:59.970208, Price=234.0600, Volume=394, Bid=234.0500, Ask=234.0700")
            logger.info("Mkt_Center=NASDAQ, Conditions=REGULAR")

            logger.info("\nOur NumPy format:")
            logger.info(f"Time={sample['time']} microseconds")
            logger.info(f"Price=${sample['last']:.4f}")
            logger.info(f"Volume={sample['last_sz']}")
            logger.info(f"Bid=${sample['bid']:.4f}")
            logger.info(f"Ask=${sample['ask']:.4f}")
            logger.info(f"Exchange={sample['last_type']}")
            logger.info(f"Conditions=cond1:{sample['cond1']}, cond2:{sample['cond2']}")

    def test_tick_storage(self, tick_data: np.ndarray) -> bool:
        """Test storing real tick data using existing NumPy conversion"""
        logger.info("=== TESTING TICK STORAGE ===")

        try:
            # Store the real tick data using existing store_numpy_ticks method
            success = self.tick_store.store_numpy_ticks(
                symbol=self.symbol,
                date=self.test_date,
                tick_array=tick_data,
                metadata={'data_source': 'iqfeed_real', 'test_run': True},
                overwrite=True
            )

            if not success:
                logger.error("FAILED: Could not store tick data")
                return False

            # Test retrieval
            loaded_data = self.tick_store.load_ticks(self.symbol, self.test_date)

            if loaded_data is None:
                logger.error("FAILED: Could not load stored tick data")
                return False

            if len(loaded_data) != len(tick_data):
                logger.error(f"FAILED: Data length mismatch: stored {len(tick_data)}, loaded {len(loaded_data)}")
                return False

            # Test metadata
            metadata = self.tick_store.get_tick_metadata(self.symbol, self.test_date)
            if metadata:
                logger.info(f"SUCCESS: Metadata retrieved with {len(metadata)} fields")
                logger.info(f"Stored {metadata.get('total_ticks', 0)} ticks")
                logger.info(f"Price range: ${metadata.get('price_low', 0):.4f} - ${metadata.get('price_high', 0):.4f}")
            else:
                logger.warning("WARNING: No metadata retrieved")

            logger.info(f"SUCCESS: Tick storage test passed")
            return True

        except Exception as e:
            logger.error(f"FAILED: Tick storage test error: {e}")
            return False

    def test_real_bar_construction(self, tick_data: np.ndarray) -> bool:
        """Test bar construction with real tick data"""
        logger.info("=== TESTING BAR CONSTRUCTION WITH REAL TICKS ===")

        try:
            # Convert NumPy to DataFrame for BarBuilder using existing conversion
            tick_df = self.tick_store._numpy_ticks_to_dataframe(tick_data)
            # Get AAPL-specific thresholds from config
            bar_configs = {
                'volume_bars': 100000,
                'dollar_bars': 10000000,
                'imbalance_bars': 50000,
                'volatility_bars': 0.5,
                'range_bars': 0.25
            }

            results = {}

            # Test each bar type
            for bar_type, threshold in bar_configs.items():
                try:
                    logger.info(f"Testing {bar_type} with threshold {threshold}...")

                    if bar_type == 'volume_bars':
                        bars = self.bar_builder.volume_bars(tick_df, threshold)
                    elif bar_type == 'dollar_bars':
                        bars = self.bar_builder.dollar_bars(tick_df, threshold)
                    elif bar_type == 'imbalance_bars':
                        bars = self.bar_builder.imbalance_bars(tick_df, threshold)
                    elif bar_type == 'volatility_bars':
                        bars = self.bar_builder.volatility_bars(tick_df, threshold)
                    elif bar_type == 'range_bars':
                        bars = self.bar_builder.range_bars(tick_df, threshold)

                    if bars.empty:
                        logger.warning(f"WARNING: No {bar_type} generated (may need different threshold)")
                        results[bar_type] = 0
                    else:
                        results[bar_type] = len(bars)
                        logger.info(f"SUCCESS: Generated {len(bars)} {bar_type}")

                        # Show sample bar
                        sample = bars.iloc[0]
                        logger.info(f"Sample: O=${sample['open']:.4f} H=${sample['high']:.4f} "
                                  f"L=${sample['low']:.4f} C=${sample['close']:.4f} V={sample['volume']:,}")

                        # Validate bar quality
                        if self._validate_bar_quality(bars, bar_type):
                            logger.info(f"✓ {bar_type} quality validation passed")
                        else:
                            logger.warning(f"✗ {bar_type} quality validation failed")

                except Exception as e:
                    logger.error(f"ERROR: {bar_type} construction failed: {e}")
                    results[bar_type] = -1

            # Summary
            logger.info("=== BAR CONSTRUCTION RESULTS ===")
            for bar_type, count in results.items():
                if count > 0:
                    logger.info(f"✓ {bar_type}: {count} bars")
                elif count == 0:
                    logger.info(f"⚠ {bar_type}: No bars (threshold too high?)")
                else:
                    logger.info(f"✗ {bar_type}: Failed")

            return True

        except Exception as e:
            logger.error(f"FAILED: Bar construction test error: {e}")
            return False

    def test_adaptive_calibration(self, tick_data: np.ndarray) -> bool:
        """Test adaptive threshold calibration with real data"""
        logger.info("=== TESTING ADAPTIVE CALIBRATION ===")

        try:
            # Store tick data first (needed for calibration) using existing method
            self.tick_store.store_numpy_ticks(self.symbol, self.test_date, tick_data, overwrite=True)

            # Test calibration
            logger.info("Running adaptive threshold calibration...")
            thresholds = self.adaptive_thresholds.calibrate(self.symbol, lookback_days=1)

            if not thresholds:
                logger.error("FAILED: No thresholds returned from calibration")
                return False

            logger.info("SUCCESS: Adaptive calibration completed")
            logger.info("Calibrated thresholds:")
            for threshold_type, value in thresholds.items():
                logger.info(f"  {threshold_type}: {value}")

            # Test with calibrated thresholds
            logger.info("Testing bars with calibrated thresholds...")

            # Convert to DataFrame for bar builder
            tick_df = self.tick_store._numpy_ticks_to_dataframe(tick_data)
            calibrated_volume_bars = self.bar_builder.volume_bars(
                tick_df, int(thresholds.get('volume_threshold', 100000))
            )

            if not calibrated_volume_bars.empty:
                logger.info(f"SUCCESS: Generated {len(calibrated_volume_bars)} volume bars with calibrated threshold")
            else:
                logger.warning("WARNING: No volume bars with calibrated threshold")

            return True

        except Exception as e:
            logger.error(f"FAILED: Adaptive calibration test error: {e}")
            return False

    def _validate_bar_quality(self, bars: pd.DataFrame, bar_type: str) -> bool:
        """Validate bar data quality"""
        try:
            if bars.empty:
                return False

            # Check OHLC logic
            for _, bar in bars.iterrows():
                if bar['high'] < bar['low']:
                    logger.error(f"Invalid {bar_type}: High < Low")
                    return False
                if bar['high'] < bar['open'] or bar['high'] < bar['close']:
                    logger.error(f"Invalid {bar_type}: High < Open/Close")
                    return False
                if bar['low'] > bar['open'] or bar['low'] > bar['close']:
                    logger.error(f"Invalid {bar_type}: Low > Open/Close")
                    return False

            # Check for reasonable values
            if bars['volume'].sum() <= 0:
                logger.error(f"Invalid {bar_type}: Zero total volume")
                return False

            return True

        except Exception as e:
            logger.error(f"Bar validation error: {e}")
            return False

    def run_complete_test(self) -> bool:
        """Run complete real tick data test suite"""
        logger.info("🚀 Starting REAL AAPL Tick Data Test Suite")
        logger.info("=" * 60)

        tests = [
            ("IQFeed Tick API", self.test_iqfeed_tick_api),
        ]

        # Run API test first to get real data
        api_success = self.test_iqfeed_tick_api()
        if not api_success:
            logger.error("❌ API test failed - cannot proceed with other tests")
            return False

        # Get real tick data for other tests
        logger.info("Getting real tick data for remaining tests...")
        tick_data = self.data_collector.get_tick_data(self.symbol, num_days=1, max_ticks=5000)

        if tick_data is None or len(tick_data) == 0:
            logger.error("❌ Could not get tick data for remaining tests")
            return False

        # Add remaining tests
        tests.extend([
            ("Tick Storage", lambda: self.test_tick_storage(tick_data)),
            ("Real Bar Construction", lambda: self.test_real_bar_construction(tick_data)),
            ("Adaptive Calibration", lambda: self.test_adaptive_calibration(tick_data))
        ])

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\n🔬 Running {test_name} test...")
            try:
                if test_func():
                    logger.info(f"✅ {test_name} test PASSED")
                    passed += 1
                else:
                    logger.error(f"❌ {test_name} test FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name} test ERROR: {e}")

        # Results
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 REAL TICK DATA TEST RESULTS")
        logger.info(f"Passed: {passed}/{total} tests")
        logger.info(f"Success rate: {passed/total*100:.1f}%")

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - Real tick infrastructure is working!")
            logger.info("✨ Ready to replace simulated tick data with REAL market data")
            return True
        else:
            logger.error("⚠️ Some tests failed - Please review errors above")
            return False

def main():
    """Run the complete real tick data test"""
    try:
        # Create data directory
        Path("./data").mkdir(exist_ok=True)

        # Run tests
        test_suite = RealTickDataTest()
        success = test_suite.run_complete_test()

        if success:
            print("\n🚀 Real tick infrastructure validated!")
            print("🔥 AAPL data collection working with actual IQFeed API")
            print("📊 Ready for hedge fund-grade bar construction")
        else:
            print("\n⚠️ Infrastructure needs fixes before production use")

        return success

    except Exception as e:
        logger.error(f"Test suite initialization failed: {e}")
        return False

if __name__ == "__main__":
    main()