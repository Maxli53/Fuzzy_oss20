"""
Data Loading Verification Test Suite
Tests that all data collection mechanisms actually work with real market data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time

# Import our data engine components
from stage_01_data_engine.core.data_engine import DataEngine
from stage_01_data_engine.collectors.tick_collector import TickCollector
from stage_01_data_engine.collectors.dtn_indicators_collector import DTNIndicatorCollector
from stage_01_data_engine.storage.tick_store import TickStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataLoadingTest:
    """Test suite to verify all data loading mechanisms work correctly"""

    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT', 'TSLA']
        self.results = {}
        self.data_samples = {}

        logger.info("Initialized data loading test suite")

    def test_iqfeed_connection(self) -> bool:
        """Test IQFeed connection and authentication"""
        logger.info("=== Testing IQFeed Connection ===")

        try:
            from pyiqfeed import get_level_1_quotes_conn, get_historical_conn

            # Test Level 1 connection
            logger.info("Testing Level 1 quotes connection...")
            quotes_conn = get_level_1_quotes_conn(name="test_l1")
            if quotes_conn:
                logger.info("✅ Level 1 connection successful")
                quotes_conn.disconnect()
            else:
                logger.error("❌ Level 1 connection failed")
                return False

            # Test Historical connection
            logger.info("Testing historical data connection...")
            hist_conn = get_historical_conn(name="test_hist")
            if hist_conn:
                logger.info("✅ Historical connection successful")
                hist_conn.disconnect()
            else:
                logger.error("❌ Historical connection failed")
                return False

            logger.info("✅ IQFeed connection test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ IQFeed connection test FAILED: {e}")
            return False

    def test_tick_data_loading(self) -> bool:
        """Test tick data collection for multiple symbols"""
        logger.info("=== Testing Tick Data Loading ===")

        try:
            collector = TickCollector()
            collected_data = {}

            for symbol in self.test_symbols:
                logger.info(f"Testing tick collection for {symbol}...")

                # Test single symbol collection
                tick_data = collector.collect([symbol], num_days=1, max_ticks=500)

                if tick_data is not None and not tick_data.empty:
                    collected_data[symbol] = tick_data
                    logger.info(f"✅ {symbol}: Collected {len(tick_data)} ticks")

                    # Store sample for reporting
                    self.data_samples[f"{symbol}_ticks"] = tick_data.head(3).to_dict('records')

                    # Test data validation
                    if collector.validate(tick_data):
                        logger.info(f"✅ {symbol}: Data validation passed")
                    else:
                        logger.warning(f"⚠️ {symbol}: Data validation failed")

                else:
                    logger.error(f"❌ {symbol}: No tick data collected")
                    return False

            # Test multi-symbol collection
            logger.info("Testing multi-symbol collection...")
            multi_ticks = collector.collect(self.test_symbols, num_days=1, max_ticks=1000)

            if multi_ticks is not None and not multi_ticks.empty:
                logger.info(f"✅ Multi-symbol: Collected {len(multi_ticks)} total ticks")

                # Check all symbols present
                symbols_found = multi_ticks['symbol'].unique()
                missing_symbols = set(self.test_symbols) - set(symbols_found)
                if missing_symbols:
                    logger.warning(f"⚠️ Missing symbols in multi-collection: {missing_symbols}")
                else:
                    logger.info("✅ All symbols present in multi-collection")
            else:
                logger.error("❌ Multi-symbol collection failed")
                return False

            self.results['tick_data'] = collected_data
            logger.info("✅ Tick data loading test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Tick data loading test FAILED: {e}")
            return False

    def test_dtn_indicators_loading(self) -> bool:
        """Test DTN indicators collection"""
        logger.info("=== Testing DTN Indicators Loading ===")

        try:
            collector = DTNIndicatorCollector()

            # Test key market indicators
            key_indicators = ['NYSE_TICK', 'NYSE_TRIN', 'TOTAL_PC_RATIO', 'NYSE_ADD']

            logger.info("Testing snapshot collection...")
            indicator_data = collector.collect(key_indicators, data_type='snapshot')

            if indicator_data is not None and not indicator_data.empty:
                logger.info(f"✅ Collected {len(indicator_data)} indicator snapshots")

                # Validate specific indicators
                for _, row in indicator_data.iterrows():
                    indicator = row['indicator']
                    value = row.get('value', None)

                    if indicator == 'NYSE_TICK':
                        if -3000 <= value <= 3000:
                            logger.info(f"✅ NYSE_TICK = {value} (valid range)")
                        else:
                            logger.warning(f"⚠️ NYSE_TICK = {value} (outside typical range)")

                    elif indicator == 'NYSE_TRIN':
                        if value > 0:
                            logger.info(f"✅ NYSE_TRIN = {value} (valid)")
                        else:
                            logger.warning(f"⚠️ NYSE_TRIN = {value} (invalid)")

                    elif indicator == 'TOTAL_PC_RATIO':
                        if 0.1 <= value <= 5.0:
                            logger.info(f"✅ TOTAL_PC_RATIO = {value} (valid range)")
                        else:
                            logger.warning(f"⚠️ TOTAL_PC_RATIO = {value} (outside typical range)")

                # Store sample for reporting
                self.data_samples['indicators'] = indicator_data.to_dict('records')

                # Test data validation
                if collector.validate(indicator_data):
                    logger.info("✅ Indicator data validation passed")
                else:
                    logger.warning("⚠️ Indicator data validation failed")

            else:
                logger.error("❌ No indicator data collected")
                return False

            # Test market sentiment snapshot
            logger.info("Testing market sentiment snapshot...")
            sentiment = collector.get_market_sentiment_snapshot()

            if sentiment:
                logger.info("✅ Market sentiment snapshot collected:")
                for key, value in sentiment.items():
                    logger.info(f"  {key}: {value}")
                self.data_samples['sentiment'] = sentiment
            else:
                logger.warning("⚠️ No sentiment data available")

            self.results['indicators'] = indicator_data
            logger.info("✅ DTN indicators loading test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ DTN indicators loading test FAILED: {e}")
            return False

    def test_fallback_strategies(self) -> bool:
        """Test all fallback strategies for tick collection"""
        logger.info("=== Testing Fallback Strategies ===")

        try:
            collector = TickCollector()

            # Test with a symbol that might not have tick data
            test_symbol = 'SPY'

            logger.info(f"Testing fallback strategies for {test_symbol}...")
            tick_data = collector.collect([test_symbol], num_days=1, max_ticks=100)

            if tick_data is not None and not tick_data.empty:
                logger.info(f"✅ Fallback successful: Collected {len(tick_data)} data points for {test_symbol}")

                # Check data source used
                stats = collector.get_stats()
                logger.info(f"Collector stats: {stats}")

                # Store sample
                self.data_samples[f"{test_symbol}_fallback"] = tick_data.head(3).to_dict('records')
            else:
                logger.error(f"❌ All fallback strategies failed for {test_symbol}")
                return False

            logger.info("✅ Fallback strategies test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Fallback strategies test FAILED: {e}")
            return False

    def test_data_engine_integration(self) -> bool:
        """Test unified data engine interface"""
        logger.info("=== Testing Data Engine Integration ===")

        try:
            engine = DataEngine()

            # Test health check
            logger.info("Testing health check...")
            health = engine.health_check()
            logger.info(f"System health: {health}")

            # Test market snapshot collection
            logger.info("Testing market snapshot collection...")
            snapshot = engine.collect_market_snapshot(['AAPL'])

            logger.info("Market snapshot results:")
            total_records = 0
            for data_type, df in snapshot.items():
                record_count = len(df) if df is not None else 0
                total_records += record_count
                logger.info(f"  {data_type}: {record_count} records")

                # Store samples
                if record_count > 0:
                    self.data_samples[f"snapshot_{data_type}"] = df.head(3).to_dict('records')

            if total_records > 0:
                logger.info(f"✅ Total records collected: {total_records}")
            else:
                logger.error("❌ No data collected in market snapshot")
                return False

            # Test market regime analysis
            logger.info("Testing market regime analysis...")
            regime = engine.get_market_regime()
            logger.info(f"Market regime: {regime}")
            self.data_samples['market_regime'] = regime

            # Test engine statistics
            stats = engine.get_stats()
            logger.info(f"Engine stats: {stats}")

            self.results['snapshot'] = snapshot
            logger.info("✅ Data engine integration test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Data engine integration test FAILED: {e}")
            return False

    def run_complete_test_suite(self) -> bool:
        """Run complete data loading verification test suite"""
        logger.info("🚀 Starting Data Loading Verification Test Suite")
        logger.info("=" * 60)

        tests = [
            ("IQFeed Connection", self.test_iqfeed_connection),
            ("Tick Data Loading", self.test_tick_data_loading),
            ("DTN Indicators Loading", self.test_dtn_indicators_loading),
            ("Fallback Strategies", self.test_fallback_strategies),
            ("Data Engine Integration", self.test_data_engine_integration)
        ]

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

        # Results summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 DATA LOADING VERIFICATION RESULTS")
        logger.info(f"Passed: {passed}/{total} tests")
        logger.info(f"Success rate: {passed/total*100:.1f}%")

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - Data loading is working perfectly!")
            logger.info("✨ Ready to proceed with data verification!")
            return True
        else:
            logger.error(f"⚠️ {total - passed} tests failed - Review data loading implementation")
            return False

    def generate_data_samples_report(self):
        """Generate a report of collected data samples"""
        logger.info("\n📊 DATA SAMPLES REPORT")
        logger.info("=" * 40)

        for data_type, samples in self.data_samples.items():
            logger.info(f"\n{data_type.upper()}:")
            if isinstance(samples, list):
                for i, sample in enumerate(samples[:3]):  # Show first 3 samples
                    logger.info(f"  Sample {i+1}: {sample}")
            elif isinstance(samples, dict):
                for key, value in samples.items():
                    logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {samples}")


def main():
    """Run the data loading verification test suite"""
    try:
        # Ensure data directory exists
        Path("./data").mkdir(exist_ok=True)

        # Run test suite
        test_suite = DataLoadingTest()
        success = test_suite.run_complete_test_suite()

        # Generate samples report
        test_suite.generate_data_samples_report()

        if success:
            print("\n🚀 Data loading verification successful!")
            print("🔥 All data collection mechanisms are working!")
            print("📊 Ready for storage verification tests!")
        else:
            print("\n⚠️ Data loading needs attention before proceeding")
            print("📋 Review test results and fix failing components")

        return success

    except Exception as e:
        logger.error(f"Test suite initialization failed: {e}")
        return False


if __name__ == "__main__":
    main()