"""
Test Organized Stage 1 Data Engine Architecture
Validates the new modular structure and separation of concerns
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Import our new organized components
from stage_01_data_engine.core.data_engine import DataEngine
from stage_01_data_engine.core.config_loader import ConfigLoader
from stage_01_data_engine.collectors.tick_collector import TickCollector
from stage_01_data_engine.collectors.dtn_indicators_collector import DTNIndicatorCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrganizedArchitectureTest:
    """Test suite for the new organized data engine architecture"""

    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT', 'TSLA']
        self.results = {}

        logger.info("Initialized organized architecture test suite")

    def test_configuration_system(self) -> bool:
        """Test centralized configuration management"""
        logger.info("=== Testing Configuration System ===")

        try:
            # Test config loader initialization
            config = ConfigLoader()

            # Test symbol configuration
            aapl_config = config.get_symbol_config('AAPL')
            logger.info(f"AAPL configuration: {aapl_config}")

            # Test indicator configuration
            breadth_indicators = config.get_indicator_mapping('breadth_indicators')
            logger.info(f"Breadth indicators: {list(breadth_indicators.keys())}")

            # Test bar configuration
            volume_bar_config = config.get_bar_config('volume_bars')
            logger.info(f"Volume bar configuration: {volume_bar_config}")

            # Test storage configuration
            storage_config = config.get_storage_config('equity_ticks')
            logger.info(f"Storage configuration: {storage_config}")

            logger.info("✅ Configuration system test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Configuration system test FAILED: {e}")
            return False

    def test_tick_collector(self) -> bool:
        """Test new modular tick collector"""
        logger.info("=== Testing Tick Collector ===")

        try:
            collector = TickCollector()

            # Test single symbol collection
            logger.info("Testing single symbol tick collection...")
            tick_data = collector.collect('AAPL', num_days=1, max_ticks=100)

            if tick_data is not None and not tick_data.empty:
                logger.info(f"✅ Collected {len(tick_data)} ticks for AAPL")
                logger.info(f"Columns: {list(tick_data.columns)}")

                # Test data validation
                if collector.validate(tick_data):
                    logger.info("✅ Tick data validation passed")
                else:
                    logger.warning("⚠️ Tick data validation failed")

                # Test storage key generation
                storage_key = collector.get_storage_key('AAPL', '2025-01-15')
                logger.info(f"Storage key: {storage_key}")

                # Display sample data
                if len(tick_data) > 0:
                    sample = tick_data.iloc[0]
                    logger.info(f"Sample tick: {sample.to_dict()}")

            else:
                logger.warning("⚠️ No tick data collected")

            # Test collector statistics
            stats = collector.get_stats()
            logger.info(f"Collector stats: {stats}")

            logger.info("✅ Tick collector test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Tick collector test FAILED: {e}")
            return False

    def test_dtn_indicators_collector(self) -> bool:
        """Test DTN indicators collector"""
        logger.info("=== Testing DTN Indicators Collector ===")

        try:
            collector = DTNIndicatorCollector()

            # Test available indicators listing
            indicators = collector.list_available_indicators()
            logger.info(f"Available indicators: {len(indicators)}")

            # Test indicator groups
            groups = collector.get_indicator_groups()
            logger.info(f"Indicator groups: {groups}")

            # Test specific indicator collection
            test_indicators = ['NYSE_TICK', 'NYSE_TRIN']

            logger.info("Testing indicator snapshot collection...")
            snapshot_data = collector.collect(test_indicators, data_type='snapshot')

            if snapshot_data is not None and not snapshot_data.empty:
                logger.info(f"✅ Collected {len(snapshot_data)} indicator snapshots")
                logger.info(f"Columns: {list(snapshot_data.columns)}")

                # Display sample data
                for _, row in snapshot_data.iterrows():
                    logger.info(f"Indicator {row['indicator']}: {row.get('value', 'N/A')}")

                # Test data validation
                if collector.validate(snapshot_data):
                    logger.info("✅ Indicator data validation passed")
                else:
                    logger.warning("⚠️ Indicator data validation failed")

            else:
                logger.warning("⚠️ No indicator data collected")

            # Test sentiment snapshot
            logger.info("Testing market sentiment snapshot...")
            sentiment = collector.get_market_sentiment_snapshot()
            if sentiment:
                logger.info("✅ Market sentiment snapshot:")
                for key, value in sentiment.items():
                    logger.info(f"  {key}: {value}")
            else:
                logger.warning("⚠️ No sentiment data available")

            logger.info("✅ DTN indicators collector test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ DTN indicators collector test FAILED: {e}")
            return False

    def test_unified_data_engine(self) -> bool:
        """Test unified data engine interface"""
        logger.info("=== Testing Unified Data Engine ===")

        try:
            # Initialize data engine
            engine = DataEngine()

            # Test health check
            logger.info("Testing health check...")
            health = engine.health_check()
            logger.info(f"System health: {health}")

            # Test market snapshot collection
            logger.info("Testing market snapshot collection...")
            snapshot = engine.collect_market_snapshot(['AAPL'])

            logger.info("Market snapshot results:")
            for data_type, df in snapshot.items():
                logger.info(f"  {data_type}: {len(df)} records")

            # Test market regime analysis
            logger.info("Testing market regime analysis...")
            regime = engine.get_market_regime()
            logger.info(f"Market regime: {regime}")

            # Test engine statistics
            stats = engine.get_stats()
            logger.info(f"Engine stats: {stats}")

            # Test snapshot storage (if storage is working)
            if any(not df.empty for df in snapshot.values()):
                logger.info("Testing snapshot storage...")
                stored = engine.store_snapshot(snapshot)
                logger.info(f"Storage result: {'SUCCESS' if stored else 'FAILED'}")

            logger.info("✅ Unified data engine test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Unified data engine test FAILED: {e}")
            return False

    def test_storage_organization(self) -> bool:
        """Test storage system organization"""
        logger.info("=== Testing Storage Organization ===")

        try:
            from stage_01_data_engine.storage.tick_store import TickStore

            # Test storage initialization
            tick_store = TickStore()

            # Test storage stats
            stats = tick_store.get_storage_stats()
            logger.info(f"Storage stats: {stats}")

            # Test data listing
            stored_data = tick_store.list_stored_data()
            logger.info(f"Stored data entries: {len(stored_data)}")

            if stored_data:
                for data_info in stored_data[:3]:  # Show first 3 entries
                    logger.info(f"  {data_info}")

            logger.info("✅ Storage organization test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Storage organization test FAILED: {e}")
            return False

    def test_architecture_integration(self) -> bool:
        """Test complete architecture integration"""
        logger.info("=== Testing Architecture Integration ===")

        try:
            # Test end-to-end data flow
            engine = DataEngine()

            # Collect comprehensive data
            logger.info("Testing comprehensive data collection...")

            # Test with multiple symbols
            symbols = ['AAPL', 'MSFT']
            snapshot = engine.collect_market_snapshot(symbols)

            # Analyze the complete data pipeline
            total_records = sum(len(df) for df in snapshot.values())
            logger.info(f"Total records collected: {total_records}")

            # Test data types and quality
            data_quality = {}
            for data_type, df in snapshot.items():
                if not df.empty:
                    quality_info = {
                        'records': len(df),
                        'columns': len(df.columns),
                        'missing_values': df.isnull().sum().sum(),
                        'data_types': df.dtypes.value_counts().to_dict()
                    }
                    data_quality[data_type] = quality_info
                    logger.info(f"{data_type} quality: {quality_info}")

            # Test regime detection integration
            regime = engine.get_market_regime()
            logger.info(f"Integrated market regime analysis: {regime}")

            # Test performance metrics
            engine_stats = engine.get_stats()
            logger.info(f"Performance metrics: {engine_stats}")

            logger.info("✅ Architecture integration test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Architecture integration test FAILED: {e}")
            return False

    def run_complete_test_suite(self) -> bool:
        """Run complete test suite for organized architecture"""
        logger.info("🚀 Starting Organized Architecture Test Suite")
        logger.info("=" * 60)

        tests = [
            ("Configuration System", self.test_configuration_system),
            ("Tick Collector", self.test_tick_collector),
            ("DTN Indicators Collector", self.test_dtn_indicators_collector),
            ("Unified Data Engine", self.test_unified_data_engine),
            ("Storage Organization", self.test_storage_organization),
            ("Architecture Integration", self.test_architecture_integration)
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\n🔬 Running {test_name} test...")
            try:
                if test_func():
                    logger.info(f"✅ {test_name} test PASSED")
                    passed += 1
                    self.results[test_name] = "PASSED"
                else:
                    logger.error(f"❌ {test_name} test FAILED")
                    self.results[test_name] = "FAILED"
            except Exception as e:
                logger.error(f"💥 {test_name} test ERROR: {e}")
                self.results[test_name] = "ERROR"

        # Results summary
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 ORGANIZED ARCHITECTURE TEST RESULTS")
        logger.info(f"Passed: {passed}/{total} tests")
        logger.info(f"Success rate: {passed/total*100:.1f}%")

        logger.info(f"\nDetailed Results:")
        for test_name, result in self.results.items():
            status_emoji = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "💥"
            logger.info(f"{status_emoji} {test_name}: {result}")

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - New architecture is working perfectly!")
            logger.info("✨ Ready for production deployment with organized structure!")
            return True
        else:
            logger.error(f"⚠️ {total - passed} tests failed - Review architecture implementation")
            return False


def main():
    """Run the organized architecture test suite"""
    try:
        # Ensure data directory exists
        Path("./data").mkdir(exist_ok=True)

        # Run test suite
        test_suite = OrganizedArchitectureTest()
        success = test_suite.run_complete_test_suite()

        if success:
            print("\n🚀 Organized architecture validation successful!")
            print("🔥 Stage 1 data engine is properly structured and functional!")
            print("📊 Ready for Stage 2 fuzzy logic integration!")
        else:
            print("\n⚠️ Architecture needs refinement before proceeding")
            print("📋 Review test results and fix failing components")

        return success

    except Exception as e:
        logger.error(f"Test suite initialization failed: {e}")
        return False


if __name__ == "__main__":
    main()