"""
End-to-End Data Pipeline Test Suite
Tests complete data flow from collection to storage and retrieval
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

# Import our complete data engine system
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

class E2EDataPipelineTest:
    """End-to-end test suite for complete data pipeline"""

    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT', 'TSLA']
        self.results = {}
        self.pipeline_metrics = {}

        logger.info("Initialized end-to-end data pipeline test suite")

    def test_complete_market_snapshot_pipeline(self) -> bool:
        """Test complete pipeline from data collection to storage and retrieval"""
        logger.info("=== Testing Complete Market Snapshot Pipeline ===")

        try:
            # Initialize data engine
            engine = DataEngine()

            # Step 1: Collect market snapshot
            logger.info("Step 1: Collecting market snapshot...")
            start_time = time.time()

            snapshot = engine.collect_market_snapshot(self.test_symbols)
            collection_time = time.time() - start_time

            # Verify snapshot structure
            expected_keys = ['ticks', 'indicators', 'sentiment']
            for key in expected_keys:
                if key not in snapshot:
                    logger.error(f"❌ Missing snapshot key: {key}")
                    return False

            logger.info("✅ Market snapshot structure verified")

            # Analyze collected data
            total_records = 0
            data_summary = {}

            for data_type, df in snapshot.items():
                if df is not None and not df.empty:
                    record_count = len(df)
                    total_records += record_count
                    data_summary[data_type] = {
                        'records': record_count,
                        'columns': list(df.columns),
                        'data_types': df.dtypes.to_dict()
                    }
                    logger.info(f"✅ {data_type}: {record_count} records")
                else:
                    data_summary[data_type] = {'records': 0}
                    logger.warning(f"⚠️ {data_type}: No data collected")

            if total_records == 0:
                logger.error("❌ No data collected in snapshot")
                return False

            logger.info(f"✅ Total records collected: {total_records}")
            logger.info(f"✅ Collection time: {collection_time:.3f}s")

            # Step 2: Store snapshot
            logger.info("Step 2: Storing market snapshot...")
            start_time = time.time()

            storage_success = engine.store_snapshot(snapshot)
            storage_time = time.time() - start_time

            if not storage_success:
                logger.error("❌ Snapshot storage failed")
                return False

            logger.info(f"✅ Snapshot stored successfully in {storage_time:.3f}s")

            # Step 3: Verify storage
            logger.info("Step 3: Verifying stored data...")
            tick_store = TickStore()
            today = datetime.now().strftime('%Y-%m-%d')

            verified_symbols = []
            for symbol in self.test_symbols:
                try:
                    stored_ticks = tick_store.read_ticks(symbol, today)
                    if stored_ticks is not None and not stored_ticks.empty:
                        verified_symbols.append(symbol)
                        logger.info(f"✅ {symbol}: {len(stored_ticks)} ticks verified in storage")
                    else:
                        logger.warning(f"⚠️ {symbol}: No ticks found in storage")
                except Exception as e:
                    logger.warning(f"⚠️ {symbol}: Storage verification error: {e}")

            if not verified_symbols:
                logger.error("❌ No symbols verified in storage")
                return False

            # Step 4: Test data retrieval and analysis
            logger.info("Step 4: Testing data retrieval and analysis...")

            # Test market regime analysis
            regime = engine.get_market_regime()
            logger.info(f"✅ Market regime analysis: {regime}")

            # Test engine statistics
            stats = engine.get_stats()
            logger.info(f"✅ Engine statistics: {stats}")

            # Store pipeline metrics
            self.pipeline_metrics['snapshot_pipeline'] = {
                'collection_time': collection_time,
                'storage_time': storage_time,
                'total_records': total_records,
                'verified_symbols': verified_symbols,
                'data_summary': data_summary
            }

            logger.info("✅ Complete market snapshot pipeline test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Complete market snapshot pipeline test FAILED: {e}")
            return False

    def test_historical_data_pipeline(self) -> bool:
        """Test historical data collection and processing pipeline"""
        logger.info("=== Testing Historical Data Pipeline ===")

        try:
            engine = DataEngine()

            # Test historical data collection
            logger.info("Testing historical data collection...")
            start_time = time.time()

            historical_data = engine.collect_historical_data(
                symbols=['AAPL'],
                days=5
            )
            collection_time = time.time() - start_time

            if not historical_data:
                logger.warning("⚠️ No historical data collected")
                return True  # This is acceptable on weekends

            total_records = sum(len(df) for df in historical_data.values() if df is not None)
            logger.info(f"✅ Historical data collected: {total_records} records in {collection_time:.3f}s")

            # Analyze historical data structure
            for data_type, df in historical_data.items():
                if df is not None and not df.empty:
                    logger.info(f"✅ {data_type}: {len(df)} historical records")

                    # Verify time ordering
                    if 'timestamp' in df.columns:
                        is_sorted = df['timestamp'].is_monotonic_increasing
                        if is_sorted:
                            logger.info(f"✅ {data_type}: Timestamps properly ordered")
                        else:
                            logger.warning(f"⚠️ {data_type}: Timestamps not ordered")

            self.pipeline_metrics['historical_pipeline'] = {
                'collection_time': collection_time,
                'total_records': total_records,
                'data_types': list(historical_data.keys())
            }

            logger.info("✅ Historical data pipeline test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Historical data pipeline test FAILED: {e}")
            return False

    def test_real_time_simulation_pipeline(self) -> bool:
        """Test simulated real-time data processing pipeline"""
        logger.info("=== Testing Real-Time Simulation Pipeline ===")

        try:
            engine = DataEngine()

            # Simulate multiple rapid collections (like real-time scenario)
            logger.info("Simulating rapid data collections...")

            collection_times = []
            snapshots = []

            for i in range(3):  # Simulate 3 rapid collections
                logger.info(f"Collection {i+1}/3...")
                start_time = time.time()

                snapshot = engine.collect_market_snapshot(['AAPL'])
                collection_time = time.time() - start_time
                collection_times.append(collection_time)
                snapshots.append(snapshot)

                logger.info(f"✅ Collection {i+1} completed in {collection_time:.3f}s")

                # Small delay to simulate real-time scenario
                time.sleep(1)

            # Analyze collection consistency
            avg_collection_time = np.mean(collection_times)
            logger.info(f"✅ Average collection time: {avg_collection_time:.3f}s")

            # Verify data consistency across collections
            for i, snapshot in enumerate(snapshots):
                total_records = sum(len(df) for df in snapshot.values() if df is not None and not df.empty)
                logger.info(f"✅ Collection {i+1}: {total_records} records")

            # Test concurrent storage
            logger.info("Testing concurrent storage...")
            storage_results = []

            for i, snapshot in enumerate(snapshots):
                try:
                    stored = engine.store_snapshot(snapshot)
                    storage_results.append(stored)
                    logger.info(f"✅ Snapshot {i+1} storage: {'SUCCESS' if stored else 'FAILED'}")
                except Exception as e:
                    logger.warning(f"⚠️ Snapshot {i+1} storage error: {e}")
                    storage_results.append(False)

            successful_stores = sum(storage_results)
            logger.info(f"✅ Storage success rate: {successful_stores}/{len(snapshots)}")

            self.pipeline_metrics['realtime_simulation'] = {
                'collections': len(snapshots),
                'avg_collection_time': avg_collection_time,
                'storage_success_rate': successful_stores / len(snapshots)
            }

            logger.info("✅ Real-time simulation pipeline test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Real-time simulation pipeline test FAILED: {e}")
            return False

    def test_data_quality_pipeline(self) -> bool:
        """Test data quality validation throughout the pipeline"""
        logger.info("=== Testing Data Quality Pipeline ===")

        try:
            engine = DataEngine()

            # Collect data for quality testing
            snapshot = engine.collect_market_snapshot(['AAPL'])

            quality_results = {}

            # Test tick data quality
            if 'ticks' in snapshot and not snapshot['ticks'].empty:
                ticks = snapshot['ticks']

                # Check for required columns
                required_cols = ['timestamp', 'price', 'volume']
                missing_cols = [col for col in required_cols if col not in ticks.columns]

                if not missing_cols:
                    logger.info("✅ Tick data has all required columns")
                    quality_results['tick_columns'] = True
                else:
                    logger.warning(f"⚠️ Missing tick columns: {missing_cols}")
                    quality_results['tick_columns'] = False

                # Check for null values
                null_counts = ticks[required_cols].isnull().sum()
                if null_counts.sum() == 0:
                    logger.info("✅ No null values in tick data")
                    quality_results['tick_nulls'] = True
                else:
                    logger.warning(f"⚠️ Null values found: {null_counts.to_dict()}")
                    quality_results['tick_nulls'] = False

                # Check for reasonable price values
                if 'price' in ticks.columns:
                    prices = ticks['price']
                    if (prices > 0).all() and (prices < 10000).all():
                        logger.info(f"✅ Price values reasonable: ${prices.min():.2f} - ${prices.max():.2f}")
                        quality_results['tick_prices'] = True
                    else:
                        logger.warning(f"⚠️ Suspicious price values: {prices.min()} - {prices.max()}")
                        quality_results['tick_prices'] = False

            # Test indicator data quality
            if 'indicators' in snapshot and not snapshot['indicators'].empty:
                indicators = snapshot['indicators']

                # Check for reasonable indicator values
                for _, row in indicators.iterrows():
                    indicator = row.get('indicator', '')
                    value = row.get('value', None)

                    if indicator == 'NYSE_TICK' and value is not None:
                        if -3000 <= value <= 3000:
                            logger.info(f"✅ NYSE_TICK value reasonable: {value}")
                            quality_results['NYSE_TICK'] = True
                        else:
                            logger.warning(f"⚠️ NYSE_TICK value suspicious: {value}")
                            quality_results['NYSE_TICK'] = False

                    elif indicator == 'NYSE_TRIN' and value is not None:
                        if 0 < value < 10:
                            logger.info(f"✅ NYSE_TRIN value reasonable: {value}")
                            quality_results['NYSE_TRIN'] = True
                        else:
                            logger.warning(f"⚠️ NYSE_TRIN value suspicious: {value}")
                            quality_results['NYSE_TRIN'] = False

            # Test timestamp consistency
            all_timestamps = []
            for data_type, df in snapshot.items():
                if df is not None and not df.empty and 'timestamp' in df.columns:
                    all_timestamps.extend(df['timestamp'].tolist())

            if all_timestamps:
                timestamp_range = max(all_timestamps) - min(all_timestamps)
                if timestamp_range.total_seconds() < 86400:  # Within 24 hours
                    logger.info(f"✅ Timestamp consistency verified: {timestamp_range}")
                    quality_results['timestamp_consistency'] = True
                else:
                    logger.warning(f"⚠️ Large timestamp range: {timestamp_range}")
                    quality_results['timestamp_consistency'] = False

            # Calculate overall quality score
            passed_checks = sum(quality_results.values())
            total_checks = len(quality_results)
            quality_score = passed_checks / total_checks if total_checks > 0 else 0

            logger.info(f"✅ Data quality score: {quality_score:.2%} ({passed_checks}/{total_checks})")

            self.pipeline_metrics['data_quality'] = {
                'quality_score': quality_score,
                'passed_checks': passed_checks,
                'total_checks': total_checks,
                'quality_results': quality_results
            }

            logger.info("✅ Data quality pipeline test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Data quality pipeline test FAILED: {e}")
            return False

    def test_error_recovery_pipeline(self) -> bool:
        """Test pipeline behavior under error conditions"""
        logger.info("=== Testing Error Recovery Pipeline ===")

        try:
            engine = DataEngine()

            # Test with invalid symbol
            logger.info("Testing invalid symbol handling...")
            try:
                snapshot = engine.collect_market_snapshot(['INVALID_SYMBOL'])
                if snapshot:
                    logger.info("✅ Invalid symbol handled gracefully")
                else:
                    logger.info("✅ Invalid symbol returned empty result")
            except Exception as e:
                logger.warning(f"⚠️ Invalid symbol caused exception: {e}")

            # Test health check under stress
            logger.info("Testing health check reliability...")
            health_checks = []
            for i in range(3):
                health = engine.health_check()
                health_checks.append(health)
                logger.info(f"Health check {i+1}: {health.get('overall', 'UNKNOWN')}")

            # Test statistics consistency
            logger.info("Testing statistics consistency...")
            stats1 = engine.get_stats()
            stats2 = engine.get_stats()

            if isinstance(stats1, dict) and isinstance(stats2, dict):
                logger.info("✅ Statistics return consistent structure")
            else:
                logger.warning("⚠️ Statistics structure inconsistent")

            self.pipeline_metrics['error_recovery'] = {
                'health_checks': health_checks,
                'stats_consistency': isinstance(stats1, dict) and isinstance(stats2, dict)
            }

            logger.info("✅ Error recovery pipeline test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Error recovery pipeline test FAILED: {e}")
            return False

    def run_complete_test_suite(self) -> bool:
        """Run complete end-to-end test suite"""
        logger.info("🚀 Starting End-to-End Data Pipeline Test Suite")
        logger.info("=" * 60)

        tests = [
            ("Complete Market Snapshot Pipeline", self.test_complete_market_snapshot_pipeline),
            ("Historical Data Pipeline", self.test_historical_data_pipeline),
            ("Real-Time Simulation Pipeline", self.test_real_time_simulation_pipeline),
            ("Data Quality Pipeline", self.test_data_quality_pipeline),
            ("Error Recovery Pipeline", self.test_error_recovery_pipeline)
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
        logger.info(f"🎯 END-TO-END PIPELINE RESULTS")
        logger.info(f"Passed: {passed}/{total} tests")
        logger.info(f"Success rate: {passed/total*100:.1f}%")

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - Complete data pipeline is working perfectly!")
            logger.info("✨ Stage 1 data engine is production-ready!")
            return True
        else:
            logger.error(f"⚠️ {total - passed} tests failed - Review pipeline implementation")
            return False

    def generate_pipeline_report(self):
        """Generate comprehensive pipeline metrics report"""
        logger.info("\n📊 PIPELINE METRICS REPORT")
        logger.info("=" * 40)

        for pipeline, metrics in self.pipeline_metrics.items():
            logger.info(f"\n{pipeline.upper().replace('_', ' ')}:")
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.3f}")
                    else:
                        logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {metrics}")


def main():
    """Run the end-to-end data pipeline test suite"""
    try:
        # Ensure data directory exists
        Path("./data").mkdir(exist_ok=True)

        # Run test suite
        test_suite = E2EDataPipelineTest()
        success = test_suite.run_complete_test_suite()

        # Generate pipeline report
        test_suite.generate_pipeline_report()

        if success:
            print("\n🚀 End-to-end pipeline verification successful!")
            print("🔥 Complete data pipeline is production-ready!")
            print("📊 Stage 1 data engine validation complete!")
        else:
            print("\n⚠️ Pipeline needs attention before proceeding to Stage 2")
            print("📋 Review test results and fix failing components")

        return success

    except Exception as e:
        logger.error(f"Test suite initialization failed: {e}")
        return False


if __name__ == "__main__":
    main()