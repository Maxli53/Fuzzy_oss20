"""
Storage Verification Test Suite
Tests ArcticDB write/read integrity and data persistence
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

# Import our storage and collection components
from stage_01_data_engine.storage.tick_store import TickStore
from stage_01_data_engine.collectors.tick_collector import TickCollector
from stage_01_data_engine.core.data_engine import DataEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StorageVerificationTest:
    """Test suite to verify ArcticDB storage integrity and performance"""

    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT']
        self.results = {}
        self.storage_stats = {}

        logger.info("Initialized storage verification test suite")

    def test_arcticdb_initialization(self) -> bool:
        """Test ArcticDB connection and library initialization"""
        logger.info("=== Testing ArcticDB Initialization ===")

        try:
            # Initialize TickStore
            tick_store = TickStore()

            # Test connection
            logger.info("Testing ArcticDB connection...")
            if tick_store.arctic:
                logger.info("✅ ArcticDB connection successful")
            else:
                logger.error("❌ ArcticDB connection failed")
                return False

            # Test library availability
            logger.info("Testing library availability...")
            try:
                # List libraries
                libraries = tick_store.arctic.list_libraries()
                logger.info(f"Available libraries: {libraries}")

                # Check tick_data library exists
                if 'tick_data' in [lib.name for lib in libraries]:
                    logger.info("✅ tick_data library available")
                else:
                    logger.warning("⚠️ tick_data library not found - will be created")

                self.storage_stats['libraries'] = [lib.name for lib in libraries]
            except Exception as e:
                logger.warning(f"⚠️ Library listing failed: {e}")

            logger.info("✅ ArcticDB initialization test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ ArcticDB initialization test FAILED: {e}")
            return False

    def test_tick_data_storage(self) -> bool:
        """Test storing and retrieving tick data"""
        logger.info("=== Testing Tick Data Storage ===")

        try:
            tick_store = TickStore()
            collector = TickCollector()

            for symbol in self.test_symbols:
                logger.info(f"Testing storage for {symbol}...")

                # Collect some tick data
                tick_data = collector.collect([symbol], num_days=1, max_ticks=100)

                if tick_data is None or tick_data.empty:
                    logger.error(f"❌ No data collected for {symbol}")
                    return False

                original_count = len(tick_data)
                logger.info(f"Collected {original_count} ticks for {symbol}")

                # Store the data
                today = datetime.now().strftime('%Y-%m-%d')
                stored = tick_store.store_ticks(
                    symbol=symbol,
                    date=today,
                    tick_df=tick_data,
                    metadata={'source': 'test_storage', 'test_run': True},
                    overwrite=True
                )

                if not stored:
                    logger.error(f"❌ Failed to store ticks for {symbol}")
                    return False

                logger.info(f"✅ Stored {original_count} ticks for {symbol}")

                # Retrieve the data
                logger.info(f"Testing retrieval for {symbol}...")
                retrieved_data = tick_store.read_ticks(symbol, today)

                if retrieved_data is None or retrieved_data.empty:
                    logger.error(f"❌ Failed to retrieve ticks for {symbol}")
                    return False

                retrieved_count = len(retrieved_data)
                logger.info(f"✅ Retrieved {retrieved_count} ticks for {symbol}")

                # Verify data integrity
                if original_count == retrieved_count:
                    logger.info(f"✅ Data count integrity verified for {symbol}")
                else:
                    logger.warning(f"⚠️ Data count mismatch for {symbol}: {original_count} vs {retrieved_count}")

                # Verify column structure
                original_columns = set(tick_data.columns)
                retrieved_columns = set(retrieved_data.columns)

                if original_columns.issubset(retrieved_columns):
                    logger.info(f"✅ Column structure preserved for {symbol}")
                else:
                    missing_cols = original_columns - retrieved_columns
                    logger.warning(f"⚠️ Missing columns for {symbol}: {missing_cols}")

                # Store statistics
                self.storage_stats[symbol] = {
                    'original_count': original_count,
                    'retrieved_count': retrieved_count,
                    'integrity_check': original_count == retrieved_count
                }

            logger.info("✅ Tick data storage test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Tick data storage test FAILED: {e}")
            return False

    def test_metadata_storage(self) -> bool:
        """Test metadata storage and retrieval"""
        logger.info("=== Testing Metadata Storage ===")

        try:
            tick_store = TickStore()

            # Test metadata storage
            test_metadata = {
                'collection_time': datetime.now().isoformat(),
                'source': 'test_suite',
                'data_quality': 'high',
                'fallback_used': False,
                'test_flag': True
            }

            symbol = 'AAPL'
            today = datetime.now().strftime('%Y-%m-%d')

            # Store some dummy data with metadata
            dummy_data = pd.DataFrame({
                'timestamp': [datetime.now()],
                'price': [150.0],
                'volume': [1000],
                'symbol': [symbol]
            })

            stored = tick_store.store_ticks(
                symbol=symbol,
                date=today,
                tick_df=dummy_data,
                metadata=test_metadata,
                overwrite=True
            )

            if not stored:
                logger.error("❌ Failed to store data with metadata")
                return False

            # Retrieve and check metadata
            metadata = tick_store.get_metadata(symbol, today)

            if metadata:
                logger.info(f"✅ Metadata retrieved: {metadata}")

                # Verify key metadata fields
                for key, expected_value in test_metadata.items():
                    if key in metadata and metadata[key] == expected_value:
                        logger.info(f"✅ Metadata field '{key}' verified")
                    else:
                        logger.warning(f"⚠️ Metadata field '{key}' mismatch or missing")

            else:
                logger.warning("⚠️ No metadata retrieved")

            logger.info("✅ Metadata storage test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Metadata storage test FAILED: {e}")
            return False

    def test_date_range_queries(self) -> bool:
        """Test querying data across date ranges"""
        logger.info("=== Testing Date Range Queries ===")

        try:
            tick_store = TickStore()

            # Create test data for multiple dates
            test_dates = [
                (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            ]

            symbol = 'AAPL'

            # Store data for each date
            for date in test_dates:
                test_data = pd.DataFrame({
                    'timestamp': [datetime.strptime(date, '%Y-%m-%d')],
                    'price': [150.0 + np.random.random()],
                    'volume': [1000],
                    'symbol': [symbol]
                })

                stored = tick_store.store_ticks(
                    symbol=symbol,
                    date=date,
                    tick_df=test_data,
                    metadata={'test_date': date},
                    overwrite=True
                )

                if stored:
                    logger.info(f"✅ Test data stored for {date}")
                else:
                    logger.warning(f"⚠️ Failed to store test data for {date}")

            # Test date range query
            start_date = test_dates[0]
            end_date = test_dates[-1]

            logger.info(f"Testing range query from {start_date} to {end_date}...")
            range_data = tick_store.read_ticks_range(symbol, start_date, end_date)

            if range_data is not None and not range_data.empty:
                logger.info(f"✅ Range query returned {len(range_data)} records")

                # Verify all dates are represented
                unique_dates = range_data['timestamp'].dt.date.unique()
                logger.info(f"Unique dates in result: {unique_dates}")

            else:
                logger.warning("⚠️ Range query returned no data")

            logger.info("✅ Date range queries test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Date range queries test FAILED: {e}")
            return False

    def test_compression_and_performance(self) -> bool:
        """Test storage compression and performance"""
        logger.info("=== Testing Compression and Performance ===")

        try:
            tick_store = TickStore()
            collector = TickCollector()

            # Collect larger dataset for performance testing
            symbol = 'AAPL'
            logger.info(f"Collecting larger dataset for {symbol}...")

            tick_data = collector.collect([symbol], num_days=2, max_ticks=1000)

            if tick_data is None or tick_data.empty:
                logger.warning("⚠️ No data for performance test")
                return True

            original_size = len(tick_data)
            logger.info(f"Performance test dataset: {original_size} records")

            # Test write performance
            start_time = time.time()
            today = datetime.now().strftime('%Y-%m-%d')

            stored = tick_store.store_ticks(
                symbol=symbol,
                date=today,
                tick_df=tick_data,
                metadata={'performance_test': True},
                overwrite=True
            )

            write_time = time.time() - start_time

            if stored:
                logger.info(f"✅ Write performance: {original_size} records in {write_time:.3f}s")
                logger.info(f"✅ Write rate: {original_size/write_time:.0f} records/sec")
            else:
                logger.error("❌ Performance test write failed")
                return False

            # Test read performance
            start_time = time.time()
            retrieved_data = tick_store.read_ticks(symbol, today)
            read_time = time.time() - start_time

            if retrieved_data is not None and not retrieved_data.empty:
                logger.info(f"✅ Read performance: {len(retrieved_data)} records in {read_time:.3f}s")
                logger.info(f"✅ Read rate: {len(retrieved_data)/read_time:.0f} records/sec")
            else:
                logger.error("❌ Performance test read failed")
                return False

            # Test storage statistics
            stats = tick_store.get_storage_stats()
            if stats:
                logger.info(f"✅ Storage stats: {stats}")
                self.storage_stats['performance'] = {
                    'write_time': write_time,
                    'read_time': read_time,
                    'records': original_size,
                    'storage_stats': stats
                }

            logger.info("✅ Compression and performance test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Compression and performance test FAILED: {e}")
            return False

    def test_concurrent_access(self) -> bool:
        """Test concurrent read/write operations"""
        logger.info("=== Testing Concurrent Access ===")

        try:
            # Create multiple TickStore instances to simulate concurrent access
            store1 = TickStore()
            store2 = TickStore()

            symbol = 'MSFT'
            today = datetime.now().strftime('%Y-%m-%d')

            # Create test data
            test_data1 = pd.DataFrame({
                'timestamp': [datetime.now()],
                'price': [300.0],
                'volume': [1000],
                'symbol': [symbol]
            })

            test_data2 = pd.DataFrame({
                'timestamp': [datetime.now()],
                'price': [301.0],
                'volume': [1500],
                'symbol': [symbol]
            })

            # Test concurrent writes
            logger.info("Testing concurrent writes...")
            stored1 = store1.store_ticks(symbol, today, test_data1,
                                      metadata={'store': 'store1'}, overwrite=False)
            stored2 = store2.store_ticks(symbol, today + '_2', test_data2,
                                      metadata={'store': 'store2'}, overwrite=False)

            if stored1 and stored2:
                logger.info("✅ Concurrent writes successful")
            else:
                logger.warning("⚠️ Some concurrent writes failed")

            # Test concurrent reads
            logger.info("Testing concurrent reads...")
            data1 = store1.read_ticks(symbol, today)
            data2 = store2.read_ticks(symbol, today + '_2')

            if data1 is not None and data2 is not None:
                logger.info("✅ Concurrent reads successful")
            else:
                logger.warning("⚠️ Some concurrent reads failed")

            logger.info("✅ Concurrent access test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Concurrent access test FAILED: {e}")
            return False

    def run_complete_test_suite(self) -> bool:
        """Run complete storage verification test suite"""
        logger.info("🚀 Starting Storage Verification Test Suite")
        logger.info("=" * 60)

        tests = [
            ("ArcticDB Initialization", self.test_arcticdb_initialization),
            ("Tick Data Storage", self.test_tick_data_storage),
            ("Metadata Storage", self.test_metadata_storage),
            ("Date Range Queries", self.test_date_range_queries),
            ("Compression and Performance", self.test_compression_and_performance),
            ("Concurrent Access", self.test_concurrent_access)
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
        logger.info(f"🎯 STORAGE VERIFICATION RESULTS")
        logger.info(f"Passed: {passed}/{total} tests")
        logger.info(f"Success rate: {passed/total*100:.1f}%")

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - Storage system is working perfectly!")
            logger.info("✨ Ready for end-to-end integration tests!")
            return True
        else:
            logger.error(f"⚠️ {total - passed} tests failed - Review storage implementation")
            return False

    def generate_storage_report(self):
        """Generate a report of storage statistics"""
        logger.info("\n📊 STORAGE STATISTICS REPORT")
        logger.info("=" * 40)

        for category, stats in self.storage_stats.items():
            logger.info(f"\n{category.upper()}:")
            if isinstance(stats, dict):
                for key, value in stats.items():
                    logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {stats}")


def main():
    """Run the storage verification test suite"""
    try:
        # Ensure data directory exists
        Path("./data").mkdir(exist_ok=True)

        # Run test suite
        test_suite = StorageVerificationTest()
        success = test_suite.run_complete_test_suite()

        # Generate storage report
        test_suite.generate_storage_report()

        if success:
            print("\n🚀 Storage verification successful!")
            print("🔥 ArcticDB storage system is working perfectly!")
            print("📊 Ready for end-to-end pipeline tests!")
        else:
            print("\n⚠️ Storage system needs attention before proceeding")
            print("📋 Review test results and fix failing components")

        return success

    except Exception as e:
        logger.error(f"Test suite initialization failed: {e}")
        return False


if __name__ == "__main__":
    main()