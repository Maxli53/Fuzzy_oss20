"""
Test script for hedge fund-grade tick infrastructure
Tests TickStore, BarBuilder, and integration with IQFeed
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Import our modules
from stage_01_data_engine.storage import TickStore, BarBuilder, TimezoneHandler
from stage_01_data_engine.collector import DataCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TickInfrastructureTest:
    """Test suite for tick storage infrastructure"""

    def __init__(self):
        self.symbol = "AAPL"
        self.test_date = datetime.now().strftime('%Y-%m-%d')

        # Initialize components
        self.tick_store = TickStore()
        self.bar_builder = BarBuilder()
        self.timezone_handler = TimezoneHandler()
        self.data_collector = DataCollector()

        logger.info("Initialized test suite for tick infrastructure")

    def create_sample_tick_data(self, num_ticks: int = 1000) -> pd.DataFrame:
        """
        Create sample tick data for testing

        Args:
            num_ticks: Number of sample ticks to generate

        Returns:
            DataFrame with realistic tick data
        """
        logger.info(f"Creating {num_ticks} sample ticks for {self.symbol}")

        # Start with realistic AAPL price
        base_price = 235.0

        # Generate timestamps (last 1000 seconds)
        end_time = pd.Timestamp.now()
        start_time = end_time - pd.Timedelta(seconds=num_ticks)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=num_ticks)

        # Generate realistic price movements
        np.random.seed(42)  # Reproducible results
        price_changes = np.random.normal(0, 0.02, num_ticks)
        prices = base_price + np.cumsum(price_changes)

        # Generate realistic volume data
        volumes = np.random.lognormal(mean=6, sigma=1.5, size=num_ticks).astype(int)
        volumes = np.clip(volumes, 100, 50000)  # Reasonable volume range

        # Generate buy/sell sides (60% buy bias during uptrend)
        sides = np.random.choice(['buy', 'sell'], size=num_ticks, p=[0.6, 0.4])

        # Generate exchange codes
        exchanges = np.random.choice(['NASDAQ', 'ARCA', 'NYSE'], size=num_ticks, p=[0.7, 0.2, 0.1])

        # Create bid/ask spread
        spreads = np.random.uniform(0.01, 0.03, num_ticks)
        bids = prices - spreads/2
        asks = prices + spreads/2

        tick_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes,
            'side': sides,
            'exchange': exchanges,
            'bid': bids,
            'ask': asks,
            'bid_size': np.random.randint(100, 1000, num_ticks),
            'ask_size': np.random.randint(100, 1000, num_ticks),
            'conditions': 'R'  # Regular trade
        })

        logger.info(f"Generated sample tick data: {len(tick_data)} ticks")
        logger.info(f"Price range: ${tick_data['price'].min():.2f} - ${tick_data['price'].max():.2f}")
        logger.info(f"Volume range: {tick_data['volume'].min():,} - {tick_data['volume'].max():,}")

        return tick_data

    def test_tick_storage(self) -> bool:
        """Test tick storage functionality"""
        logger.info("=== Testing Tick Storage ===")

        try:
            # Create sample data
            tick_data = self.create_sample_tick_data(1000)

            # Test storage
            success = self.tick_store.store_ticks(
                symbol=self.symbol,
                date=self.test_date,
                tick_df=tick_data,
                metadata={'test_run': True, 'data_source': 'synthetic'}
            )

            if not success:
                logger.error("Failed to store tick data")
                return False

            # Test retrieval
            loaded_data = self.tick_store.load_ticks(self.symbol, self.test_date)

            if loaded_data is None:
                logger.error("Failed to load tick data")
                return False

            # Validate data integrity
            if len(loaded_data) != len(tick_data):
                logger.error(f"Data mismatch: stored {len(tick_data)}, loaded {len(loaded_data)}")
                return False

            # Test metadata
            metadata = self.tick_store.get_tick_metadata(self.symbol, self.test_date)
            if metadata is None:
                logger.error("Failed to retrieve metadata")
                return False

            logger.info(f"Tick storage test PASSED")
            logger.info(f"Stored and retrieved {len(loaded_data)} ticks")
            logger.info(f"Metadata includes: {list(metadata.keys())}")

            return True

        except Exception as e:
            logger.error(f"Tick storage test FAILED: {e}")
            return False

    def test_bar_construction(self) -> bool:
        """Test all bar construction methods"""
        logger.info("=== Testing Bar Construction ===")

        try:
            # Load tick data
            tick_data = self.tick_store.load_ticks(self.symbol, self.test_date)

            if tick_data is None:
                logger.error("No tick data available for bar construction test")
                return False

            # Test different bar types
            bar_tests = [
                ('tick_bars', lambda df: self.bar_builder.tick_bars(df, n=50)),
                ('volume_bars', lambda df: self.bar_builder.volume_bars(df, 10000)),
                ('dollar_bars', lambda df: self.bar_builder.dollar_bars(df, 1000000)),
                ('imbalance_bars', lambda df: self.bar_builder.imbalance_bars(df, 5000)),
                ('volatility_bars', lambda df: self.bar_builder.volatility_bars(df, 0.5)),
                ('range_bars', lambda df: self.bar_builder.range_bars(df, 0.25))
            ]

            results = {}

            for bar_type, bar_func in bar_tests:
                try:
                    bars = bar_func(tick_data)

                    if bars.empty:
                        logger.warning(f"No {bar_type} generated")
                        results[bar_type] = 0
                        continue

                    # Validate bar data
                    if not self._validate_bars(bars, bar_type):
                        logger.error(f"Invalid {bar_type} data")
                        return False

                    results[bar_type] = len(bars)
                    logger.info(f"{bar_type}: {len(bars)} bars generated")

                    # Show sample bar
                    if len(bars) > 0:
                        sample_bar = bars.iloc[0]
                        logger.info(f"Sample {bar_type}: O={sample_bar['open']:.2f} "
                                  f"H={sample_bar['high']:.2f} L={sample_bar['low']:.2f} "
                                  f"C={sample_bar['close']:.2f} V={sample_bar['volume']:,}")

                except Exception as e:
                    logger.error(f"Error testing {bar_type}: {e}")
                    return False

            logger.info(f"Bar construction test PASSED")
            logger.info(f"Bar generation results: {results}")

            return True

        except Exception as e:
            logger.error(f"Bar construction test FAILED: {e}")
            return False

    def test_real_iqfeed_data(self) -> bool:
        """Test with real IQFeed data"""
        logger.info("=== Testing Real IQFeed Data ===")

        try:
            # Get real AAPL data from IQFeed
            bars = self.data_collector.get_daily_bars(self.symbol, num_days=5)

            if not bars:
                logger.error("Failed to get real IQFeed data")
                return False

            # Convert to tick-like data (simulate tick data from daily bars)
            tick_data = self._simulate_ticks_from_bars(bars)

            if tick_data.empty:
                logger.error("Failed to simulate tick data")
                return False

            # Store the simulated tick data
            success = self.tick_store.store_ticks(
                symbol=self.symbol,
                date=self.test_date,
                tick_df=tick_data,
                metadata={'data_source': 'iqfeed_simulated', 'bars_used': len(bars)},
                overwrite=True
            )

            if not success:
                logger.error("Failed to store IQFeed-based tick data")
                return False

            # Test bar construction on real data
            volume_bars = self.bar_builder.volume_bars(tick_data, 5000)
            dollar_bars = self.bar_builder.dollar_bars(tick_data, 500000)

            logger.info(f"Real IQFeed test PASSED")
            logger.info(f"Generated {len(tick_data)} simulated ticks from {len(bars)} daily bars")
            logger.info(f"Created {len(volume_bars)} volume bars and {len(dollar_bars)} dollar bars")

            return True

        except Exception as e:
            logger.error(f"Real IQFeed test FAILED: {e}")
            return False

    def test_storage_stats(self) -> bool:
        """Test storage statistics and monitoring"""
        logger.info("=== Testing Storage Statistics ===")

        try:
            # Get storage stats
            stats = self.tick_store.get_storage_stats()
            logger.info(f"Storage statistics: {stats}")

            # List stored data
            data_list = self.tick_store.list_stored_data(self.symbol)
            logger.info(f"Stored data for {self.symbol}: {len(data_list)} entries")

            for data_info in data_list:
                logger.info(f"  {data_info['date']}: {data_info['tick_count']:,} ticks, "
                          f"{data_info['storage_size_mb']:.2f} MB")

            return True

        except Exception as e:
            logger.error(f"Storage stats test FAILED: {e}")
            return False

    def _validate_bars(self, bars: pd.DataFrame, bar_type: str) -> bool:
        """Validate bar data integrity"""
        try:
            required_cols = ['open', 'high', 'low', 'close', 'volume']

            # Check required columns
            for col in required_cols:
                if col not in bars.columns:
                    logger.error(f"Missing column {col} in {bar_type}")
                    return False

            # Check OHLC logic
            for idx, row in bars.iterrows():
                if row['high'] < row['low']:
                    logger.error(f"Invalid OHLC: High < Low in {bar_type}")
                    return False

                if row['high'] < row['open'] or row['high'] < row['close']:
                    logger.error(f"Invalid OHLC: High < Open/Close in {bar_type}")
                    return False

                if row['low'] > row['open'] or row['low'] > row['close']:
                    logger.error(f"Invalid OHLC: Low > Open/Close in {bar_type}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Bar validation error: {e}")
            return False

    def _simulate_ticks_from_bars(self, daily_bars: list) -> pd.DataFrame:
        """Simulate tick data from daily bars"""
        try:
            all_ticks = []

            for bar in daily_bars:
                # Simulate 100 ticks per bar
                num_ticks = 100

                # Generate timestamps throughout the day
                bar_date = bar['date']
                start_time = pd.Timestamp.combine(bar_date, pd.Timestamp('09:30:00').time())
                end_time = pd.Timestamp.combine(bar_date, pd.Timestamp('16:00:00').time())
                timestamps = pd.date_range(start=start_time, end=end_time, periods=num_ticks)

                # Generate price path from OHLC
                o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
                prices = self._generate_ohlc_path(o, h, l, c, num_ticks)

                # Generate volumes
                total_volume = bar['volume']
                volumes = np.random.dirichlet(np.ones(num_ticks)) * total_volume
                volumes = volumes.astype(int)

                # Create tick records
                for i, (ts, price, volume) in enumerate(zip(timestamps, prices, volumes)):
                    all_ticks.append({
                        'timestamp': ts,
                        'price': price,
                        'volume': volume,
                        'side': np.random.choice(['buy', 'sell']),
                        'exchange': 'NASDAQ'
                    })

            return pd.DataFrame(all_ticks)

        except Exception as e:
            logger.error(f"Error simulating ticks: {e}")
            return pd.DataFrame()

    def _generate_ohlc_path(self, o: float, h: float, l: float, c: float, n: int) -> list:
        """Generate realistic price path that hits OHLC"""
        try:
            prices = [o]
            current = o

            # Ensure we hit high and low
            high_hit = False
            low_hit = False

            for i in range(1, n-1):
                # Random walk with bias toward close
                if not high_hit and np.random.random() < 0.3:
                    current = h
                    high_hit = True
                elif not low_hit and np.random.random() < 0.3:
                    current = l
                    low_hit = True
                else:
                    # Random movement within bounds
                    move = np.random.normal(0, (h-l)*0.02)
                    current = np.clip(current + move, l, h)

                prices.append(current)

            # Force close price
            prices.append(c)

            return prices

        except Exception as e:
            logger.error(f"Error generating OHLC path: {e}")
            return [o] * n

    def run_all_tests(self) -> bool:
        """Run complete test suite"""
        logger.info("Starting hedge fund-grade tick infrastructure test suite")

        tests = [
            ('Tick Storage', self.test_tick_storage),
            ('Bar Construction', self.test_bar_construction),
            ('Real IQFeed Data', self.test_real_iqfeed_data),
            ('Storage Statistics', self.test_storage_stats)
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\nRunning {test_name} test...")
            try:
                if test_func():
                    logger.info(f"✓ {test_name} test PASSED")
                    passed += 1
                else:
                    logger.error(f"✗ {test_name} test FAILED")
            except Exception as e:
                logger.error(f"✗ {test_name} test ERROR: {e}")

        logger.info(f"\n=== Test Results ===")
        logger.info(f"Passed: {passed}/{total} tests")
        logger.info(f"Success rate: {passed/total*100:.1f}%")

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - Infrastructure ready for production!")
            return True
        else:
            logger.error("❌ Some tests failed - Please review errors above")
            return False

def main():
    """Run the test suite"""
    try:
        # Create data directory if it doesn't exist
        Path("./data").mkdir(exist_ok=True)

        # Initialize and run tests
        test_suite = TickInfrastructureTest()
        success = test_suite.run_all_tests()

        if success:
            print("\n🚀 Tick infrastructure is ready for hedge fund-grade trading!")
        else:
            print("\n⚠️ Infrastructure needs fixes before production use.")

        return success

    except Exception as e:
        logger.error(f"Test suite failed to initialize: {e}")
        return False

if __name__ == "__main__":
    main()