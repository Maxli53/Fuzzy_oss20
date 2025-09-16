"""
Fetch 8 Days of AAPL Tick Data
Following the narrow-but-deep strategy from Data_policy.md
"""
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Setup paths
sys.path.insert(0, '.')
sys.path.insert(0, 'stage_01_data_engine')
sys.path.insert(0, 'foundation')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import directly to avoid circular imports
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.insert(0, 'stage_01_data_engine/storage')

import iqfeed_collector
import enhanced_tick_store
from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic

IQFeedCollector = iqfeed_collector.IQFeedCollector
EnhancedTickStore = enhanced_tick_store.EnhancedTickStore


class AAPLDataFetcher:
    """Fetch 8 days of AAPL tick data"""

    def __init__(self):
        self.collector = IQFeedCollector()
        self.store = EnhancedTickStore()
        self.symbol = 'AAPL'

    def fetch_week_data(self, days_back=8):
        """Fetch 8 days of tick data for AAPL"""
        logger.info("=" * 80)
        logger.info(f"FETCHING {days_back} DAYS OF {self.symbol} TICK DATA")
        logger.info("=" * 80)
        logger.info("Following narrow-but-deep strategy")

        # Ensure IQFeed connection
        if not self.collector.ensure_connection():
            logger.error("Failed to connect to IQFeed")
            return False

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back + 1)  # Extra day for buffer

        logger.info(f"\nDate range: {start_date.date()} to {end_date.date()}")

        total_ticks = 0
        successful_days = 0
        failed_days = []

        # Process each day
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')

            # Skip weekends
            if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                logger.info(f"  Skipping weekend: {date_str}")
                current_date += timedelta(days=1)
                continue

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing {date_str}...")

            try:
                # Fetch tick data for this day
                tick_array = self.collector.get_tick_data(
                    self.symbol,
                    num_days=1,
                    max_ticks=500000,  # 500K ticks max per day
                    start_date=current_date
                )

                if tick_array is None or len(tick_array) == 0:
                    logger.warning(f"  No data for {date_str}")
                    current_date += timedelta(days=1)
                    continue

                logger.info(f"  Fetched {len(tick_array)} ticks")

                # Convert to Pydantic
                pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, self.symbol)

                # Store with bar generation
                success, bar_counts = self.store.store_ticks_with_bars(
                    self.symbol,
                    date_str,
                    pydantic_ticks,
                    overwrite=True
                )

                if success:
                    logger.info(f"  ✓ Stored {len(pydantic_ticks)} ticks")
                    logger.info(f"  ✓ Generated bars: {bar_counts}")
                    total_ticks += len(pydantic_ticks)
                    successful_days += 1
                else:
                    logger.error(f"  Failed to store data for {date_str}")
                    failed_days.append(date_str)

            except Exception as e:
                logger.error(f"  Error processing {date_str}: {e}")
                failed_days.append(date_str)

            current_date += timedelta(days=1)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("FETCH COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total ticks fetched: {total_ticks:,}")
        logger.info(f"Successful days: {successful_days}")
        if failed_days:
            logger.warning(f"Failed days: {failed_days}")

        # Verify storage
        logger.info("\n" + "=" * 60)
        logger.info("VERIFYING STORAGE")
        logger.info("=" * 60)

        try:
            from arcticdb import Arctic
            arctic = Arctic('lmdb://./data/arctic_storage')
            tick_lib = arctic['tick_data']

            symbols = tick_lib.list_symbols()
            aapl_symbols = [s for s in symbols if self.symbol in s]

            logger.info(f"Stored symbols: {len(aapl_symbols)}")
            for symbol in sorted(aapl_symbols)[:10]:  # Show first 10
                logger.info(f"  {symbol}")

            if len(aapl_symbols) > 10:
                logger.info(f"  ... and {len(aapl_symbols) - 10} more")

        except Exception as e:
            logger.error(f"Could not verify storage: {e}")

        return successful_days > 0

    def verify_data_quality(self):
        """Verify the quality of stored data"""
        logger.info("\n" + "=" * 60)
        logger.info("DATA QUALITY CHECK")
        logger.info("=" * 60)

        try:
            from arcticdb import Arctic
            arctic = Arctic('lmdb://./data/arctic_storage')
            tick_lib = arctic['tick_data']

            symbols = tick_lib.list_symbols()
            aapl_symbols = [s for s in symbols if self.symbol in s]

            if not aapl_symbols:
                logger.warning("No AAPL data found")
                return

            # Check first and last days
            first_symbol = sorted(aapl_symbols)[0]
            last_symbol = sorted(aapl_symbols)[-1]

            for symbol in [first_symbol, last_symbol]:
                df = tick_lib.read(symbol).data
                logger.info(f"\n{symbol}:")
                logger.info(f"  Rows: {len(df):,}")
                logger.info(f"  Columns: {df.columns.tolist()[:5]}...")
                logger.info(f"  Time range: {df.index[0]} to {df.index[-1]}")
                logger.info(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
                logger.info(f"  Total volume: {df['size'].sum():,}")

        except Exception as e:
            logger.error(f"Quality check failed: {e}")


if __name__ == "__main__":
    fetcher = AAPLDataFetcher()

    # Fetch the data
    success = fetcher.fetch_week_data(days_back=8)

    if success:
        # Verify data quality
        fetcher.verify_data_quality()

        logger.info("\n" + "=" * 80)
        logger.info("✓ AAPL DATA FETCH COMPLETE")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("  1. Run generate_standard_bars.py to create time bars")
        logger.info("  2. Run test_comprehensive_bars.py to verify")
    else:
        logger.error("Failed to fetch AAPL data")