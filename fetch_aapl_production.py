#!/usr/bin/env python3
"""
Production AAPL Data Fetcher
Uses the WORKING implementation from the 13-hour debugging session
"""

import sys
import logging
from datetime import datetime

# Setup paths - EXACTLY as in the working production pipeline
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('.')
sys.path.append('foundation')
sys.path.append('stage_01_data_engine/storage')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the WORKING components
from iqfeed_collector import IQFeedCollector
from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic
from enhanced_tick_store import EnhancedTickStore

def main():
    """Fetch AAPL data using the production pipeline"""

    symbol = 'AAPL'

    logger.info("=" * 80)
    logger.info("PRODUCTION AAPL DATA FETCHER")
    logger.info("Using the WORKING implementation from 13-hour debug session")
    logger.info("=" * 80)

    # Initialize components
    collector = IQFeedCollector()
    store = EnhancedTickStore()

    # Connect to IQFeed
    logger.info("\nConnecting to IQFeed...")
    if not collector.ensure_connection():
        logger.error("Failed to connect to IQFeed")
        return False
    logger.info("✓ Connected to IQFeed")

    # Fetch tick data - start with a reasonable amount
    logger.info(f"\nFetching {symbol} ticks...")
    tick_array = collector.get_tick_data(
        symbol,
        num_days=1,
        max_ticks=100000  # 100K ticks - reasonable for testing
    )

    if tick_array is None or len(tick_array) == 0:
        logger.error("No data fetched")
        return False

    logger.info(f"✓ Fetched {len(tick_array)} ticks")

    # Convert to Pydantic using the WORKING converter
    logger.info("Converting to Pydantic models...")
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)
    logger.info(f"✓ Converted {len(pydantic_ticks)} ticks")

    # Store with bar generation using the WORKING enhanced store
    date_str = datetime.now().strftime('%Y-%m-%d')
    logger.info(f"\nStoring ticks and generating bars for {date_str}...")

    success, bar_counts = store.store_ticks_with_bars(
        symbol=symbol,
        date=date_str,
        pydantic_ticks=pydantic_ticks,
        overwrite=True
    )

    if success:
        logger.info(f"✓ Successfully stored {len(pydantic_ticks)} ticks")
        logger.info(f"✓ Generated bars: {bar_counts}")

        # Verify storage
        logger.info("\nVerifying storage...")
        try:
            from arcticdb import Arctic
            arctic = Arctic('lmdb://./data/arctic_storage')

            # Check tick data
            if 'tick_data' in arctic.list_libraries():
                tick_lib = arctic['tick_data']
                symbols = tick_lib.list_symbols()
                aapl_symbols = [s for s in symbols if symbol in s]
                logger.info(f"✓ Stored {len(aapl_symbols)} tick symbols")

                if aapl_symbols:
                    # Read first symbol to verify
                    test_df = tick_lib.read(aapl_symbols[0]).data
                    logger.info(f"✓ Verified: {len(test_df)} ticks in {aapl_symbols[0]}")

            # Check bar data
            for bar_type in ['bars_time_bars', 'bars_tick_bars', 'bars_volume_bars']:
                if bar_type in arctic.list_libraries():
                    bar_lib = arctic[bar_type]
                    bar_symbols = bar_lib.list_symbols()
                    if bar_symbols:
                        logger.info(f"✓ {bar_type}: {len(bar_symbols)} symbols stored")

        except Exception as e:
            logger.warning(f"Verification warning: {e}")

        return True
    else:
        logger.error("Failed to store ticks and generate bars")
        return False

if __name__ == "__main__":
    success = main()

    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✓ AAPL DATA FETCH COMPLETE")
        logger.info("=" * 80)
        logger.info("Data successfully stored using production pipeline")
    else:
        logger.error("\nFailed to complete AAPL data fetch")
        sys.exit(1)