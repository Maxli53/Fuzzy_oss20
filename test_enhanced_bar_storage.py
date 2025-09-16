"""
Test Enhanced Tick Store with Bar Processing
"""
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add path for imports
sys.path.insert(0, '.')
sys.path.insert(0, 'stage_01_data_engine/storage')
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.insert(0, 'foundation')

def test_enhanced_store():
    """Test the enhanced tick store with bar processing"""

    # Import directly to avoid circular imports
    sys.path.insert(0, 'stage_01_data_engine/storage')
    sys.path.insert(0, 'stage_01_data_engine/collectors')

    import enhanced_tick_store
    import iqfeed_collector
    from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic

    EnhancedTickStore = enhanced_tick_store.EnhancedTickStore
    IQFeedCollector = iqfeed_collector.IQFeedCollector

    logger.info("=" * 80)
    logger.info("Testing Enhanced Tick Store with Bar Processing")
    logger.info("=" * 80)

    # Initialize components
    logger.info("Initializing Enhanced Tick Store...")
    store = EnhancedTickStore()

    logger.info("Initializing IQFeed Collector...")
    collector = IQFeedCollector()

    # Connect to IQFeed
    logger.info("Connecting to IQFeed...")
    if not collector.ensure_connection():
        logger.error("Failed to connect to IQFeed")
        return

    # Fetch test data
    symbol = 'AAPL'
    logger.info(f"\nFetching {symbol} ticks...")
    tick_array = collector.get_tick_data(symbol, num_days=1, max_ticks=5000)
    logger.info(f"✓ Fetched {len(tick_array)} ticks")

    if len(tick_array) == 0:
        logger.error("No ticks received from IQFeed")
        return

    # Convert to Pydantic
    logger.info(f"\nConverting to Pydantic models...")
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)
    logger.info(f"✓ Converted {len(pydantic_ticks)} ticks to Pydantic models")

    # Store with bar generation
    date = datetime.now().strftime('%Y-%m-%d')
    logger.info(f"\nStoring ticks with bar generation for {symbol} on {date}...")

    success, bar_counts = store.store_ticks_with_bars(
        symbol, date, pydantic_ticks, overwrite=True
    )

    if success:
        logger.info(f"✓ Successfully stored ticks and generated bars")
        logger.info(f"\nBar Generation Summary:")
        logger.info("-" * 40)

        total_bars = sum(bar_counts.values())
        logger.info(f"Total bars generated: {total_bars}")

        logger.info("\nBar counts by type:")
        for bar_type, count in sorted(bar_counts.items()):
            logger.info(f"  {bar_type:20s}: {count:4d} bars")

        # Retrieve some bars to verify
        logger.info("\nRetrieving sample bars...")
        time_bars = store.get_bars(symbol, date, 'time', 60)

        if not time_bars.empty:
            logger.info(f"✓ Retrieved {len(time_bars)} 1-minute bars")
            logger.info(f"\nSample 1-minute bar:")
            sample_bar = time_bars.iloc[0]
            for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'tick_count']:
                if col in sample_bar:
                    logger.info(f"  {col}: {sample_bar[col]}")

        # Get bar statistics
        stats = store.get_bar_statistics()
        logger.info(f"\nBar Processing Statistics:")
        logger.info("-" * 40)
        logger.info(f"Total bars generated: {stats['total_bars_generated']}")
        logger.info(f"Processing time: {stats.get('processing_time_ms', 0):.1f}ms")
        logger.info(f"Last processed: {stats.get('last_processed', 'N/A')}")

        # Check library symbol counts
        if 'libraries' in stats:
            logger.info("\nLibrary symbol counts:")
            for lib_name, count in stats['libraries'].items():
                logger.info(f"  {lib_name:20s}: {count} symbols")

        logger.info("\n" + "=" * 80)
        logger.info("✓ Enhanced Bar Storage Test SUCCESSFUL")
        logger.info("=" * 80)

    else:
        logger.error("✗ Failed to store ticks with bars")


if __name__ == "__main__":
    try:
        test_enhanced_store()
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)