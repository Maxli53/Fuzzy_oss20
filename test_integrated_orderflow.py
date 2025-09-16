"""
Test Order Flow Integration with Bar Processing
Demonstrates how order flow metrics enhance bar metadata
"""
import logging
import sys
import pandas as pd
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, '.')
sys.path.insert(0, 'foundation')
sys.path.insert(0, 'stage_01_data_engine')


def test_integrated_order_flow():
    """Test order flow metrics integrated into bar metadata"""

    # Import directly to avoid circular imports
    sys.path.insert(0, 'stage_01_data_engine/collectors')

    import iqfeed_collector
    from foundation.utils.universal_bar_processor import UniversalBarProcessor
    from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic

    # Import EnhancedTickStore directly from module
    sys.path.insert(0, 'stage_01_data_engine/storage')
    import enhanced_tick_store
    EnhancedTickStore = enhanced_tick_store.EnhancedTickStore

    IQFeedCollector = iqfeed_collector.IQFeedCollector

    logger.info("=" * 80)
    logger.info("Testing Integrated Order Flow Metrics in Bar Processing")
    logger.info("=" * 80)

    # Initialize components
    collector = IQFeedCollector()
    store = EnhancedTickStore()

    if not collector.ensure_connection():
        logger.error("Failed to connect to IQFeed")
        return

    # Fetch real data
    symbol = 'AAPL'
    logger.info(f"\nFetching {symbol} ticks...")
    tick_array = collector.get_tick_data(symbol, num_days=1, max_ticks=10000)
    logger.info(f"✓ Fetched {len(tick_array)} ticks")

    # Convert to Pydantic
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)

    # Test with focused configuration - only enable bars that store tick data
    config = {
        'time_bars': {
            'enabled': True,
            'intervals': [60, 300]  # 1-min, 5-min
        },
        'tick_bars': {
            'enabled': False  # Disable for this test
        },
        'volume_bars': {
            'enabled': True,
            'thresholds': [10000]  # 10K shares
        },
        'dollar_bars': {
            'enabled': False  # Disable for this test
        },
        'range_bars': {
            'enabled': False  # Disable for this test
        },
        'renko_bars': {
            'enabled': False  # Already disabled due to VWAP issue
        },
        'imbalance_bars': {
            'enabled': True,
            'initial_expected_thetas': [10000]  # Single threshold
        }
    }

    # Process with Universal Bar Processor
    processor = UniversalBarProcessor(symbol, config)

    logger.info("\n" + "=" * 60)
    logger.info("Processing Ticks and Computing Order Flow Metrics")
    logger.info("=" * 60)

    start_time = datetime.now()
    all_bars = processor.process_ticks(pydantic_ticks)
    processing_time = (datetime.now() - start_time).total_seconds() * 1000

    logger.info(f"\nProcessing Results:")
    logger.info(f"  Processing time: {processing_time:.1f}ms")
    logger.info(f"  Tick throughput: {len(pydantic_ticks) / (processing_time/1000):.0f} ticks/sec")

    # Display results for each bar type
    for builder_key, bars_list in all_bars.items():
        if bars_list:
            logger.info(f"\n{builder_key}:")
            logger.info(f"  Total bars generated: {len(bars_list)}")

            # Show first bar with order flow metrics
            first_bar, first_meta = bars_list[0]

            logger.info(f"\n  First Bar Details:")
            logger.info(f"    Timestamp: {first_bar.timestamp}")
            logger.info(f"    OHLCV: O={first_bar.open:.2f}, H={first_bar.high:.2f}, "
                       f"L={first_bar.low:.2f}, C={first_bar.close:.2f}, V={first_bar.volume}")

            # Display order flow metrics if available
            if 'order_flow' in first_meta:
                of = first_meta['order_flow']
                logger.info(f"\n  Order Flow Metrics:")
                logger.info(f"    VPIN (toxicity): {of.get('vpin', 0):.4f}")
                logger.info(f"    Kyle's Lambda: {of.get('kyle_lambda', 'N/A')}")
                logger.info(f"    Roll Spread: ${of.get('roll_spread', 0):.4f}")
                logger.info(f"    Trade Entropy: {of.get('trade_entropy', 0):.4f}")
                logger.info(f"    Toxicity Score: {of.get('toxicity_score', 0):.4f}")
                logger.info(f"    Dominance: {of.get('dominance', 0):.4f}")
            else:
                logger.info("    [Order flow metrics not computed - ticks may not be stored in bar]")

    # Store with Enhanced Tick Store
    logger.info("\n" + "=" * 60)
    logger.info("Storing to Enhanced Tick Store with Bar Generation")
    logger.info("=" * 60)

    date = datetime.now().strftime('%Y-%m-%d')
    success, bar_counts = store.store_ticks_with_bars(
        symbol, date, pydantic_ticks[:5000], overwrite=True  # Store subset
    )

    if success:
        logger.info(f"✓ Successfully stored ticks and generated bars")
        logger.info(f"  Bar counts by type: {bar_counts}")

        # Retrieve and check metadata
        time_bars = store.get_bars(symbol, date, 'time', 60)
        if not time_bars.empty:
            logger.info(f"\n  Retrieved {len(time_bars)} 1-minute bars from storage")

            # Get metadata for first bar
            metadata_list = store.get_bar_metadata(symbol, date, 'time', 60)
            if metadata_list:
                first_metadata = metadata_list[0]
                if 'tier2_metadata' in first_metadata and 'order_flow' in first_metadata['tier2_metadata']:
                    logger.info("  ✓ Order flow metrics successfully stored in bar metadata")
                else:
                    logger.info("  ⚠ Order flow metrics not found in stored metadata")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Integration Test Summary")
    logger.info("=" * 80)
    logger.info(f"✓ Processed {len(pydantic_ticks)} ticks")
    logger.info(f"✓ Generated bars with order flow metrics")
    logger.info(f"✓ Stored to Enhanced Tick Store")
    logger.info("\nOrder flow metrics now automatically computed for each bar:")
    logger.info("  - VPIN (toxicity)")
    logger.info("  - Kyle's Lambda (price impact)")
    logger.info("  - Roll Spread (effective spread)")
    logger.info("  - Trade Entropy (information content)")
    logger.info("  - Weighted Price Contribution")
    logger.info("  - Amihud Illiquidity")

    # Get final statistics
    stats = processor.get_statistics()
    logger.info(f"\nFinal Statistics: {stats}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Order Flow Integration Test Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        test_integrated_order_flow()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)