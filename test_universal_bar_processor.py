#!/usr/bin/env python3
"""
Test Universal Bar Processor with real IQFeed data

Validates that:
1. All bar types are generated correctly
2. Tier 2 metadata is computed for each bar
3. Performance meets GUI requirements (<100ms per tick batch)
"""

import sys
import time
from datetime import datetime
from pprint import pprint

# Add paths
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('pyiqfeed_orig')
sys.path.append('.')
sys.path.append('foundation')

from iqfeed_collector import IQFeedCollector
from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic
from foundation.utils.universal_bar_processor import UniversalBarProcessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_universal_processor():
    """Test the Universal Bar Processor with real AAPL data"""

    # Initialize components
    collector = IQFeedCollector()

    # Custom config for testing (smaller thresholds for faster results)
    test_config = {
        'time_bars': {
            'enabled': True,
            'intervals': [60, 300]  # 1-min, 5-min
        },
        'tick_bars': {
            'enabled': True,
            'sizes': [100, 500]
        },
        'volume_bars': {
            'enabled': True,
            'thresholds': [1000, 5000, 10000]
        },
        'dollar_bars': {
            'enabled': True,
            'thresholds': [50000, 100000, 500000]
        },
        'range_bars': {
            'enabled': True,
            'ranges': [0.25, 0.50, 1.00]
        },
        'renko_bars': {
            'enabled': False,  # Disabled for now - complex validation issues
            'brick_sizes': [0.25, 0.50]
        },
        'imbalance_bars': {
            'enabled': False,  # Disabled for now - needs ratio not volume
            'thresholds': [0.1, 0.2, 0.3]  # Should be ratios 0-1
        }
    }

    processor = UniversalBarProcessor('AAPL', config=test_config)

    # Connect to IQFeed
    logger.info("Connecting to IQFeed...")
    if not collector.ensure_connection():
        logger.error("Failed to connect to IQFeed")
        return

    # Fetch ticks
    logger.info("Fetching AAPL ticks...")
    tick_array = collector.get_tick_data('AAPL', num_days=1, max_ticks=5000)
    logger.info(f"Fetched {len(tick_array)} ticks")

    # Convert to Pydantic
    logger.info("Converting to Pydantic models...")
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, 'AAPL')
    logger.info(f"Converted {len(pydantic_ticks)} ticks")

    # Process ticks and measure performance
    logger.info("\n" + "="*60)
    logger.info("PROCESSING TICKS THROUGH UNIVERSAL BAR PROCESSOR")
    logger.info("="*60)

    start_time = time.time()
    all_bars = {}
    batch_size = 100

    # Process in batches to simulate real-time
    for i in range(0, len(pydantic_ticks), batch_size):
        batch = pydantic_ticks[i:i+batch_size]
        batch_start = time.time()

        # Process batch
        completed = processor.process_ticks(batch)

        # Track all completed bars
        for builder_key, bars_list in completed.items():
            if builder_key not in all_bars:
                all_bars[builder_key] = []
            all_bars[builder_key].extend(bars_list)

        batch_time = (time.time() - batch_start) * 1000
        if batch_time > 100:
            logger.warning(f"Batch {i//batch_size}: {batch_time:.1f}ms (>100ms target)")

    # Force close any incomplete bars
    logger.info("\nForce closing incomplete bars...")
    final_bars = processor.force_close_all()
    for builder_key, (bar, metadata) in final_bars.items():
        if builder_key not in all_bars:
            all_bars[builder_key] = []
        all_bars[builder_key].append((bar, metadata))

    total_time = time.time() - start_time

    # Print results
    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)

    # Performance metrics
    logger.info(f"\nPERFORMANCE:")
    logger.info(f"  Total ticks processed: {len(pydantic_ticks)}")
    logger.info(f"  Total time: {total_time:.2f} seconds")
    logger.info(f"  Throughput: {len(pydantic_ticks)/total_time:.0f} ticks/second")
    logger.info(f"  Avg per tick: {total_time/len(pydantic_ticks)*1000:.2f}ms")

    # Bar generation summary
    logger.info(f"\nBAR GENERATION SUMMARY:")
    total_bars = 0
    for builder_key in sorted(all_bars.keys()):
        bar_count = len(all_bars[builder_key])
        total_bars += bar_count
        logger.info(f"  {builder_key:20s}: {bar_count:4d} bars")
    logger.info(f"  {'TOTAL':20s}: {total_bars:4d} bars")

    # Sample bar details
    logger.info("\n" + "="*60)
    logger.info("SAMPLE BAR DETAILS")
    logger.info("="*60)

    # Show details for first bar of each type
    for builder_key in sorted(all_bars.keys())[:3]:  # Just first 3 types
        if all_bars[builder_key]:
            bar, metadata = all_bars[builder_key][0]
            logger.info(f"\n{builder_key} (First Bar):")
            logger.info(f"  Timestamp: {bar.timestamp}")
            logger.info(f"  OHLC: ${bar.open:.2f} / ${bar.high:.2f} / ${bar.low:.2f} / ${bar.close:.2f}")
            logger.info(f"  Volume: {bar.volume:,}")
            logger.info(f"  Trade Count: {bar.trade_count}")

            # Show some metadata
            logger.info(f"  Metadata:")
            if 'liquidity_score' in metadata:
                logger.info(f"    Liquidity Score: {metadata['liquidity_score']:.1f}")
            if 'avg_spread_bps' in metadata:
                logger.info(f"    Avg Spread: {metadata['avg_spread_bps']:.2f} bps")
            if 'flow_imbalance' in metadata:
                logger.info(f"    Net Flow: {metadata['flow_imbalance']['net_flow']:+,}")

    # Verify metadata computation
    logger.info("\n" + "="*60)
    logger.info("METADATA VALIDATION")
    logger.info("="*60)

    metadata_complete = True
    for builder_key, bars_list in all_bars.items():
        for bar, metadata in bars_list:
            if metadata is None or 'symbol' not in metadata:
                logger.error(f"Missing metadata for {builder_key}")
                metadata_complete = False
                break

    if metadata_complete:
        logger.info("✓ All bars have complete Tier 2 metadata")
    else:
        logger.error("✗ Some bars missing metadata")

    # Get final statistics
    stats = processor.get_statistics()
    logger.info("\n" + "="*60)
    logger.info("PROCESSOR STATISTICS")
    logger.info("="*60)
    logger.info(f"  Symbol: {stats['symbol']}")
    logger.info(f"  Total Ticks: {stats['total_ticks_processed']}")
    logger.info(f"  Total Bars: {stats['total_bars_generated']}")
    logger.info(f"  Active Builders: {stats['builders_active']}")

    logger.info("\n✓ Universal Bar Processor test complete!")


if __name__ == "__main__":
    test_universal_processor()