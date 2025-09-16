"""
Test Fixed Imbalance Bars and Order Flow Analytics
"""
import logging
import sys
import pandas as pd
from datetime import datetime

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


def test_imbalance_bars_and_order_flow():
    """Test imbalance bars with López de Prado methodology and order flow metrics"""

    # Import directly to avoid circular imports
    sys.path.insert(0, 'stage_01_data_engine/collectors')

    import iqfeed_collector
    from foundation.utils.universal_bar_processor import UniversalBarProcessor
    from foundation.utils.order_flow_analyzer import OrderFlowAnalyzer
    from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic

    IQFeedCollector = iqfeed_collector.IQFeedCollector

    logger.info("=" * 80)
    logger.info("Testing Imbalance Bars and Order Flow Analytics")
    logger.info("=" * 80)

    # Initialize IQFeed collector
    collector = IQFeedCollector()

    if not collector.ensure_connection():
        logger.error("Failed to connect to IQFeed")
        return

    # Fetch real data
    symbol = 'AAPL'
    logger.info(f"\nFetching {symbol} ticks...")
    tick_array = collector.get_tick_data(symbol, num_days=1, max_ticks=5000)
    logger.info(f"✓ Fetched {len(tick_array)} ticks")

    # Convert to Pydantic
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)

    # Test 1: Imbalance Bars
    logger.info("\n" + "=" * 60)
    logger.info("Testing Imbalance Bars (López de Prado)")
    logger.info("=" * 60)

    config = {
        'time_bars': {'enabled': False},
        'tick_bars': {'enabled': False},
        'volume_bars': {'enabled': False},
        'dollar_bars': {'enabled': False},
        'range_bars': {'enabled': False},
        'renko_bars': {'enabled': False},
        'imbalance_bars': {
            'enabled': True,
            'initial_expected_thetas': [5000, 10000, 20000]
        }
    }

    processor = UniversalBarProcessor(symbol, config)

    # Process ticks
    start_time = datetime.now()
    all_bars = processor.process_ticks(pydantic_ticks)
    processing_time = (datetime.now() - start_time).total_seconds() * 1000

    logger.info(f"\nImbalance Bar Results:")
    logger.info(f"Processing time: {processing_time:.1f}ms")

    for builder_key, bars_list in all_bars.items():
        if 'imbalance' in builder_key:
            logger.info(f"\n{builder_key}:")
            logger.info(f"  Bars generated: {len(bars_list)}")

            if bars_list:
                # Show first bar details
                first_bar, first_meta = bars_list[0]
                logger.info(f"  First bar:")
                logger.info(f"    Timestamp: {first_bar.timestamp}")
                logger.info(f"    Cumulative imbalance: {first_bar.cumulative_imbalance:.2f}")
                logger.info(f"    Trigger direction: {'BUY' if first_bar.trigger_direction > 0 else 'SELL'}")
                logger.info(f"    Imbalance threshold: {first_bar.imbalance_threshold:.4f}")
                logger.info(f"    Tick count: {first_bar.tick_count}")
                logger.info(f"    Volume: {first_bar.volume}")

    # Test 2: Order Flow Analytics
    logger.info("\n" + "=" * 60)
    logger.info("Testing Order Flow Analytics")
    logger.info("=" * 60)

    # Convert ticks to DataFrame for order flow analysis
    tick_dicts = [tick.model_dump() for tick in pydantic_ticks[:1000]]  # Use first 1000 for speed
    df = pd.DataFrame(tick_dicts)

    # Rename columns as needed
    if 'size' in df.columns:
        df['volume'] = df['size']

    # Compute order flow metrics
    logger.info("\nComputing order flow metrics...")
    metrics = OrderFlowAnalyzer.compute_all_metrics(df)

    logger.info("\nOrder Flow Metrics:")
    logger.info("-" * 40)

    # Display metrics in organized groups
    logger.info("\nToxicity Metrics:")
    logger.info(f"  VPIN: {metrics.get('vpin', 0):.4f}")
    logger.info(f"  Toxicity Score: {metrics.get('toxicity_toxicity_score', 0):.4f}")
    logger.info(f"  Sign Persistence: {metrics.get('toxicity_sign_persistence', 0):.4f}")
    logger.info(f"  Volume Concentration: {metrics.get('toxicity_volume_concentration', 0):.4f}")

    logger.info("\nLiquidity Metrics:")
    logger.info(f"  Kyle's Lambda: {metrics.get('kyle_lambda', 'N/A')}")
    logger.info(f"  Roll Spread: ${metrics.get('roll_spread', 0):.4f}")
    logger.info(f"  Amihud Illiquidity: {metrics.get('amihud_illiquidity', 0):.2f}")

    logger.info("\nInformation Content:")
    logger.info(f"  Trade Entropy: {metrics.get('trade_entropy', 0):.4f}")

    logger.info("\nPrice Contribution:")
    logger.info(f"  Buy WPC: {metrics.get('buy_wpc', 0):.4f}")
    logger.info(f"  Sell WPC: {metrics.get('sell_wpc', 0):.4f}")
    logger.info(f"  Dominance: {metrics.get('dominance', 0):.4f}")

    # Test 3: Combine Metrics with Bars
    logger.info("\n" + "=" * 60)
    logger.info("Enhanced Bar Metadata with Order Flow")
    logger.info("=" * 60)

    # Process a smaller batch to show how metrics enhance bars
    small_batch = pydantic_ticks[:500]
    small_df = pd.DataFrame([tick.model_dump() for tick in small_batch])

    if 'size' in small_df.columns:
        small_df['volume'] = small_df['size']

    # Compute metrics for this batch
    batch_metrics = OrderFlowAnalyzer.compute_all_metrics(small_df)

    logger.info("\nSample Enhanced Bar Metadata:")
    logger.info(f"  Base metrics (existing): spread, liquidity_score, vwap")
    logger.info(f"  + Order flow toxicity: VPIN = {batch_metrics.get('vpin', 0):.4f}")
    logger.info(f"  + Price impact: Kyle's λ = {batch_metrics.get('kyle_lambda', 'N/A')}")
    logger.info(f"  + Effective spread: Roll = ${batch_metrics.get('roll_spread', 0):.4f}")
    logger.info(f"  + Trade entropy: {batch_metrics.get('trade_entropy', 0):.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Imbalance Bars and Order Flow Analytics Test Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        test_imbalance_bars_and_order_flow()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)