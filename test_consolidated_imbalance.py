"""
Test Consolidated ImbalanceBarBuilder with Both Modes
Demonstrates López de Prado mode and simple mode functionality
"""
import logging
import sys
from decimal import Decimal
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


def test_consolidated_imbalance_builder():
    """Test both modes of the consolidated ImbalanceBarBuilder"""

    # Import directly to avoid circular imports
    sys.path.insert(0, 'stage_01_data_engine/collectors')

    import iqfeed_collector
    from foundation.utils.bar_builder import ImbalanceBarBuilder
    from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic

    IQFeedCollector = iqfeed_collector.IQFeedCollector

    logger.info("=" * 80)
    logger.info("Testing Consolidated ImbalanceBarBuilder")
    logger.info("=" * 80)

    # Initialize IQFeed collector
    collector = IQFeedCollector()

    if not collector.ensure_connection():
        logger.error("Failed to connect to IQFeed")
        return

    # Fetch real data
    symbol = 'AAPL'
    logger.info(f"\nFetching {symbol} ticks...")
    tick_array = collector.get_tick_data(symbol, num_days=1, max_ticks=3000)
    logger.info(f"✓ Fetched {len(tick_array)} ticks")

    # Convert to Pydantic
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)

    # Test 1: López de Prado Mode (Default)
    logger.info("\n" + "=" * 60)
    logger.info("Test 1: López de Prado Mode (Dynamic Threshold)")
    logger.info("=" * 60)

    lopez_builder = ImbalanceBarBuilder(
        symbol=symbol,
        initial_expected_theta=5000,
        use_simple_mode=False,  # Default
        store_ticks=True  # Enable tick storage for order flow analysis
    )

    lopez_bars = []
    for tick in pydantic_ticks:
        bar = lopez_builder.add_tick(tick)
        if bar is not None:
            lopez_bars.append(bar)
            logger.info(f"  López Bar #{len(lopez_bars)}:")
            logger.info(f"    Cumulative imbalance: {bar.cumulative_imbalance:.0f}")
            logger.info(f"    Trigger: {'BUY' if bar.trigger_direction > 0 else 'SELL'}")
            logger.info(f"    Tick count: {bar.tick_count}")
            logger.info(f"    Volume: {bar.volume}")
            logger.info(f"    Has ticks stored: {hasattr(bar, 'ticks') and bar.ticks is not None}")

    logger.info(f"\n✓ Generated {len(lopez_bars)} bars using López de Prado mode")

    # Test 2: Simple Mode (Fixed Threshold)
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: Simple Mode (Fixed Threshold)")
    logger.info("=" * 60)

    simple_builder = ImbalanceBarBuilder(
        symbol=symbol,
        use_simple_mode=True,
        fixed_threshold=Decimal(10000),  # Fixed 10K volume imbalance
        store_ticks=False  # Don't store ticks in simple mode
    )

    simple_bars = []
    for tick in pydantic_ticks:
        bar = simple_builder.add_tick(tick)
        if bar is not None:
            simple_bars.append(bar)
            logger.info(f"  Simple Bar #{len(simple_bars)}:")
            logger.info(f"    Imbalance: {bar.cumulative_imbalance:.0f}")
            logger.info(f"    Direction: {'BUY' if bar.trigger_direction > 0 else 'SELL'}")
            logger.info(f"    Volume: {bar.volume}")

    logger.info(f"\n✓ Generated {len(simple_bars)} bars using simple mode")

    # Test 3: Verify Order Flow Metrics Can Be Computed
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Order Flow Metrics on Bars with Stored Ticks")
    logger.info("=" * 60)

    if lopez_bars and hasattr(lopez_bars[0], 'ticks') and lopez_bars[0].ticks:
        from foundation.utils.order_flow_analyzer import OrderFlowAnalyzer
        import pandas as pd

        # Use first López bar (which has ticks stored)
        test_bar = lopez_bars[0]
        tick_dicts = [tick.model_dump() for tick in test_bar.ticks]
        df = pd.DataFrame(tick_dicts)

        if 'size' in df.columns:
            df['volume'] = df['size']

        # Compute order flow metrics
        metrics = OrderFlowAnalyzer.compute_all_metrics(df)

        logger.info(f"Order flow metrics for first López bar:")
        logger.info(f"  VPIN: {metrics.get('vpin', 0):.4f}")
        logger.info(f"  Kyle's Lambda: {metrics.get('kyle_lambda', 'N/A')}")
        logger.info(f"  Roll Spread: ${metrics.get('roll_spread', 0):.4f}")
        logger.info(f"  Trade Entropy: {metrics.get('trade_entropy', 0):.4f}")
        logger.info(f"  Toxicity Score: {metrics.get('toxicity_toxicity_score', 0):.4f}")
    else:
        logger.warning("No ticks stored in bars for order flow analysis")

    # Comparison Summary
    logger.info("\n" + "=" * 60)
    logger.info("Comparison Summary")
    logger.info("=" * 60)
    logger.info(f"López de Prado mode: {len(lopez_bars)} bars (adaptive threshold)")
    logger.info(f"Simple mode: {len(simple_bars)} bars (fixed threshold)")
    logger.info(f"Data processed: {len(pydantic_ticks)} ticks")

    # Show threshold evolution for López mode
    if hasattr(lopez_builder, 'bar_thetas') and lopez_builder.bar_thetas:
        logger.info(f"\nLópez de Prado threshold evolution:")
        logger.info(f"  Initial: {5000}")
        logger.info(f"  Final: {lopez_builder.expected_theta:.0f}")
        logger.info(f"  Adaptation: {(float(lopez_builder.expected_theta) / 5000 - 1) * 100:.1f}%")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Consolidated ImbalanceBarBuilder Test Complete")
    logger.info("=" * 80)

    # Test 4: Storage Test with Enhanced Tick Store
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Storage with UUID Serialization Fix")
    logger.info("=" * 60)

    try:
        sys.path.insert(0, 'stage_01_data_engine/storage')
        import enhanced_tick_store
        EnhancedTickStore = enhanced_tick_store.EnhancedTickStore

        store = EnhancedTickStore()
        date = datetime.now().strftime('%Y-%m-%d')

        # Store a subset of ticks with bar generation
        success, bar_counts = store.store_ticks_with_bars(
            symbol, date, pydantic_ticks[:1000], overwrite=True
        )

        if success:
            logger.info(f"✓ Successfully stored ticks and bars")
            logger.info(f"  Bar counts: {bar_counts}")
        else:
            logger.error("Failed to store ticks and bars")

    except Exception as e:
        logger.error(f"Storage test failed: {e}")

    return lopez_bars, simple_bars


if __name__ == "__main__":
    try:
        test_consolidated_imbalance_builder()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)