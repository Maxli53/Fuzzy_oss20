"""
Test script to demonstrate and fix Renko bar VWAP validation issue
"""
import sys
import logging
from decimal import Decimal

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


def test_renko_vwap_issue():
    """Test and fix the Renko bar VWAP validation issue"""

    # Import directly to avoid circular imports
    sys.path.insert(0, 'stage_01_data_engine/collectors')

    import iqfeed_collector
    from foundation.utils.bar_builder import RenkoBarBuilder
    from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic

    IQFeedCollector = iqfeed_collector.IQFeedCollector

    logger.info("=" * 80)
    logger.info("Testing Renko Bar VWAP Validation Issue")
    logger.info("=" * 80)

    # Initialize IQFeed collector
    collector = IQFeedCollector()

    if not collector.ensure_connection():
        logger.error("Failed to connect to IQFeed")
        return

    # Fetch real data
    symbol = 'AAPL'
    logger.info(f"\nFetching {symbol} ticks...")
    tick_array = collector.get_tick_data(symbol, num_days=1, max_ticks=1000)
    logger.info(f"✓ Fetched {len(tick_array)} ticks")

    # Convert to Pydantic
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)

    # Test with small brick size to trigger multiple bars
    brick_size = Decimal('0.10')  # 10 cent bricks for AAPL
    logger.info(f"\nBuilding Renko bars with brick size: ${brick_size}")

    builder = RenkoBarBuilder(symbol, brick_size)

    bars = []
    error_count = 0

    for i, tick in enumerate(pydantic_ticks):
        try:
            bar = builder.add_tick(tick)
            if bar is not None:
                bars.append(bar)
                logger.info(f"  Bar #{len(bars)} created:")
                logger.info(f"    Open: ${bar.open:.2f}, Close: ${bar.close:.2f}")
                logger.info(f"    High: ${bar.high:.2f}, Low: ${bar.low:.2f}")
                logger.info(f"    VWAP: ${bar.vwap:.2f}")
                logger.info(f"    Direction: {'UP' if bar.brick_direction > 0 else 'DOWN'}")

                # Check if VWAP is within synthetic OHLC range
                if not (bar.low <= bar.vwap <= bar.high):
                    logger.warning(f"    ⚠️ VWAP ${bar.vwap:.2f} outside synthetic range [{bar.low:.2f}, {bar.high:.2f}]")

        except Exception as e:
            error_count += 1
            if "VWAP must be within OHLC range" in str(e):
                logger.error(f"  VWAP validation error at tick {i}: {e}")
                # Log the problematic accumulator state
                logger.info(f"    Accumulator VWAP: ${builder.accumulator.get_vwap():.2f}")
                logger.info(f"    Brick bounds: [{builder.current_brick_low:.2f}, {builder.current_brick_high:.2f}]")
            else:
                logger.error(f"  Unexpected error at tick {i}: {e}")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Results:")
    logger.info(f"  Total bars created: {len(bars)}")
    logger.info(f"  Validation errors: {error_count}")

    if error_count > 0:
        logger.info(f"\n⚠️ Found {error_count} VWAP validation errors!")
        logger.info("This confirms the issue: Renko bars use synthetic OHLC values")
        logger.info("but calculate VWAP from actual tick data, which can fall outside")
        logger.info("the synthetic brick boundaries.")
    else:
        logger.info("\n✓ No VWAP validation errors found")

    return bars, error_count


if __name__ == "__main__":
    try:
        bars, errors = test_renko_vwap_issue()

        if errors > 0:
            logger.info("\n" + "=" * 80)
            logger.info("PROPOSED FIX:")
            logger.info("=" * 80)
            logger.info("For Renko bars, we should use the actual high/low from")
            logger.info("accumulated ticks instead of synthetic brick boundaries")
            logger.info("for the high/low fields, while keeping the synthetic")
            logger.info("open/close for the brick structure.")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)