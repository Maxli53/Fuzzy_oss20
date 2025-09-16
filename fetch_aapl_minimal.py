#!/usr/bin/env python3
"""
Minimal AAPL Fetcher - Just store ticks, no bars
Uses ONLY the working converter, nothing else
"""

import sys
import logging
from datetime import datetime
import pandas as pd
from arcticdb import Arctic

# Setup paths
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('.')
sys.path.append('foundation')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import ONLY what we need
from iqfeed_collector import IQFeedCollector
from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic

def main():
    """Minimal fetch - just get and store ticks"""

    symbol = 'AAPL'

    logger.info("=" * 60)
    logger.info("MINIMAL AAPL FETCHER")
    logger.info("=" * 60)

    # Connect to IQFeed
    collector = IQFeedCollector()
    if not collector.ensure_connection():
        logger.error("Failed to connect to IQFeed")
        return False

    # Fetch ticks
    logger.info(f"Fetching {symbol} ticks...")
    tick_array = collector.get_tick_data(symbol, num_days=1, max_ticks=50000)

    if tick_array is None or len(tick_array) == 0:
        logger.error("No data fetched")
        return False

    logger.info(f"✓ Fetched {len(tick_array)} ticks")

    # Convert to Pydantic
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)
    logger.info(f"✓ Converted {len(pydantic_ticks)} ticks")

    # Convert to DataFrame manually
    import json
    from decimal import Decimal

    tick_dicts = []
    for tick in pydantic_ticks:
        tick_dict = tick.model_dump()

        # Handle special types
        for key, value in tick_dict.items():
            if hasattr(value, '__class__'):
                if value.__class__.__name__ == 'UUID':
                    tick_dict[key] = str(value)
                elif isinstance(value, Decimal):
                    tick_dict[key] = float(value)
                elif isinstance(value, dict):
                    tick_dict[key] = json.dumps(value)

        tick_dicts.append(tick_dict)

    df = pd.DataFrame(tick_dicts)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Store in ArcticDB - minimal approach
    arctic = Arctic('lmdb://./data/arctic_storage')

    # Create only ONE library
    if 'ticks' not in arctic.list_libraries():
        arctic.create_library('ticks')

    tick_lib = arctic['ticks']

    # Store with simple key
    date_str = datetime.now().strftime('%Y-%m-%d')
    key = f"{symbol}/{date_str}"

    tick_lib.write(key, df, metadata={'symbol': symbol, 'date': date_str})
    logger.info(f"✓ Stored {len(df)} ticks with key: {key}")

    # Verify
    read_df = tick_lib.read(key).data
    logger.info(f"✓ Verified: {len(read_df)} rows in database")

    return True

if __name__ == "__main__":
    if main():
        logger.info("\n✓ SUCCESS - AAPL data stored")
    else:
        logger.error("\n✗ FAILED")