"""
Simple AAPL Data Fetcher
Minimal script to fetch AAPL data without creating extra libraries
"""
import sys
import logging
from datetime import datetime, timedelta

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

from stage_01_data_engine.collectors.iqfeed_collector import IQFeedCollector
from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic
from arcticdb import Arctic
import pandas as pd


def fetch_aapl_data():
    """Fetch AAPL tick data for today only"""
    symbol = 'AAPL'

    logger.info("=" * 80)
    logger.info("FETCHING AAPL DATA (Simple Mode)")
    logger.info("=" * 80)

    # Initialize collector
    collector = IQFeedCollector()

    # Ensure connection
    if not collector.ensure_connection():
        logger.error("Failed to connect to IQFeed")
        return False

    # Fetch tick data for today
    logger.info(f"Fetching {symbol} ticks for today...")
    tick_array = collector.get_tick_data(
        symbol,
        num_days=1,
        max_ticks=100000  # 100K ticks max
    )

    if tick_array is None or len(tick_array) == 0:
        logger.warning("No data fetched")
        return False

    logger.info(f"✓ Fetched {len(tick_array)} ticks")

    # Convert to Pydantic
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)

    # Store directly in ArcticDB
    try:
        arctic = Arctic('lmdb://./data/arctic_storage')

        # Use existing tick_data library
        tick_lib = arctic['tick_data']

        # Convert to DataFrame for storage
        tick_dicts = []
        for tick in pydantic_ticks:
            tick_dict = tick.model_dump()
            # Convert UUID to string
            if 'tick_id' in tick_dict:
                tick_dict['tick_id'] = str(tick_dict['tick_id'])
            tick_dicts.append(tick_dict)

        df = pd.DataFrame(tick_dicts)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Store with simple key
        date_str = datetime.now().strftime('%Y-%m-%d')
        key = f"STOCKS/{symbol}/{date_str}"

        tick_lib.write(key, df, metadata={'symbol': symbol, 'date': date_str})
        logger.info(f"✓ Stored data with key: {key}")

        # Verify
        read_df = tick_lib.read(key).data
        logger.info(f"✓ Verified: {len(read_df)} rows stored")

        return True

    except Exception as e:
        logger.error(f"Storage failed: {e}")
        return False


if __name__ == "__main__":
    success = fetch_aapl_data()

    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✓ AAPL DATA FETCH COMPLETE")
        logger.info("=" * 80)
    else:
        logger.error("Failed to fetch AAPL data")