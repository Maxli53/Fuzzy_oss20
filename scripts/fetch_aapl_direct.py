"""
Direct AAPL Data Fetcher
Minimal script using direct imports
"""
import sys
import logging
from datetime import datetime
import pandas as pd

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

# Direct imports
import pyiqfeed as iq
from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic
from arcticdb import Arctic


def fetch_aapl_direct():
    """Direct fetch of AAPL data using PyIQFeed"""
    symbol = 'AAPL'

    logger.info("=" * 80)
    logger.info("DIRECT AAPL DATA FETCH")
    logger.info("=" * 80)

    # Fetch tick data using PyIQFeed pattern
    logger.info(f"Fetching {symbol} ticks...")
    try:
        hist_conn = iq.HistoryConn(name=f"pyiqfeed-{symbol}-tick")

        with iq.ConnConnector([hist_conn]) as connector:
            tick_array = hist_conn.request_ticks(
                ticker=symbol,
                max_ticks=50000  # 50K ticks - will get most recent
            )

            if tick_array is None or len(tick_array) == 0:
                logger.warning("No data fetched")
                return False

            logger.info(f"✓ Fetched {len(tick_array)} ticks")

    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        return False

    # Convert to Pydantic
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)
    logger.info(f"✓ Converted to {len(pydantic_ticks)} Pydantic models")

    # Store in ArcticDB
    try:
        arctic = Arctic('lmdb://./data/arctic_storage')
        tick_lib = arctic['tick_data']

        # Convert to DataFrame
        import json
        from decimal import Decimal

        tick_dicts = []
        for tick in pydantic_ticks:
            tick_dict = tick.model_dump()

            # Convert UUID fields to strings
            if 'tick_id' in tick_dict:
                tick_dict['tick_id'] = str(tick_dict['tick_id'])
            if 'id' in tick_dict:
                tick_dict['id'] = str(tick_dict['id'])

            # Convert Decimal fields to float
            for field in ['price', 'bid', 'ask', 'bid_price', 'ask_price', 'vwap']:
                if field in tick_dict and isinstance(tick_dict[field], Decimal):
                    tick_dict[field] = float(tick_dict[field])

            # Convert metadata dict to JSON string
            if 'metadata' in tick_dict and isinstance(tick_dict['metadata'], dict):
                tick_dict['metadata'] = json.dumps(tick_dict['metadata'])

            tick_dicts.append(tick_dict)

        df = pd.DataFrame(tick_dicts)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Store
        date_str = datetime.now().strftime('%Y-%m-%d')
        key = f"STOCKS/{symbol}/{date_str}"

        tick_lib.write(key, df, metadata={'symbol': symbol, 'date': date_str})
        logger.info(f"✓ Stored {len(df)} ticks with key: {key}")

        # Verify
        read_df = tick_lib.read(key).data
        logger.info(f"✓ Verified: {len(read_df)} rows in database")

        # Show sample
        logger.info("\nSample data:")
        logger.info(f"  Time range: {read_df.index[0]} to {read_df.index[-1]}")
        logger.info(f"  Price range: ${read_df['price'].min():.2f} - ${read_df['price'].max():.2f}")
        logger.info(f"  Total volume: {read_df['size'].sum():,}")

        return True

    except Exception as e:
        logger.error(f"Storage failed: {e}")
        return False


if __name__ == "__main__":
    success = fetch_aapl_direct()

    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✓ AAPL DATA FETCH COMPLETE")
        logger.info("=" * 80)
        logger.info("Next: Generate time bars with bar builders")
    else:
        logger.error("Failed to fetch AAPL data")