"""
Professional Tick Storage with ArcticDB
Hedge Fund-Grade Infrastructure with Full Metadata
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path
import warnings

# ArcticDB imports
try:
    from arcticdb import Arctic
    from arcticdb.exceptions import LibraryNotFound, NormalizationException
except ImportError:
    raise ImportError("ArcticDB not installed. Run: pip install arcticdb")

from .timezone_handler import TimezoneHandler

logger = logging.getLogger(__name__)

class TickStore:
    """
    Hedge fund-grade tick storage with ArcticDB backend.

    Features:
    - Full tick metadata (bid/ask, exchange, conditions)
    - Symbol/date partitioning for SP500 scale
    - LZ4 compression for efficient storage
    - Timezone normalization
    - Rich metadata tracking
    - Error handling and recovery
    """

    def __init__(self,
                 arctic_uri: str = "lmdb://./data/arctic_storage",
                 enable_compression: bool = True,
                 cache_size_mb: int = 1000):
        """
        Initialize tick storage system.

        Args:
            arctic_uri: ArcticDB connection string
            enable_compression: Enable LZ4 compression
            cache_size_mb: Memory cache size in MB
        """
        self.arctic_uri = arctic_uri
        self.enable_compression = enable_compression
        self.cache_size_mb = cache_size_mb

        # Initialize ArcticDB connection
        self._init_arctic()

        # Initialize timezone handler
        self.timezone_handler = TimezoneHandler()

        # Performance tracking
        self.stats = {
            'ticks_stored': 0,
            'bytes_stored': 0,
            'queries_executed': 0,
            'errors': 0
        }

        logger.info(f"TickStore initialized with URI: {arctic_uri}")

    def _init_arctic(self):
        """Initialize ArcticDB connection and libraries"""
        try:
            # Ensure data directory exists
            if self.arctic_uri.startswith("lmdb://"):
                data_path = Path(self.arctic_uri.replace("lmdb://", ""))
                data_path.mkdir(parents=True, exist_ok=True)

            # Connect to ArcticDB
            self.arctic = Arctic(self.arctic_uri)

            # Initialize required libraries
            self._init_libraries()

            logger.info("ArcticDB connection established successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ArcticDB: {e}")
            raise

    def _init_libraries(self):
        """Initialize all required ArcticDB libraries"""
        libraries = {
            'tick_data': 'Raw tick data with full metadata',
            'ohlc_5s': 'Derived 5-second OHLC bars',
            'ohlc_1m': 'Derived 1-minute OHLC bars',
            'ohlc_5m': 'Derived 5-minute OHLC bars',
            'adaptive_bars': 'Adaptive bars (volume, dollar, imbalance)',
            'metadata': 'System metadata and statistics'
        }

        for lib_name, description in libraries.items():
            try:
                if lib_name not in self.arctic.list_libraries():
                    self.arctic.create_library(lib_name)
                    logger.info(f"Created library: {lib_name}")

                # Get library reference
                setattr(self, f"{lib_name}_lib", self.arctic[lib_name])

            except Exception as e:
                logger.error(f"Error initializing library {lib_name}: {e}")
                raise

    def store_ticks(self,
                   symbol: str,
                   date: str,
                   tick_df: pd.DataFrame,
                   metadata: Optional[Dict] = None,
                   overwrite: bool = False) -> bool:
        """
        Store raw tick data with full metadata.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            date: Date string (YYYY-MM-DD)
            tick_df: DataFrame with tick data
            metadata: Additional metadata dict
            overwrite: Whether to overwrite existing data

        Returns:
            True if successful, False otherwise
        """
        try:
            if tick_df.empty:
                logger.warning(f"Empty DataFrame for {symbol} on {date}")
                return False

            # Validate required columns
            required_cols = ['timestamp', 'price', 'volume']
            missing_cols = [col for col in required_cols if col not in tick_df.columns]
            if missing_cols:
                logger.error(f"Missing required columns for {symbol}: {missing_cols}")
                return False

            # Create storage key with symbol/date partitioning
            storage_key = f"{symbol}/{date}"

            # Check if data already exists
            if not overwrite and self._data_exists(storage_key):
                logger.info(f"Data already exists for {storage_key}, skipping")
                return True

            # Normalize timestamps to exchange timezone
            normalized_df = self.timezone_handler.normalize_dataframe(
                tick_df.copy(), symbol, 'timestamp'
            )

            # Handle object/string columns for ArcticDB compatibility
            for col in normalized_df.columns:
                col_dtype = str(normalized_df[col].dtype)
                if normalized_df[col].dtype == 'object' or 'string' in col_dtype:
                    # Convert to object type (compatible with ArcticDB)
                    normalized_df[col] = normalized_df[col].astype(str).astype('object')

            # Prepare metadata
            tick_metadata = self._prepare_tick_metadata(symbol, date, normalized_df, metadata)

            # Store to ArcticDB with compression
            self.tick_data_lib.write(
                storage_key,
                normalized_df,
                metadata=tick_metadata
            )

            # Update statistics
            self.stats['ticks_stored'] += len(normalized_df)
            self.stats['bytes_stored'] += normalized_df.memory_usage(deep=True).sum()

            logger.info(f"Stored {len(normalized_df)} ticks for {storage_key}")
            return True

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error storing ticks for {symbol} on {date}: {e}")
            return False

    def store_numpy_ticks(self,
                         symbol: str,
                         date: str,
                         tick_array: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None,
                         overwrite: bool = False) -> bool:
        """
        Store raw NumPy tick array from IQFeed.
        Converts NumPy structured array to DataFrame with proper timestamp.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            date: Date string (YYYY-MM-DD)
            tick_array: NumPy structured array from IQFeedCollector
            metadata: Additional metadata dict
            overwrite: Whether to overwrite existing data

        Returns:
            True if successful, False otherwise
        """
        try:
            if tick_array is None or len(tick_array) == 0:
                logger.warning(f"Empty tick array for {symbol} on {date}")
                return False

            # Convert NumPy structured array to DataFrame
            tick_df = self._numpy_ticks_to_dataframe(tick_array)

            # Use existing store_ticks method
            return self.store_ticks(symbol, date, tick_df, metadata, overwrite)

        except Exception as e:
            logger.error(f"Error storing NumPy ticks for {symbol}: {e}")
            return False

    def _numpy_ticks_to_dataframe(self, tick_array: np.ndarray) -> pd.DataFrame:
        """
        Convert IQFeed NumPy tick array to DataFrame with proper timestamp.

        IQFeed tick array fields (from pyiqfeed):
        - tick_id: Request ID
        - date: Date (datetime64[D])
        - time: Microseconds since midnight ET (timedelta64[us])
        - last: Trade price
        - last_sz: Trade size
        - last_type: Exchange code (bytes)
        - mkt_ctr: Market center ID
        - tot_vlm: Total cumulative volume
        - bid: Bid price
        - ask: Ask price
        - cond1-4: Trade condition codes
        """
        # Create DataFrame preserving ALL fields from NumPy array
        df = pd.DataFrame({
            # Core trade data
            'tick_id': tick_array['tick_id'].astype(int),
            'date': tick_array['date'],
            'time_us': tick_array['time'],  # Original microseconds since midnight
            'price': tick_array['last'].astype(float),
            'volume': tick_array['last_sz'].astype(int),

            # Market data
            'exchange': tick_array['last_type'],  # Exchange code
            'market_center': tick_array['mkt_ctr'].astype(int),
            'total_volume': tick_array['tot_vlm'].astype(int),

            # Bid/Ask data
            'bid': tick_array['bid'].astype(float),
            'ask': tick_array['ask'].astype(float),

            # Trade conditions (preserve all 4)
            'condition_1': tick_array['cond1'].astype(int),
            'condition_2': tick_array['cond2'].astype(int),
            'condition_3': tick_array['cond3'].astype(int),
            'condition_4': tick_array['cond4'].astype(int)
        })

        # Combine date and time into proper timestamp
        df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time_us'])

        # Keep original date and time_us for reference (useful for debugging)
        # but drop the duplicate 'date' column
        df = df.drop(['date'], axis=1)

        # Decode exchange codes if they're bytes
        if df['exchange'].dtype == object:
            df['exchange'] = df['exchange'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )

        # Calculate derived fields
        df['spread'] = df['ask'] - df['bid']  # Bid-ask spread
        df['midpoint'] = (df['bid'] + df['ask']) / 2  # Mid price

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.debug(f"Converted {len(tick_array)} NumPy ticks to DataFrame with {len(df.columns)} columns")
        return df

    def load_ticks(self,
                  symbol: str,
                  date_range: Union[str, Tuple[str, str]],
                  columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Load tick data for symbol and date range.

        Args:
            symbol: Stock symbol
            date_range: Single date string or (start_date, end_date) tuple
            columns: Specific columns to load (None for all)

        Returns:
            DataFrame with tick data or None if not found
        """
        try:
            self.stats['queries_executed'] += 1

            if isinstance(date_range, str):
                # Single date
                storage_key = f"{symbol}/{date_range}"

                if not self._data_exists(storage_key):
                    logger.warning(f"No data found for {storage_key}")
                    return None

                result = self.tick_data_lib.read(storage_key)
                tick_df = result.data

                if columns:
                    available_cols = [col for col in columns if col in tick_df.columns]
                    tick_df = tick_df[available_cols]

                logger.info(f"Loaded {len(tick_df)} ticks for {storage_key}")
                return tick_df

            else:
                # Date range
                start_date, end_date = date_range
                all_dfs = []

                # Generate date range
                date_list = pd.date_range(start=start_date, end=end_date, freq='D')

                for single_date in date_list:
                    date_str = single_date.strftime('%Y-%m-%d')
                    single_df = self.load_ticks(symbol, date_str, columns)

                    if single_df is not None:
                        all_dfs.append(single_df)

                if not all_dfs:
                    logger.warning(f"No data found for {symbol} in range {start_date} to {end_date}")
                    return None

                # Combine all DataFrames
                combined_df = pd.concat(all_dfs, ignore_index=True)
                combined_df = combined_df.sort_values('timestamp')

                logger.info(f"Loaded {len(combined_df)} ticks for {symbol} from {start_date} to {end_date}")
                return combined_df

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error loading ticks for {symbol}: {e}")
            return None

    def append_ticks(self,
                    symbol: str,
                    date: str,
                    new_ticks: pd.DataFrame) -> bool:
        """
        Append new ticks to existing data (for real-time updates).

        Args:
            symbol: Stock symbol
            date: Date string
            new_ticks: New tick data to append

        Returns:
            True if successful
        """
        try:
            if new_ticks.empty:
                return True

            storage_key = f"{symbol}/{date}"

            # Normalize timestamps
            normalized_ticks = self.timezone_handler.normalize_dataframe(
                new_ticks.copy(), symbol, 'timestamp'
            )

            # Check if base data exists
            if self._data_exists(storage_key):
                # Load existing data
                existing_data = self.tick_data_lib.read(storage_key)
                existing_df = existing_data.data

                # Combine with new data
                combined_df = pd.concat([existing_df, normalized_ticks], ignore_index=True)
                combined_df = combined_df.sort_values('timestamp').drop_duplicates('timestamp')

                # Update metadata
                metadata = existing_data.metadata
                metadata['last_updated'] = datetime.now().isoformat()
                metadata['total_ticks'] = len(combined_df)
                metadata['last_tick_time'] = combined_df['timestamp'].iloc[-1].isoformat()

                # Store updated data
                self.tick_data_lib.write(storage_key, combined_df, metadata=metadata)
            else:
                # No existing data, store as new
                self.store_ticks(symbol, date, normalized_ticks)

            logger.info(f"Appended {len(normalized_ticks)} ticks to {storage_key}")
            return True

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error appending ticks for {symbol} on {date}: {e}")
            return False

    def get_tick_metadata(self, symbol: str, date: str) -> Optional[Dict]:
        """
        Get metadata for stored tick data.

        Args:
            symbol: Stock symbol
            date: Date string

        Returns:
            Metadata dictionary or None if not found
        """
        try:
            storage_key = f"{symbol}/{date}"

            if not self._data_exists(storage_key):
                return None

            result = self.tick_data_lib.read(storage_key)
            return result.metadata

        except Exception as e:
            logger.error(f"Error getting metadata for {symbol} on {date}: {e}")
            return None

    def list_stored_data(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        List all stored tick data.

        Args:
            symbol: Filter by symbol (None for all)

        Returns:
            List of data info dictionaries
        """
        try:
            all_keys = self.tick_data_lib.list_symbols()
            data_list = []

            for key in all_keys:
                if '/' not in key:
                    continue

                key_symbol, key_date = key.split('/', 1)

                if symbol and key_symbol != symbol:
                    continue

                metadata = self.get_tick_metadata(key_symbol, key_date)

                data_info = {
                    'symbol': key_symbol,
                    'date': key_date,
                    'storage_key': key,
                    'tick_count': metadata.get('total_ticks', 0) if metadata else 0,
                    'first_tick': metadata.get('first_tick_time', '') if metadata else '',
                    'last_tick': metadata.get('last_tick_time', '') if metadata else '',
                    'storage_size_mb': metadata.get('storage_size_mb', 0) if metadata else 0
                }
                data_list.append(data_info)

            # Sort by symbol and date
            data_list.sort(key=lambda x: (x['symbol'], x['date']))
            return data_list

        except Exception as e:
            logger.error(f"Error listing stored data: {e}")
            return []

    def get_storage_stats(self) -> Dict:
        """Get storage system statistics"""
        try:
            return {
                **self.stats,
                'libraries': list(self.arctic.list_libraries()),
                'total_symbols': len(self.list_stored_data()),
                'arctic_uri': self.arctic_uri,
                'compression_enabled': self.enable_compression
            }
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return self.stats

    def _data_exists(self, storage_key: str) -> bool:
        """Check if data exists for given storage key"""
        try:
            return self.tick_data_lib.has_symbol(storage_key)
        except Exception:
            return False

    def _prepare_tick_metadata(self,
                              symbol: str,
                              date: str,
                              tick_df: pd.DataFrame,
                              additional_metadata: Optional[Dict] = None) -> Dict:
        """Prepare comprehensive metadata for tick data"""
        try:
            # Basic statistics
            prices = tick_df['price'].values
            volumes = tick_df['volume'].values

            metadata = {
                # Basic info
                'symbol': symbol,
                'date': date,
                'stored_at': datetime.now().isoformat(),
                'total_ticks': len(tick_df),

                # Time range
                'first_tick_time': tick_df['timestamp'].iloc[0].isoformat(),
                'last_tick_time': tick_df['timestamp'].iloc[-1].isoformat(),

                # Price statistics
                'price_open': float(prices[0]),
                'price_high': float(np.max(prices)),
                'price_low': float(np.min(prices)),
                'price_close': float(prices[-1]),
                'price_mean': float(np.mean(prices)),
                'price_std': float(np.std(prices)),

                # Volume statistics
                'volume_total': int(np.sum(volumes)),
                'volume_mean': float(np.mean(volumes)),
                'volume_std': float(np.std(volumes)),
                'volume_max': int(np.max(volumes)),

                # Trading activity
                'vwap': float(np.sum(prices * volumes) / np.sum(volumes)),
                'dollar_volume': float(np.sum(prices * volumes)),

                # Data quality
                'missing_values': int(tick_df.isnull().sum().sum()),
                'duplicate_timestamps': int(tick_df['timestamp'].duplicated().sum()),

                # Storage info
                'storage_size_mb': float(tick_df.memory_usage(deep=True).sum() / 1024**2),
                'columns': list(tick_df.columns),
                'dtypes': {col: str(dtype) for col, dtype in tick_df.dtypes.items()}
            }

            # Add exchange and market session info
            exchange = self.timezone_handler.get_exchange_for_symbol(symbol)
            metadata['primary_exchange'] = exchange

            # Market hours analysis
            market_sessions = self.timezone_handler.get_market_sessions(symbol, date)
            if market_sessions:
                metadata['market_sessions'] = {
                    k: v.isoformat() for k, v in market_sessions.items()
                }

            # Add side/exchange info if available
            if 'side' in tick_df.columns:
                side_counts = tick_df['side'].value_counts().to_dict()
                metadata['side_distribution'] = {str(k): int(v) for k, v in side_counts.items()}

            if 'exchange' in tick_df.columns:
                exchange_counts = tick_df['exchange'].value_counts().to_dict()
                metadata['exchange_distribution'] = {str(k): int(v) for k, v in exchange_counts.items()}

            # Add any additional metadata
            if additional_metadata:
                metadata.update(additional_metadata)

            return metadata

        except Exception as e:
            logger.error(f"Error preparing metadata: {e}")
            return {
                'symbol': symbol,
                'date': date,
                'stored_at': datetime.now().isoformat(),
                'total_ticks': len(tick_df),
                'error': str(e)
            }

    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """
        Clean up old tick data beyond retention period.

        Args:
            days_to_keep: Number of days to retain

        Returns:
            Statistics about cleanup operation
        """
        try:
            cutoff_date = (datetime.now().date() - pd.Timedelta(days=days_to_keep)).strftime('%Y-%m-%d')

            all_data = self.list_stored_data()
            deleted_count = 0
            deleted_size = 0

            for data_info in all_data:
                if data_info['date'] < cutoff_date:
                    storage_key = data_info['storage_key']

                    try:
                        self.tick_data_lib.delete(storage_key)
                        deleted_count += 1
                        deleted_size += data_info.get('storage_size_mb', 0)
                        logger.info(f"Deleted old data: {storage_key}")
                    except Exception as e:
                        logger.error(f"Error deleting {storage_key}: {e}")

            cleanup_stats = {
                'deleted_count': deleted_count,
                'deleted_size_mb': deleted_size,
                'cutoff_date': cutoff_date,
                'days_kept': days_to_keep
            }

            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return {'error': str(e)}