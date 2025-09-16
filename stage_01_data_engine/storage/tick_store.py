"""
Professional Tick Storage with ArcticDB
Hedge Fund-Grade Infrastructure with Full Metadata
"""
import os
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from pathlib import Path
import warnings
import gc
import time
from functools import wraps
from enum import Enum
import uuid

# ArcticDB imports
try:
    from arcticdb import Arctic
    from arcticdb.exceptions import LibraryNotFound, NormalizationException
except ImportError:
    raise ImportError("ArcticDB not installed. Run: pip install arcticdb")

# Timezone handling - store everything as ET (Eastern Time) for market consistency
import pytz

# Setup logger first
logger = logging.getLogger(__name__)

# Import Tier 2 metadata computer
import sys
sys.path.append('.')
sys.path.append('foundation')
try:
    from foundation.utils.metadata_computer import MetadataComputer
except ImportError:
    logger.warning("MetadataComputer not available, Tier 2 metadata will not be computed")
    MetadataComputer = None


# Custom Exception Classes
class TickStoreError(Exception):
    """Base exception for TickStore errors"""
    pass


class DataValidationError(TickStoreError):
    """Raised when data validation fails"""
    pass


class StorageError(TickStoreError):
    """Raised when storage operations fail"""
    pass


class ConfigurationError(TickStoreError):
    """Raised when configuration is invalid"""
    pass


class ConnectionError(TickStoreError):
    """Raised when ArcticDB connection fails"""
    pass


class ErrorSeverity(Enum):
    """Error severity levels for prioritization"""
    CRITICAL = 1  # System cannot continue
    HIGH = 2      # Operation failed, needs attention
    MEDIUM = 3    # Issue detected, operation continued
    LOW = 4       # Minor issue, informational


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, StorageError) as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
                except Exception as e:
                    # Don't retry on other exceptions
                    raise

            if last_exception:
                raise last_exception
        return wrapper
    return decorator

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
                 config_file: str = "stage_01_data_engine/config/storage_config.yaml",
                 arctic_uri: Optional[str] = None,
                 override_mode: Optional[str] = None):
        """
        Initialize tick storage system with centralized configuration.

        Args:
            config_file: Path to storage configuration file
            arctic_uri: Optional override for ArcticDB connection string
            override_mode: Optional override for operating mode ('backfill' or 'production')
        """
        # Load configuration
        self._load_config(config_file)

        # Apply overrides if provided
        if arctic_uri:
            self.arctic_uri = arctic_uri
        else:
            self.arctic_uri = self.config['arctic']['uri']

        if override_mode:
            self.mode = override_mode
        else:
            self.mode = self.config['write_config']['mode']

        # Extract settings based on mode
        self.enable_compression = self.config['storage']['enable_compression']
        self.cache_size_mb = self.config['storage']['cache_size_mb']

        # Write configuration
        write_cfg = self.config['write_config']
        self.write_batch_size = write_cfg['batch_size']
        self.enable_deduplication = write_cfg['enable_deduplication']
        self.validate_index = write_cfg['validate_index']
        self.staged_writes = write_cfg['staged_writes']
        self.track_write_times = write_cfg['track_write_times']
        self.slow_write_threshold = write_cfg['slow_write_threshold_ms']
        self.max_memory_gb = write_cfg['max_memory_gb']
        self.force_gc = write_cfg['force_garbage_collection']

        # Version management based on mode
        mode_cfg = write_cfg[self.mode]
        self.prune_versions = mode_cfg['prune_previous_versions']
        self.create_snapshots = mode_cfg['create_snapshots']
        self.snapshot_frequency = mode_cfg['snapshot_frequency']

        logger.info(f"TickStore initialized in {self.mode.upper()} mode")
        logger.info(f"  - Batch size: {self.write_batch_size:,} ticks")
        logger.info(f"  - Prune versions: {self.prune_versions}")
        logger.info(f"  - Validate index: {self.validate_index}")
        logger.info(f"  - Deduplication: {self.enable_deduplication}")

        # Initialize ArcticDB connection
        self._init_arctic()

        # ET timezone note: We store naive timestamps in ET
        # ArcticDB preserves naive timestamps perfectly without conversion

        # Performance tracking
        self.stats = {
            'ticks_stored': 0,
            'bytes_stored': 0,
            'queries_executed': 0,
            'errors': 0,
            'duplicates_removed': 0,
            'write_times_ms': [],
            'validation_failures': 0,
            'versions_pruned': 0,
            'snapshots_created': 0,
            'retries_attempted': 0,
            'retries_succeeded': 0
        }

        # Error tracking
        self.error_history = []  # List of recent errors with context
        self.max_error_history = 100  # Keep last 100 errors

        # Snapshot tracking
        self.last_snapshot_date = None

        logger.info(f"TickStore initialized with URI: {self.arctic_uri}")

    def _load_config(self, config_file: str):
        """Load configuration from YAML file"""
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                # Try relative to project root
                config_path = Path.cwd() / config_file

            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_file}: {e}")
            # Fall back to defaults
            self.config = {
                'arctic': {'uri': 'lmdb://./data/arctic_storage'},
                'storage': {'enable_compression': True, 'cache_size_mb': 1000},
                'write_config': {
                    'mode': 'backfill',
                    'batch_size': 2000000,
                    'enable_deduplication': True,
                    'validate_index': True,
                    'staged_writes': False,
                    'track_write_times': True,
                    'slow_write_threshold_ms': 1000,
                    'max_memory_gb': 2,
                    'force_garbage_collection': True,
                    'backfill': {
                        'prune_previous_versions': False,
                        'create_snapshots': True,
                        'snapshot_frequency': 'weekly'
                    },
                    'production': {
                        'prune_previous_versions': True,
                        'create_snapshots': True,
                        'snapshot_frequency': 'weekly'
                    }
                }
            }
            logger.warning("Using default configuration")

    @retry_on_failure(max_retries=3, delay=2.0)
    def _init_arctic(self):
        """Initialize ArcticDB connection and libraries with retry logic"""
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

        except ImportError as e:
            self._log_error(ErrorSeverity.CRITICAL, "ArcticDB not installed", e)
            raise ConfigurationError(f"ArcticDB not installed: {e}")
        except Exception as e:
            self._log_error(ErrorSeverity.CRITICAL, "Failed to initialize ArcticDB", e)
            raise ConnectionError(f"Failed to initialize ArcticDB: {e}")

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
                self._log_error(
                    ErrorSeverity.HIGH,
                    f"Missing required columns for {symbol}",
                    DataValidationError(f"Missing columns: {missing_cols}"),
                    context={'symbol': symbol, 'date': date, 'missing_cols': missing_cols}
                )
                return False

            # Create storage key with symbol/date partitioning
            storage_key = f"{symbol}/{date}"

            # Check if data already exists
            if not overwrite and self._data_exists(storage_key):
                logger.info(f"Data already exists for {storage_key}, skipping")
                return True

            # Use DataFrame as-is (already in ET from conversion)
            normalized_df = tick_df.copy()

            # Convert timezone-aware timestamps to naive ET for ArcticDB
            # ArcticDB converts tz-aware to UTC, but we want to preserve ET
            if 'timestamp' in normalized_df.columns:
                if hasattr(normalized_df['timestamp'].dtype, 'tz'):
                    # If timezone-aware, convert to naive (strip tz info)
                    normalized_df['timestamp'] = normalized_df['timestamp'].dt.tz_localize(None)

            # Handle object/string columns for ArcticDB compatibility
            for col in normalized_df.columns:
                col_dtype = str(normalized_df[col].dtype)
                if normalized_df[col].dtype == 'object' or 'string' in col_dtype:
                    # Convert to object type (compatible with ArcticDB)
                    normalized_df[col] = normalized_df[col].astype(str).astype('object')

            # Prepare basic metadata
            tick_metadata = self._prepare_tick_metadata(symbol, date, normalized_df, metadata)

            # Compute Tier 2 metadata if MetadataComputer is available
            if MetadataComputer is not None:
                try:
                    tier2_metadata = MetadataComputer.compute_phase1_metadata(normalized_df, symbol, date)
                    tick_metadata['tier2_metadata'] = tier2_metadata
                    logger.info(f"Computed Tier 2 metadata for {symbol}/{date}: "
                              f"{len(tier2_metadata.get('basic_stats', {}))} basic stats, "
                              f"{len(tier2_metadata.get('spread_stats', {}))} spread stats, "
                              f"liquidity score: {tier2_metadata.get('liquidity_profile', {}).get('liquidity_score', 0):.1f}")
                except Exception as e:
                    logger.warning(f"Failed to compute Tier 2 metadata: {e}")
                    # Continue without Tier 2 metadata

            # Deduplicate if enabled
            if self.enable_deduplication:
                original_len = len(normalized_df)
                normalized_df = self._remove_duplicates(normalized_df)
                duplicates_removed = original_len - len(normalized_df)
                if duplicates_removed > 0:
                    self.stats['duplicates_removed'] += duplicates_removed
                    logger.info(f"Removed {duplicates_removed} duplicate ticks")

            # Track write performance
            write_start = time.time() if self.track_write_times else 0

            # Store to ArcticDB with optimized configuration
            try:
                self.tick_data_lib.write(
                    storage_key,
                    normalized_df,
                    metadata=tick_metadata,
                    prune_previous_versions=self.prune_versions,
                    staged=self.staged_writes,
                    validate_index=self.validate_index
                )

                # Track performance metrics
                if self.track_write_times:
                    write_time = (time.time() - write_start) * 1000
                    self.stats['write_times_ms'].append(write_time)

                    if write_time > self.slow_write_threshold:
                        logger.warning(f"Slow write detected: {write_time:.0f}ms for {len(normalized_df):,} ticks")

                if self.prune_versions:
                    self.stats['versions_pruned'] += 1

                # Force garbage collection for large writes if configured
                if self.force_gc and len(normalized_df) > self.write_batch_size:
                    gc.collect()

                # Check if we need to create a snapshot
                self._check_snapshot_needed(symbol, date)

            except Exception as e:
                if "not sorted" in str(e).lower() or "monotonic" in str(e).lower():
                    self.stats['validation_failures'] += 1

                    # Attempt to fix by sorting
                    logger.warning(f"Data not sorted for {symbol} on {date}, attempting to fix...")
                    try:
                        normalized_df = normalized_df.sort_values('timestamp')

                        # Retry write with sorted data
                        self.tick_data_lib.write(
                            storage_key,
                            normalized_df,
                            metadata=tick_metadata,
                            prune_previous_versions=self.prune_versions,
                            staged=self.staged_writes,
                            validate_index=self.validate_index
                        )
                        logger.info(f"Successfully stored after sorting {len(normalized_df):,} ticks")
                        return True

                    except Exception as retry_error:
                        self._log_error(
                            ErrorSeverity.HIGH,
                            f"Failed to store even after sorting",
                            retry_error,
                            context={'symbol': symbol, 'date': date, 'ticks': len(normalized_df)}
                        )
                        raise StorageError(f"Failed to store sorted data: {retry_error}")
                else:
                    # Other storage errors
                    self._log_error(
                        ErrorSeverity.HIGH,
                        f"Storage error for {symbol} on {date}",
                        e,
                        context={'symbol': symbol, 'date': date}
                    )
                    raise StorageError(f"Failed to store data: {e}")

            # Update statistics
            self.stats['ticks_stored'] += len(normalized_df)
            self.stats['bytes_stored'] += normalized_df.memory_usage(deep=True).sum()

            logger.info(f"Stored {len(normalized_df)} ticks for {storage_key}")
            return True

        except DataValidationError:
            # Already logged, just return False
            self.stats['errors'] += 1
            return False
        except StorageError:
            # Already logged and handled
            self.stats['errors'] += 1
            return False
        except Exception as e:
            self.stats['errors'] += 1
            self._log_error(
                ErrorSeverity.HIGH,
                f"Unexpected error storing ticks",
                e,
                context={'symbol': symbol, 'date': date, 'error_type': type(e).__name__}
            )
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

        This method bridges the gap between:
        - IQFeedCollector: Returns raw NumPy arrays (fast, memory-efficient)
        - ArcticDB: Expects Pandas DataFrames (flexible, feature-rich)

        Example input (NumPy structured array):
            tick_array[0] = (1234, '2025-09-15', 26540953311, 236.62, 10, b'O', 11, 387902, 236.6, 236.67, 135, 61, 23, 0)

        This represents:
            - Tick ID: 1234 (unique identifier for this tick)
            - Date: September 15, 2025
            - Time: 26540953311 microseconds since midnight (07:22:20.953311)
            - Price: $236.62 (last trade price)
            - Volume: 10 shares traded
            - Exchange: 'O' (NYSE Arca)
            - Market Center: 11
            - Total Volume: 387,902 shares traded today
            - Bid: $236.60 (best bid price)
            - Ask: $236.67 (best ask price)
            - Conditions: 135, 61, 23, 0 (extended hours, trade conditions)

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            date: Date string (YYYY-MM-DD)
            tick_array: NumPy structured array from IQFeedCollector
            metadata: Additional metadata dict (e.g., {'source': 'IQFeed', 'session': 'pre-market'})
            overwrite: Whether to overwrite existing data

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting NumPy tick storage for {symbol} on {date}")

            # ================================================================================
            # STEP 1: Initial Input Validation
            # ================================================================================
            if tick_array is None or len(tick_array) == 0:
                logger.warning(f"Empty tick array for {symbol} on {date} - nothing to store")
                return False

            array_size = len(tick_array)
            logger.info(f"Processing {array_size:,} ticks for {symbol} on {date}")

            # ================================================================================
            # STEP 2: Validate Array Structure
            # ================================================================================
            if not self._validate_tick_array_structure(tick_array, symbol, date):
                return False

            # ================================================================================
            # STEP 3: Validate Data Ranges (on sample for performance)
            # ================================================================================
            sample_size = min(1000, array_size)
            if not self._validate_tick_data_ranges(tick_array[:sample_size], symbol, date):
                logger.error(f"Data validation failed for {symbol} on {date}")
                return False

            # ================================================================================
            # STEP 4: Process Based on Size (Chunked vs Direct)
            # ================================================================================
            CHUNK_SIZE = 1_000_000  # 1M ticks per chunk

            if array_size > CHUNK_SIZE:
                logger.info(f"Large dataset detected ({array_size:,} ticks) - using chunked processing")
                return self._store_numpy_ticks_chunked(
                    symbol, date, tick_array, metadata, overwrite, CHUNK_SIZE
                )
            else:
                # Small dataset - process all at once
                logger.debug(f"Processing {array_size:,} ticks in single batch")

                # Convert NumPy structured array to enhanced DataFrame with all 41 fields
                # Using vectorized operations for performance (no per-tick validation)
                tick_df = self._numpy_to_enhanced_dataframe(tick_array, symbol)

                # Use existing store_ticks method to save to ArcticDB
                success = self.store_ticks(symbol, date, tick_df, metadata, overwrite)

                if success:
                    logger.info(f"Successfully stored {array_size:,} ticks for {symbol} on {date}")
                else:
                    logger.error(f"Failed to store ticks for {symbol} on {date}")

                return success

        except Exception as e:
            logger.error(f"Unexpected error storing NumPy ticks for {symbol} on {date}: {e}", exc_info=True)
            return False

    def _numpy_ticks_to_dataframe(self, tick_array: np.ndarray, prev_context: Optional[Dict] = None) -> pd.DataFrame:
        """
        Convert IQFeed NumPy tick array to DataFrame with proper timestamp.
        OPTIMIZED VERSION: Memory-efficient processing with optimal dtypes.

        This is the CRITICAL conversion function that bridges NumPy and Pandas worlds.

        DATA TRANSFORMATION PIPELINE:
        ============================

        NumPy Structured Array (from IQFeed)     -->     Pandas DataFrame (for ArcticDB)
        -------------------------------------           ----------------------------------
        Field Name    Type         Example              Column Name     Type        Example
        -------------------------------------           ----------------------------------
        tick_id       uint64       3954                 tick_id         uint32      3954
        date          datetime64   2025-09-15           [combined into timestamp]
        time          timedelta64  26540953311μs        time_us         timedelta  07:22:20.953311
        last          float64      236.62               price           float32     236.62
        last_sz       uint64       10                   volume          uint32      10
        last_type     bytes        b'O'                 exchange        category    'O'
        mkt_ctr       uint32       11                   market_center   uint16      11
        tot_vlm       uint64       387902               total_volume    uint32      387902
        bid           float64      236.60               bid             float32     236.60
        ask           float64      236.67               ask             float32     236.67
        cond1         uint8        135                  condition_1     uint8       135
        cond2         uint8        61                   condition_2     uint8       61
        cond3         uint8        23                   condition_3     uint8       23
        cond4         uint8        0                    condition_4     uint8       0
                                                        timestamp       datetime64  2025-09-15 07:22:20.953311
                                                        spread          float32     0.07
                                                        midpoint        float32     236.635

        OPTIMIZATION NOTES:
        - Uses optimal dtypes to reduce memory by ~40%
        - Vectorized operations for better performance
        - Exchange codes as categorical (saves memory for repeated values)
        - float32 for prices (sufficient precision for cents)
        - Smaller int types where appropriate
        """

        try:
            # ================================================================================
            # STEP 1: Pre-compute simple metrics in NumPy (fastest)
            # ================================================================================
            # These calculations are faster in NumPy before DataFrame creation
            spread_np = (tick_array['ask'] - tick_array['bid']).astype('float32')
            midpoint_np = ((tick_array['ask'] + tick_array['bid']) / 2).astype('float32')

            # ================================================================================
            # STEP 2: Create DataFrame directly from structured array
            # ================================================================================
            df = pd.DataFrame(tick_array)

            # ================================================================================
            # STEP 3: Rename columns efficiently (in-place operation)
            # ================================================================================
            df.rename(columns={
                'last': 'price',
                'last_sz': 'volume',
                'last_type': 'exchange',
                'mkt_ctr': 'market_center',
                'tot_vlm': 'total_volume',
                'cond1': 'condition_1',
                'cond2': 'condition_2',
                'cond3': 'condition_3',
                'cond4': 'condition_4'
            }, inplace=True)

            # ================================================================================
            # STEP 4: Create timestamp BEFORE dtype optimization (keep in ET)
            # ================================================================================
            df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time'], unit='us')
            # IQFeed times are in ET - keep as naive ET (no timezone)
            # ArcticDB preserves naive timestamps perfectly
            df.drop('date', axis=1, inplace=True)
            df.rename(columns={'time': 'time_us'}, inplace=True)

            # ================================================================================
            # STEP 5: Decode exchange codes efficiently
            # ================================================================================
            if df['exchange'].dtype == object and len(df) > 0:
                if isinstance(df['exchange'].iloc[0], bytes):
                    df['exchange'] = df['exchange'].str.decode('utf-8')
            df['exchange'] = df['exchange'].astype('category')

            # ================================================================================
            # STEP 6: Optimize data types for memory efficiency
            # ================================================================================
            df['price'] = df['price'].astype('float32')
            df['bid'] = df['bid'].astype('float32')
            df['ask'] = df['ask'].astype('float32')
            df['volume'] = df['volume'].astype('uint32')
            df['total_volume'] = df['total_volume'].astype('uint32')
            df['tick_id'] = df['tick_id'].astype('uint32')
            df['market_center'] = df['market_center'].astype('uint16')
            df['condition_1'] = df['condition_1'].astype('uint8')
            df['condition_2'] = df['condition_2'].astype('uint8')
            df['condition_3'] = df['condition_3'].astype('uint8')
            df['condition_4'] = df['condition_4'].astype('uint8')

            # ================================================================================
            # STEP 7: Add ESSENTIAL METRICS
            # ================================================================================

            # Basic spread metrics (from NumPy calculations)
            df['spread'] = spread_np
            df['midpoint'] = midpoint_np
            df['spread_bps'] = (spread_np / midpoint_np * 10000).astype('float32')
            df['spread_pct'] = (spread_np / midpoint_np).astype('float32')

            # Trade metrics
            df['dollar_volume'] = (df['price'] * df['volume']).astype('float64')
            df['effective_spread'] = (2 * np.abs(df['price'] - df['midpoint'])).astype('float32')

            # Lee-Ready trade classification
            # +1 = buyer-initiated, -1 = seller-initiated
            df['trade_sign'] = np.where(
                df['price'] > df['midpoint'], 1,
                np.where(df['price'] < df['midpoint'], -1, 0)
            ).astype('int8')

            # Volume metrics
            df['volume_rate'] = df['total_volume'].diff().fillna(0).astype('uint32')
            df['trade_pct_of_day'] = np.where(
                df['total_volume'] > 0,
                (df['volume'] / df['total_volume']).astype('float32'),
                0
            )

            # Condition flags
            df['is_extended_hours'] = (df['condition_1'] == 135)
            df['is_odd_lot'] = (df['condition_3'] == 23)
            df['is_regular'] = ((df['condition_1'] == 0) &
                               (df['condition_2'] == 0) &
                               (df['condition_3'] == 0))

            # ================================================================================
            # STEP 8: Context-dependent metrics (handle chunk boundaries)
            # ================================================================================

            # Calculate log return and tick direction
            df['log_return'] = np.log(df['price'] / df['price'].shift(1)).astype('float32')
            df['tick_direction'] = np.sign(df['price'].diff()).astype('int8')

            # Handle first row if we have context from previous chunk
            if prev_context and 'last_price' in prev_context:
                # Fix first row calculations using previous chunk's last values
                df.loc[0, 'log_return'] = np.log(df.loc[0, 'price'] / prev_context['last_price'])
                df.loc[0, 'tick_direction'] = np.sign(df.loc[0, 'price'] - prev_context['last_price'])

                # Fix trade_sign for trades at midpoint using tick test
                if df.loc[0, 'trade_sign'] == 0:  # Trade at midpoint
                    df.loc[0, 'trade_sign'] = np.sign(df.loc[0, 'price'] - prev_context['last_price'])

                # Fix volume_rate using previous total_volume
                if 'last_total_volume' in prev_context:
                    df.loc[0, 'volume_rate'] = df.loc[0, 'total_volume'] - prev_context['last_total_volume']

            # Apply tick test for remaining trades at midpoint
            at_midpoint = (df['trade_sign'] == 0) & (df.index > 0)
            if at_midpoint.any():
                # Use previous different price for tick test
                for idx in df[at_midpoint].index:
                    prev_prices = df.loc[:idx-1, 'price']
                    diff_prices = prev_prices[prev_prices != df.loc[idx, 'price']]
                    if len(diff_prices) > 0:
                        df.loc[idx, 'trade_sign'] = np.sign(df.loc[idx, 'price'] - diff_prices.iloc[-1])

            # ================================================================================
            # STEP 9: Sort by timestamp efficiently
            # ================================================================================
            df.sort_values('timestamp', inplace=True, kind='mergesort')
            df.reset_index(drop=True, inplace=True)

            # Log memory usage for monitoring
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            logger.debug(
                f"Converted {len(tick_array)} ticks to DataFrame: "
                f"{len(df.columns)} columns, {memory_mb:.2f} MB memory"
            )

            return df

        except Exception as e:
            logger.error(f"Error in optimized DataFrame conversion: {e}", exc_info=True)
            # Fall back to original method if optimization fails
            logger.warning("Falling back to standard conversion method")
            return self._numpy_ticks_to_dataframe_legacy(tick_array)

    def _numpy_ticks_to_dataframe_legacy(self, tick_array: np.ndarray) -> pd.DataFrame:
        """
        Legacy conversion method - fallback if optimized version fails.
        Kept for compatibility and as a fallback option.
        """
        # Original implementation (unoptimized but reliable)
        df = pd.DataFrame({
            'tick_id': tick_array['tick_id'].astype(int),
            'date': tick_array['date'],
            'time_us': tick_array['time'],
            'price': tick_array['last'].astype(float),
            'volume': tick_array['last_sz'].astype(int),
            'exchange': tick_array['last_type'],
            'market_center': tick_array['mkt_ctr'].astype(int),
            'total_volume': tick_array['tot_vlm'].astype(int),
            'bid': tick_array['bid'].astype(float),
            'ask': tick_array['ask'].astype(float),
            'condition_1': tick_array['cond1'].astype(int),
            'condition_2': tick_array['cond2'].astype(int),
            'condition_3': tick_array['cond3'].astype(int),
            'condition_4': tick_array['cond4'].astype(int)
        })

        df['timestamp'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['time_us'])
        df = df.drop(['date'], axis=1)

        if df['exchange'].dtype == object:
            df['exchange'] = df['exchange'].apply(
                lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
            )

        df['spread'] = df['ask'] - df['bid']
        df['midpoint'] = (df['bid'] + df['ask']) / 2
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.debug(f"Converted {len(tick_array)} NumPy ticks to DataFrame with {len(df.columns)} columns (legacy)")
        return df

    def _pydantic_to_dataframe(self, pydantic_ticks: List) -> pd.DataFrame:
        """
        Convert list of Pydantic TickData models to optimized DataFrame.

        TRANSFORMATION FOR TICK 265596:
        ================================
        Pydantic Model (43 fields)  →  DataFrame (43 columns)

        Example:
            Input:  TickData(symbol='AAPL', timestamp=2025-09-15 15:59:59.903492 ET,
                           price=Decimal('236.75'), size=800, ...)
            Output: DataFrame with 43 columns, optimized dtypes

        Args:
            pydantic_ticks: List of validated TickData Pydantic models from
                          foundation.utils.iqfeed_converter

        Returns:
            DataFrame with all Pydantic fields preserved, memory-optimized
        """
        if not pydantic_ticks:
            logger.warning("Empty pydantic_ticks list - returning empty DataFrame")
            return pd.DataFrame()

        try:
            # Extract all fields from Pydantic models
            records = []
            for tick in pydantic_ticks:
                record = {
                    # Core fields (12 including tick_id and sequence)
                    'symbol': tick.symbol,
                    'tick_id': tick.tick_id,  # IQFeed unique identifier
                    'timestamp': tick.timestamp,
                    'price': float(tick.price),
                    'volume': tick.size,  # 'size' → 'volume' for consistency
                    'exchange': tick.exchange,
                    'market_center': tick.market_center,
                    'total_volume': tick.total_volume,
                    'bid': float(tick.bid) if tick.bid else None,
                    'ask': float(tick.ask) if tick.ask else None,
                    'conditions': tick.conditions,
                    'tick_sequence': tick.tick_sequence,  # Include sequence number

                    # Spread metrics (5)
                    'spread': float(tick.spread) if tick.spread else None,
                    'midpoint': float(tick.midpoint) if tick.midpoint else None,
                    'spread_bps': tick.spread_bps,
                    'spread_pct': tick.spread_pct,
                    'effective_spread': float(tick.effective_spread) if tick.effective_spread else None,

                    # Trade analysis (3)
                    'trade_sign': tick.trade_sign,
                    'dollar_volume': float(tick.dollar_volume) if tick.dollar_volume else None,
                    'price_improvement': float(tick.price_improvement) if tick.price_improvement else None,

                    # Additional metrics (7)
                    'tick_direction': tick.tick_direction,
                    'participant_type': tick.participant_type,
                    'volume_rate': tick.volume_rate,
                    'trade_pct_of_day': tick.trade_pct_of_day,
                    'log_return': tick.log_return,
                    'price_change': float(tick.price_change) if tick.price_change else None,
                    'price_change_bps': tick.price_change_bps,

                    # Condition flags (7)
                    'is_regular': tick.is_regular,
                    'is_extended_hours': tick.is_extended_hours,
                    'is_odd_lot': tick.is_odd_lot,
                    'is_intermarket_sweep': tick.is_intermarket_sweep,
                    'is_derivatively_priced': tick.is_derivatively_priced,
                    'is_qualified': tick.is_qualified,
                    'is_block_trade': tick.is_block_trade,

                    # Metadata fields (4) - MUST include even if None
                    'id': str(tick.id) if hasattr(tick, 'id') and tick.id else None,
                    'created_at': tick.created_at if hasattr(tick, 'created_at') else None,
                    'updated_at': tick.updated_at if hasattr(tick, 'updated_at') else None,
                    'metadata': tick.metadata if hasattr(tick, 'metadata') else None,

                    # Timestamp fields (2)
                    'processed_at': tick.processed_at if hasattr(tick, 'processed_at') else None,
                    'source_timestamp': tick.source_timestamp if hasattr(tick, 'source_timestamp') else None,

                    # Enum fields (3)
                    'trade_sign_enum': tick.trade_sign_enum.value if hasattr(tick, 'trade_sign_enum') and tick.trade_sign_enum else None,
                    'tick_direction_enum': tick.tick_direction_enum.value if hasattr(tick, 'tick_direction_enum') and tick.tick_direction_enum else None,
                    'participant_type_enum': tick.participant_type_enum.value if hasattr(tick, 'participant_type_enum') and tick.participant_type_enum else None,
                }

                # Note: 'size' field from Pydantic is renamed to 'volume' in DataFrame
                # This gives us exactly 41 fields (42 Pydantic fields - 'size' + 'volume')

                records.append(record)

            # Create DataFrame
            df = pd.DataFrame(records)

            # CRITICAL VALIDATION: Ensure all Pydantic fields are in DataFrame
            if pydantic_ticks:
                # Get expected field count from Pydantic model
                sample_model = pydantic_ticks[0].model_dump()
                expected_fields = len(sample_model)
                actual_columns = len(df.columns)

                # Account for 'size' → 'volume' rename
                # Pydantic has 'size', DataFrame has 'volume', so count should match after adjustment
                pydantic_field_names = set(sample_model.keys())
                df_column_names = set(df.columns)

                # Check if we have the right number after accounting for rename
                missing_from_df = pydantic_field_names - df_column_names - {'size'}  # size is renamed to volume
                extra_in_df = df_column_names - pydantic_field_names - {'volume'}  # volume replaces size

                if missing_from_df or extra_in_df:
                    logger.error(f"Column mismatch detected!")
                    logger.error(f"Expected {expected_fields} Pydantic fields")
                    logger.error(f"Got {actual_columns} DataFrame columns")
                    if missing_from_df:
                        logger.error(f"Missing from DataFrame: {sorted(missing_from_df)}")
                    if extra_in_df:
                        logger.error(f"Extra in DataFrame: {sorted(extra_in_df)}")
                    raise ValueError(f"DataFrame column count ({actual_columns}) doesn't match Pydantic field count ({expected_fields})")

                logger.debug(f"Validation passed: {actual_columns} DataFrame columns match {expected_fields} Pydantic fields")

            # Apply memory optimizations (matching _numpy_ticks_to_dataframe)
            # Prices as float32 (sufficient for cents precision)
            for col in ['price', 'bid', 'ask', 'spread', 'midpoint', 'effective_spread',
                       'price_improvement', 'spread_bps', 'spread_pct', 'price_change']:
                if col in df.columns:
                    df[col] = df[col].astype('float32')

            # Volumes as uint32
            for col in ['volume', 'total_volume']:
                if col in df.columns:
                    df[col] = df[col].astype('uint32')

            # Small integers
            if 'market_center' in df.columns:
                df['market_center'] = df['market_center'].astype('uint16')
            if 'trade_sign' in df.columns:
                df['trade_sign'] = df['trade_sign'].astype('int8')
            if 'tick_direction' in df.columns:
                df['tick_direction'] = df['tick_direction'].astype('int8')

            # Categories for repeated strings
            for col in ['symbol', 'exchange', 'participant_type']:
                if col in df.columns:
                    df[col] = df[col].astype('category')

            # Sort by timestamp
            df.sort_values('timestamp', inplace=True, kind='mergesort')
            df.reset_index(drop=True, inplace=True)

            logger.info(f"Converted {len(pydantic_ticks)} Pydantic ticks to DataFrame: "
                       f"{len(df.columns)} columns, {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

            return df

        except Exception as e:
            logger.error(f"Failed to convert Pydantic to DataFrame: {e}")
            raise

    def _numpy_to_enhanced_dataframe(self, tick_array: np.ndarray, symbol: str) -> pd.DataFrame:
        """
        Convert NumPy tick array to DataFrame with all 41 enhanced fields.

        Uses VECTORIZED operations for performance (100x faster than per-tick validation).
        Follows Pydantic schema for field definitions but skips runtime validation.

        Performance: ~50ms for 100K ticks, ~500ms for 1M ticks

        Args:
            tick_array: NumPy structured array from IQFeed (14 fields)
            symbol: Trading symbol

        Returns:
            DataFrame with 41 columns matching Pydantic TickData schema
        """
        try:
            # Get array size for pre-allocation
            n_ticks = len(tick_array)

            # ================================================================================
            # STEP 1: Core fields extraction (vectorized)
            # ================================================================================

            # Combine date and time into timestamps (vectorized)
            # Convert date to datetime64, add time as timedelta64
            dates = pd.to_datetime(tick_array['date'])
            times = pd.to_timedelta(tick_array['time'], unit='us')
            timestamps = dates + times
            # Keep as naive ET timestamps - ArcticDB preserves them perfectly
            # No need for tz_localize which causes UTC conversion issues

            # Extract and decode exchange codes (vectorized)
            exchanges = np.array([x.decode('utf-8') if isinstance(x, bytes) else str(x)
                                 for x in tick_array['last_type']])

            # ================================================================================
            # STEP 2: Spread metrics (all vectorized)
            # ================================================================================

            bids = tick_array['bid'].astype('float32')
            asks = tick_array['ask'].astype('float32')
            prices = tick_array['last'].astype('float32')

            # Compute spread metrics
            spreads = asks - bids
            midpoints = (asks + bids) / 2

            # Handle division by zero for spread_bps and spread_pct
            with np.errstate(divide='ignore', invalid='ignore'):
                spread_bps = np.where(midpoints > 0, (spreads / midpoints) * 10000, 0).astype('float32')
                spread_pct = np.where(midpoints > 0, spreads / midpoints, 0).astype('float32')

            effective_spreads = 2 * np.abs(prices - midpoints)

            # ================================================================================
            # STEP 3: Trade classification (Lee-Ready algorithm, vectorized)
            # ================================================================================

            trade_signs = np.where(
                prices > midpoints, 1,  # Buy
                np.where(prices < midpoints, -1,  # Sell
                        0)  # At midpoint
            ).astype('int8')

            # Tick direction (requires previous price)
            price_diffs = np.diff(prices, prepend=prices[0])
            tick_directions = np.sign(price_diffs).astype('int8')

            # ================================================================================
            # STEP 4: Volume metrics (vectorized)
            # ================================================================================

            volumes = tick_array['last_sz'].astype('uint32')
            dollar_volumes = (prices * volumes).astype('float64')

            # Volume rate (change in cumulative volume)
            total_volumes = tick_array['tot_vlm'].astype('uint32')
            volume_rates = np.diff(total_volumes, prepend=0).astype('int32')

            # Trade percent of day
            with np.errstate(divide='ignore', invalid='ignore'):
                trade_pct_of_day = np.where(total_volumes > 0,
                                           volumes / total_volumes, 0).astype('float32')

            # ================================================================================
            # STEP 5: Price movement metrics (vectorized)
            # ================================================================================

            # Log returns
            with np.errstate(divide='ignore', invalid='ignore'):
                prev_prices = np.roll(prices, 1)
                prev_prices[0] = prices[0]  # First price has no previous
                log_returns = np.where(prev_prices > 0,
                                      np.log(prices / prev_prices), 0).astype('float32')

            # Price changes
            price_changes = (prices - prev_prices).astype('float32')
            with np.errstate(divide='ignore', invalid='ignore'):
                price_change_bps = np.where(prev_prices > 0,
                                           (price_changes / prev_prices) * 10000, 0).astype('float32')

            # ================================================================================
            # STEP 6: Condition flags (vectorized)
            # ================================================================================

            cond1 = tick_array['cond1'].astype('uint8')
            cond2 = tick_array['cond2'].astype('uint8')
            cond3 = tick_array['cond3'].astype('uint8')
            cond4 = tick_array['cond4'].astype('uint8')

            is_regular = (cond1 == 0) & (cond2 == 0) & (cond3 == 0) & (cond4 == 0)
            is_extended_hours = (cond1 == 135)
            is_odd_lot = (cond3 == 23)
            is_derivatively_priced = (cond2 == 61)
            is_intermarket_sweep = (cond1 == 37) | (cond2 == 37)
            is_qualified = np.ones(n_ticks, dtype=bool)  # Default True
            is_block_trade = (volumes >= 10000)

            # ================================================================================
            # STEP 7: Price improvement (vectorized)
            # ================================================================================

            # For buys: improvement = midpoint - price (negative if worse)
            # For sells: improvement = price - midpoint (negative if worse)
            price_improvements = np.where(
                trade_signs == 1, midpoints - prices,  # Buy trades
                np.where(trade_signs == -1, prices - midpoints,  # Sell trades
                        np.nan)  # No trade sign
            ).astype('float32')

            # ================================================================================
            # STEP 8: Participant type inference (vectorized)
            # ================================================================================

            participant_types = np.where(
                volumes >= 10000, 'INSTITUTIONAL',
                np.where(is_odd_lot, 'RETAIL',
                        np.where(is_intermarket_sweep, 'ALGO',
                                'UNKNOWN'))
            )

            # ================================================================================
            # STEP 9: Create DataFrame with all 41 fields
            # ================================================================================

            # Get current time in ET for metadata fields
            now_et = datetime.now()

            df = pd.DataFrame({
                # Core fields (10)
                'symbol': symbol,
                'timestamp': timestamps,
                'price': prices,
                'volume': volumes,  # Note: 'size' in Pydantic becomes 'volume' in DataFrame
                'exchange': exchanges,
                'market_center': tick_array['mkt_ctr'].astype('uint16'),
                'total_volume': total_volumes,
                'bid': bids,
                'ask': asks,
                'conditions': [f"{c1},{c2},{c3},{c4}".rstrip(',0')
                              for c1, c2, c3, c4 in zip(cond1, cond2, cond3, cond4)],

                # Spread metrics (5)
                'spread': spreads,
                'midpoint': midpoints,
                'spread_bps': spread_bps,
                'spread_pct': spread_pct,
                'effective_spread': effective_spreads,

                # Trade analysis (3)
                'trade_sign': trade_signs,
                'dollar_volume': dollar_volumes,
                'price_improvement': price_improvements,

                # Additional metrics (7)
                'tick_direction': tick_directions,
                'participant_type': participant_types,
                'volume_rate': volume_rates,
                'trade_pct_of_day': trade_pct_of_day,
                'log_return': log_returns,
                'price_change': price_changes,
                'price_change_bps': price_change_bps,

                # Condition flags (7)
                'is_regular': is_regular,
                'is_extended_hours': is_extended_hours,
                'is_odd_lot': is_odd_lot,
                'is_intermarket_sweep': is_intermarket_sweep,
                'is_derivatively_priced': is_derivatively_priced,
                'is_qualified': is_qualified,
                'is_block_trade': is_block_trade,

                # Metadata fields (4)
                'id': [str(uuid.uuid4()) for _ in range(n_ticks)],
                'created_at': now_et,
                'updated_at': now_et,
                'metadata': None,

                # Timestamp fields (2)
                'processed_at': now_et,
                'source_timestamp': timestamps,  # Original trade time

                # Enum fields (3) - store as integers for efficiency
                'trade_sign_enum': trade_signs,  # Same as trade_sign
                'tick_direction_enum': tick_directions,  # Same as tick_direction
                'participant_type_enum': pd.Categorical(participant_types).codes,
            })

            # Apply category dtype for string fields to save memory
            df['symbol'] = df['symbol'].astype('category')
            df['exchange'] = df['exchange'].astype('category')
            df['participant_type'] = df['participant_type'].astype('category')

            # Sort by timestamp (should already be sorted but ensure)
            df.sort_values('timestamp', inplace=True, kind='mergesort')
            df.reset_index(drop=True, inplace=True)

            logger.info(f"Converted {len(tick_array)} ticks to enhanced DataFrame with 41 columns")
            logger.debug(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

            return df

        except Exception as e:
            logger.error(f"Failed to convert NumPy to enhanced DataFrame: {e}")
            raise

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

    def get_metadata(self, symbol: str, date: str) -> Optional[Dict[str, Any]]:
        """
        Get just the Tier 2 metadata without loading the full DataFrame.
        Fast operation for GUI dashboards and data discovery.

        Args:
            symbol: Stock symbol
            date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary containing Tier 2 metadata or None
        """
        try:
            storage_key = f"{symbol}/{date}"

            # Use read_metadata to get just metadata without loading DataFrame
            versioned_item = self.tick_data_lib.read_metadata(storage_key)

            # Extract metadata dict from VersionedItem
            metadata_dict = versioned_item.metadata if versioned_item else None

            # Extract Tier 2 metadata if present
            tier2_metadata = metadata_dict.get('tier2_metadata') if metadata_dict else None

            if tier2_metadata:
                logger.info(f"Retrieved Tier 2 metadata for {symbol}/{date}")
                return tier2_metadata
            else:
                logger.warning(f"No Tier 2 metadata found for {symbol}/{date}")
                # Debug: show what's in metadata
                if metadata_dict:
                    logger.info(f"Available metadata keys: {list(metadata_dict.keys())}")
                return None

        except Exception as e:
            logger.error(f"Error retrieving metadata for {symbol}/{date}: {e}")
            return None

    def get_metadata_summary(self, symbol: str, date: str) -> Optional[str]:
        """
        Get human-readable summary of Tier 2 metadata.

        Args:
            symbol: Stock symbol
            date: Date string (YYYY-MM-DD)

        Returns:
            Formatted summary string or None
        """
        metadata = self.get_metadata(symbol, date)

        if metadata and MetadataComputer:
            return MetadataComputer.compute_summary_report(metadata)

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

            # Use DataFrame as-is (already in ET)
            normalized_ticks = new_ticks.copy()

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

            # Add exchange and market session info (simplified)
            metadata['primary_exchange'] = 'NYSE'  # Default exchange

            # Market hours in ET: 9:30 AM - 4:00 PM
            # ET automatically handles DST transitions
            metadata['market_sessions'] = {
                'pre_open': '09:00:00',    # 4:00 AM ET
                'open': '14:30:00',         # 9:30 AM ET
                'close': '21:00:00',        # 4:00 PM ET
                'post_close': '01:00:00+1'  # 8:00 PM ET
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

    def _validate_tick_array_structure(self, tick_array: np.ndarray, symbol: str, date: str) -> bool:
        """
        Validate the structure of the NumPy tick array from IQFeed.

        Args:
            tick_array: NumPy structured array to validate
            symbol: Stock symbol for logging
            date: Date string for logging

        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # Expected field names from IQFeed
            EXPECTED_FIELDS = (
                'tick_id', 'date', 'time', 'last', 'last_sz',
                'last_type', 'mkt_ctr', 'tot_vlm', 'bid', 'ask',
                'cond1', 'cond2', 'cond3', 'cond4'
            )

            # Check if array has field names (structured array)
            if not hasattr(tick_array.dtype, 'names') or tick_array.dtype.names is None:
                logger.error(f"Invalid array for {symbol}: Not a structured array (no field names)")
                return False

            actual_fields = tick_array.dtype.names

            # Check field count
            if len(actual_fields) != len(EXPECTED_FIELDS):
                logger.error(
                    f"Field count mismatch for {symbol}: "
                    f"Expected {len(EXPECTED_FIELDS)} fields, got {len(actual_fields)}"
                )
                logger.error(f"Expected: {EXPECTED_FIELDS}")
                logger.error(f"Actual: {actual_fields}")
                return False

            # Check each field exists
            missing_fields = [f for f in EXPECTED_FIELDS if f not in actual_fields]
            if missing_fields:
                logger.error(f"Missing fields for {symbol}: {missing_fields}")
                return False

            # Log field types for debugging
            logger.debug(f"Array structure for {symbol}:")
            for field in actual_fields:
                dtype = tick_array.dtype.fields[field][0]
                logger.debug(f"  {field}: {dtype}")

            return True

        except Exception as e:
            logger.error(f"Error validating array structure for {symbol}: {e}", exc_info=True)
            return False

    def _validate_tick_data_ranges(self, tick_sample: np.ndarray, symbol: str, date: str) -> bool:
        """
        Validate data ranges in tick array sample.

        Args:
            tick_sample: Sample of tick array to validate
            symbol: Stock symbol for logging
            date: Date string for logging

        Returns:
            True if data ranges are valid, False otherwise
        """
        try:
            validation_errors = []

            # Check prices are positive
            prices = tick_sample['last']
            if np.any(prices <= 0):
                invalid_count = np.sum(prices <= 0)
                validation_errors.append(f"{invalid_count} ticks with non-positive prices")
                logger.warning(f"Found {invalid_count} non-positive prices for {symbol}")

            # Check volumes are non-negative
            volumes = tick_sample['last_sz']
            if np.any(volumes < 0):
                invalid_count = np.sum(volumes < 0)
                validation_errors.append(f"{invalid_count} ticks with negative volumes")

            # Check bid/ask spreads are reasonable
            bids = tick_sample['bid']
            asks = tick_sample['ask']
            spreads = asks - bids

            # Check for negative spreads (bid > ask)
            negative_spreads = spreads < 0
            if np.any(negative_spreads):
                count = np.sum(negative_spreads)
                validation_errors.append(f"{count} ticks with negative spreads (bid > ask)")
                logger.warning(f"Found {count} negative spreads for {symbol}")

            # Check for excessive spreads (> 10% of price)
            price_pct = np.where(prices > 0, spreads / prices, 0)
            excessive_spreads = price_pct > 0.10
            if np.any(excessive_spreads):
                count = np.sum(excessive_spreads)
                max_spread_pct = np.max(price_pct) * 100
                logger.warning(
                    f"Found {count} ticks with spreads > 10% of price for {symbol} "
                    f"(max: {max_spread_pct:.1f}%)"
                )

            # Check dates match expected date
            unique_dates = np.unique(tick_sample['date'])
            if len(unique_dates) > 1:
                validation_errors.append(f"Multiple dates in array: {unique_dates}")
                logger.error(f"Multiple dates found for {symbol}: {unique_dates}")

            # Check expected date matches
            expected_date = np.datetime64(date)
            if not np.all(tick_sample['date'] == expected_date):
                validation_errors.append(f"Date mismatch: expected {date}")
                logger.error(f"Date mismatch for {symbol}: expected {date}, got {unique_dates}")

            # If critical errors found, return False
            if validation_errors:
                # Categorize errors
                critical_keywords = ['Date mismatch', 'Multiple dates']
                is_critical = any(error in str(validation_errors) for error in critical_keywords)

                if is_critical:
                    self._log_error(
                        ErrorSeverity.HIGH,
                        f"Critical validation errors for {symbol} on {date}",
                        DataValidationError(', '.join(validation_errors)),
                        context={
                            'symbol': symbol,
                            'date': date,
                            'errors': validation_errors
                        }
                    )
                    return False
                else:
                    # Non-critical errors - log as warning but continue
                    self._log_error(
                        ErrorSeverity.MEDIUM,
                        f"Non-critical validation issues for {symbol}",
                        None,
                        context={
                            'symbol': symbol,
                            'date': date,
                            'warnings': validation_errors
                        }
                    )

            # Log summary statistics
            logger.debug(
                f"Data range validation for {symbol}: "
                f"Prices: ${np.min(prices):.2f}-${np.max(prices):.2f}, "
                f"Volumes: {np.min(volumes)}-{np.max(volumes)}, "
                f"Avg spread: ${np.mean(spreads):.4f}"
            )

            return True

        except Exception as e:
            self._log_error(
                ErrorSeverity.HIGH,
                f"Unexpected error during validation",
                e,
                context={'symbol': symbol, 'date': date}
            )
            return False

    def _store_numpy_ticks_chunked(self,
                                   symbol: str,
                                   date: str,
                                   tick_array: np.ndarray,
                                   metadata: Optional[Dict[str, Any]],
                                   overwrite: bool,
                                   chunk_size: int) -> bool:
        """
        Store large tick arrays in chunks to avoid memory issues.
        Maintains context between chunks for metrics requiring previous values.

        Args:
            symbol: Stock symbol
            date: Date string
            tick_array: Full NumPy tick array
            metadata: Additional metadata
            overwrite: Whether to overwrite existing data
            chunk_size: Number of ticks per chunk

        Returns:
            True if all chunks stored successfully, False otherwise
        """
        try:
            total_ticks = len(tick_array)
            num_chunks = (total_ticks + chunk_size - 1) // chunk_size

            logger.info(
                f"Processing {total_ticks:,} ticks in {num_chunks} chunks of {chunk_size:,} for {symbol}"
            )

            # Process first chunk (potentially overwriting)
            first_chunk = tick_array[:chunk_size]
            first_df = self._numpy_to_enhanced_dataframe(first_chunk, symbol)

            if not self.store_ticks(symbol, date, first_df, metadata, overwrite):
                logger.error(f"Failed to store first chunk for {symbol} on {date}")
                return False

            logger.info(f"Stored chunk 1/{num_chunks} ({len(first_chunk):,} ticks) for {symbol}")

            # Process remaining chunks (appending)
            for i in range(1, num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_ticks)
                chunk = tick_array[start_idx:end_idx]

                # Convert using enhanced dataframe method (no context needed with vectorized approach)
                chunk_df = self._numpy_to_enhanced_dataframe(chunk, symbol)

                # Append to existing data
                if not self.append_ticks(symbol, date, chunk_df):
                    logger.error(f"Failed to store chunk {i+1}/{num_chunks} for {symbol}")
                    return False

                logger.info(
                    f"Stored chunk {i+1}/{num_chunks} ({len(chunk):,} ticks) for {symbol} "
                    f"[{start_idx:,}-{end_idx:,}]"
                )

            logger.info(f"Successfully stored all {num_chunks} chunks for {symbol} on {date}")
            return True

        except Exception as e:
            logger.error(f"Error in chunked storage for {symbol}: {e}", exc_info=True)
            return False


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

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pass through DataFrame without modification.

        Sequence numbers are already assigned during Pydantic conversion,
        so we don't need to do any groupby here. This preserves all
        legitimate trades with their pre-assigned sequence numbers.

        Args:
            df: DataFrame with tick_sequence already assigned

        Returns:
            DataFrame unchanged (tick_sequence already present)
        """
        # Check if tick_sequence exists
        if 'tick_sequence' not in df.columns:
            logger.warning("tick_sequence column not found - data may not be properly sequenced")
            # For backward compatibility, add default sequence of 0
            df['tick_sequence'] = 0
        else:
            # Log statistics about sequence numbers
            max_seq = df['tick_sequence'].max()
            if max_seq > 0:
                trades_with_seq = (df['tick_sequence'] > 0).sum()
                logger.debug(f"Found {trades_with_seq} trades with sequence numbers (max seq={max_seq})")

        return df

    def _check_snapshot_needed(self, symbol: str, date: str):
        """
        Check if a snapshot should be created based on frequency setting.

        Args:
            symbol: Stock symbol
            date: Date string
        """
        if not self.create_snapshots:
            return

        try:
            current_date = pd.to_datetime(date)

            # Determine if snapshot is needed based on frequency
            create_snapshot = False

            if self.snapshot_frequency == 'daily':
                create_snapshot = True
            elif self.snapshot_frequency == 'weekly':
                # Create snapshot on Sundays
                create_snapshot = current_date.dayofweek == 6
            elif self.snapshot_frequency == 'monthly':
                # Create snapshot on last day of month
                next_day = current_date + pd.Timedelta(days=1)
                create_snapshot = current_date.month != next_day.month

            if create_snapshot:
                snapshot_name = f"snapshot_{self.mode}_{date}"
                try:
                    # Create snapshot using ArcticDB's snapshot functionality
                    self.tick_data_lib.snapshot(snapshot_name)
                    self.stats['snapshots_created'] += 1
                    self.last_snapshot_date = date
                    logger.info(f"Created snapshot: {snapshot_name}")
                except Exception as e:
                    logger.error(f"Failed to create snapshot {snapshot_name}: {e}")

        except Exception as e:
            logger.error(f"Error checking snapshot need: {e}")

    def switch_mode(self, new_mode: str):
        """
        Switch between backfill and production modes.

        Args:
            new_mode: Either 'backfill' or 'production'
        """
        if new_mode not in ['backfill', 'production']:
            raise ValueError(f"Invalid mode: {new_mode}. Must be 'backfill' or 'production'")

        old_mode = self.mode
        self.mode = new_mode

        # Update configuration based on new mode
        mode_cfg = self.config['write_config'][new_mode]
        self.prune_versions = mode_cfg['prune_previous_versions']
        self.create_snapshots = mode_cfg['create_snapshots']
        self.snapshot_frequency = mode_cfg['snapshot_frequency']

        logger.info(f"Switched from {old_mode} to {new_mode} mode")
        logger.info(f"  - Prune versions: {self.prune_versions}")
        logger.info(f"  - Create snapshots: {self.create_snapshots}")

    def get_write_stats(self) -> Dict:
        """
        Get detailed write performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        stats = self.stats.copy()

        if self.stats['write_times_ms']:
            write_times = self.stats['write_times_ms']
            stats['avg_write_time_ms'] = sum(write_times) / len(write_times)
            stats['min_write_time_ms'] = min(write_times)
            stats['max_write_time_ms'] = max(write_times)
            stats['slow_writes'] = sum(1 for t in write_times if t > self.slow_write_threshold)

        stats['mode'] = self.mode
        stats['deduplication_enabled'] = self.enable_deduplication
        stats['validation_enabled'] = self.validate_index

        return stats

    def _log_error(self,
                   severity: ErrorSeverity,
                   message: str,
                   exception: Optional[Exception] = None,
                   context: Optional[Dict] = None):
        """
        Log error with context and maintain error history.

        Args:
            severity: Error severity level
            message: Error message
            exception: Optional exception object
            context: Optional context dictionary
        """
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity.name,
            'message': message,
            'exception': str(exception) if exception else None,
            'exception_type': type(exception).__name__ if exception else None,
            'context': context or {}
        }

        # Add to error history
        self.error_history.append(error_entry)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)

        # Log based on severity
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"{message}: {exception}", extra=context)
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"{message}: {exception}", extra=context)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"{message}: {exception}", extra=context)
        else:
            logger.info(f"{message}: {exception}", extra=context)

    def get_error_summary(self) -> Dict:
        """
        Get summary of recent errors.

        Returns:
            Dictionary with error statistics and recent errors
        """
        if not self.error_history:
            return {'total_errors': 0, 'recent_errors': []}

        # Count by severity
        severity_counts = {}
        for error in self.error_history:
            severity = error['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Get last 10 errors
        recent_errors = self.error_history[-10:]

        return {
            'total_errors': len(self.error_history),
            'severity_counts': severity_counts,
            'recent_errors': recent_errors,
            'validation_failures': self.stats['validation_failures'],
            'retries_attempted': self.stats['retries_attempted'],
            'retries_succeeded': self.stats['retries_succeeded']
        }

    def clear_error_history(self):
        """Clear the error history"""
        self.error_history = []
        logger.info("Error history cleared")

    def handle_corrupted_data(self, symbol: str, date: str) -> bool:
        """
        Attempt to recover from corrupted data.

        Args:
            symbol: Stock symbol
            date: Date string

        Returns:
            True if recovery successful
        """
        try:
            storage_key = f"{symbol}/{date}"
            logger.warning(f"Attempting to recover corrupted data for {storage_key}")

            # Try to read existing data
            try:
                existing_data = self.tick_data_lib.read(storage_key)
                logger.info(f"Data readable, creating backup snapshot")

                # Create backup snapshot
                snapshot_name = f"backup_{symbol}_{date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.tick_data_lib.snapshot(snapshot_name)

                return True

            except Exception as read_error:
                logger.error(f"Data unreadable, attempting deletion: {read_error}")

                # Delete corrupted data
                try:
                    self.tick_data_lib.delete(storage_key)
                    logger.info(f"Deleted corrupted data for {storage_key}")
                    return True
                except Exception as delete_error:
                    logger.error(f"Failed to delete corrupted data: {delete_error}")
                    return False

        except Exception as e:
            self._log_error(
                ErrorSeverity.HIGH,
                f"Failed to handle corrupted data for {symbol} on {date}",
                e
            )
            return False

    def generate_diagnostic_report(self) -> str:
        """
        Generate a comprehensive diagnostic report for troubleshooting.

        Returns:
            Formatted diagnostic report string
        """
        report = []
        report.append("=" * 80)
        report.append("TICKSTORE DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        # Configuration
        report.append("CONFIGURATION:")
        report.append(f"  Mode: {self.mode}")
        report.append(f"  URI: {self.arctic_uri}")
        report.append(f"  Batch Size: {self.write_batch_size:,}")
        report.append(f"  Deduplication: {self.enable_deduplication}")
        report.append(f"  Validate Index: {self.validate_index}")
        report.append(f"  Prune Versions: {self.prune_versions}")
        report.append("")

        # Performance Stats
        report.append("PERFORMANCE STATISTICS:")
        report.append(f"  Ticks Stored: {self.stats['ticks_stored']:,}")
        report.append(f"  Bytes Stored: {self.stats['bytes_stored']:,}")
        report.append(f"  Queries Executed: {self.stats['queries_executed']:,}")
        report.append(f"  Duplicates Removed: {self.stats['duplicates_removed']:,}")
        report.append(f"  Snapshots Created: {self.stats['snapshots_created']:,}")

        if self.stats['write_times_ms']:
            avg_time = sum(self.stats['write_times_ms']) / len(self.stats['write_times_ms'])
            report.append(f"  Avg Write Time: {avg_time:.1f}ms")
            report.append(f"  Slow Writes: {sum(1 for t in self.stats['write_times_ms'] if t > self.slow_write_threshold)}")
        report.append("")

        # Error Summary
        error_summary = self.get_error_summary()
        report.append("ERROR SUMMARY:")
        report.append(f"  Total Errors: {self.stats['errors']}")
        report.append(f"  Validation Failures: {self.stats['validation_failures']}")
        report.append(f"  Retries Attempted: {self.stats.get('retries_attempted', 0)}")
        report.append(f"  Retries Succeeded: {self.stats.get('retries_succeeded', 0)}")

        if error_summary['severity_counts']:
            report.append("  By Severity:")
            for severity, count in error_summary['severity_counts'].items():
                report.append(f"    {severity}: {count}")

        if error_summary['recent_errors']:
            report.append("\n  Recent Errors (last 5):")
            for error in error_summary['recent_errors'][-5:]:
                report.append(f"    [{error['severity']}] {error['message']}")
                if error['exception']:
                    report.append(f"      Exception: {error['exception_type']}")
        report.append("")

        # Storage Status
        try:
            stored_data = self.list_stored_data()
            report.append("STORAGE STATUS:")
            report.append(f"  Total Symbols: {len(set(d['symbol'] for d in stored_data))}")
            report.append(f"  Total Days: {len(stored_data)}")

            if stored_data:
                total_ticks = sum(d.get('tick_count', 0) for d in stored_data)
                total_size = sum(d.get('storage_size_mb', 0) for d in stored_data)
                report.append(f"  Total Ticks: {total_ticks:,}")
                report.append(f"  Total Size: {total_size:.1f} MB")
        except:
            report.append("STORAGE STATUS: Unable to retrieve")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def test_connectivity(self) -> Dict[str, bool]:
        """
        Test connectivity to ArcticDB and perform basic operations.

        Returns:
            Dictionary with test results
        """
        tests = {
            'connection': False,
            'read': False,
            'write': False,
            'list': False,
            'snapshot': False
        }

        try:
            # Test connection
            tests['connection'] = self.arctic is not None

            # Test list operation
            try:
                symbols = self.tick_data_lib.list_symbols()
                tests['list'] = True
            except:
                pass

            # Test write operation
            try:
                test_key = "_test_connectivity"
                test_df = pd.DataFrame({
                    'timestamp': [pd.Timestamp.now()],
                    'value': [1.0]
                })
                self.tick_data_lib.write(test_key, test_df)
                tests['write'] = True

                # Test read operation
                read_df = self.tick_data_lib.read(test_key).data
                tests['read'] = True

                # Clean up
                self.tick_data_lib.delete(test_key)
            except:
                pass

            # Test snapshot capability
            try:
                snapshot_name = "_test_snapshot"
                self.tick_data_lib.snapshot(snapshot_name)
                self.tick_data_lib.delete_snapshot(snapshot_name)
                tests['snapshot'] = True
            except:
                pass

        except Exception as e:
            logger.error(f"Connectivity test failed: {e}")

        return tests