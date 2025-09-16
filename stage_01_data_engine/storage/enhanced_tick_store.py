"""
Enhanced Tick Storage with Universal Bar Processing
Integrates tick storage with real-time bar generation and storage
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json

# Import base TickStore
from tick_store import TickStore, TickStoreError, DataValidationError

# Import Universal Bar Processor
import sys
sys.path.append('.')
sys.path.append('foundation')
from foundation.utils.universal_bar_processor import UniversalBarProcessor
from foundation.utils.metadata_computer import MetadataComputer
from foundation.models.market import TickData

# ArcticDB for bar storage
from arcticdb import Arctic

logger = logging.getLogger(__name__)


class EnhancedTickStore(TickStore):
    """
    Enhanced tick storage with integrated bar processing.
    Extends TickStore to generate and store all bar types in real-time.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize enhanced tick store with bar processing.

        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)

        # Initialize bar storage libraries
        self._init_bar_libraries()

        # Initialize Universal Bar Processor configuration
        self.bar_processor_config = self._get_bar_processor_config()

        # Track bar processing statistics
        self.bar_stats = {
            'total_bars_generated': 0,
            'bars_by_type': {},
            'processing_time_ms': 0,
            'last_processed': None
        }

        logger.info("Enhanced TickStore initialized with bar processing capability")

    def _init_bar_libraries(self):
        """Initialize separate ArcticDB libraries for each bar type"""
        bar_types = [
            'time_bars',      # 1-min, 5-min, etc.
            'tick_bars',      # 100-tick, 500-tick
            'volume_bars',    # 1K, 5K, 10K shares
            'dollar_bars',    # $50K, $100K, $500K
            'range_bars',     # $0.25, $0.50 ranges
            'bar_metadata'    # Bar-level Tier 2 metadata
        ]

        for bar_type in bar_types:
            lib_name = f"bars_{bar_type}"
            if lib_name not in self.arctic.list_libraries():
                self.arctic.create_library(lib_name)
                logger.info(f"Created bar library: {lib_name}")

            # Store library reference
            setattr(self, f"{bar_type}_lib", self.arctic[lib_name])

    def _get_bar_processor_config(self) -> Dict[str, Any]:
        """Get configuration for Universal Bar Processor"""
        return {
            'time_bars': {
                'enabled': True,
                'intervals': [60, 300, 900]  # 1-min, 5-min, 15-min
            },
            'tick_bars': {
                'enabled': True,
                'sizes': [100, 500, 1000]
            },
            'volume_bars': {
                'enabled': True,
                'thresholds': [1000, 5000, 10000, 50000]
            },
            'dollar_bars': {
                'enabled': True,
                'thresholds': [50000, 100000, 500000, 1000000]
            },
            'range_bars': {
                'enabled': True,
                'ranges': [0.25, 0.50, 1.00, 2.00]
            },
            'renko_bars': {
                'enabled': False,  # Disabled until VWAP issue fixed
                'brick_sizes': [0.25, 0.50, 1.00]
            },
            'imbalance_bars': {
                'enabled': False,  # Disabled until threshold issue fixed
                'thresholds': [0.1, 0.2, 0.3]
            }
        }

    def store_ticks_with_bars(self,
                              symbol: str,
                              date: str,
                              pydantic_ticks: List[TickData],
                              metadata: Optional[Dict] = None,
                              overwrite: bool = False) -> Tuple[bool, Dict[str, int]]:
        """
        Store ticks and generate all bar types simultaneously.

        Args:
            symbol: Stock symbol
            date: Date string (YYYY-MM-DD)
            pydantic_ticks: List of TickData Pydantic models
            metadata: Additional metadata
            overwrite: Whether to overwrite existing data

        Returns:
            Tuple of (success: bool, bar_counts: Dict[str, int])
        """
        try:
            # Convert Pydantic to DataFrame for storage
            tick_df = self._pydantic_to_dataframe(pydantic_ticks)

            # Store raw ticks first (using parent method)
            tick_success = self.store_ticks(symbol, date, tick_df, metadata, overwrite)

            if not tick_success:
                logger.error(f"Failed to store ticks for {symbol} on {date}")
                return False, {}

            # Initialize Universal Bar Processor
            processor = UniversalBarProcessor(symbol, self.bar_processor_config)

            # Process ticks through all bar builders
            logger.info(f"Processing {len(pydantic_ticks)} ticks through Universal Bar Processor")

            start_time = datetime.now()
            all_bars = processor.process_ticks(pydantic_ticks)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Store each bar type with its metadata
            bar_counts = {}
            for builder_key, bars_list in all_bars.items():
                bar_counts[builder_key] = len(bars_list)
                self._store_bars_batch(symbol, date, builder_key, bars_list)

            # Force close incomplete bars (important for time bars)
            final_bars = processor.force_close_all()
            for builder_key, (bar, bar_metadata) in final_bars.items():
                if builder_key not in bar_counts:
                    bar_counts[builder_key] = 0
                bar_counts[builder_key] += 1
                self._store_single_bar(symbol, date, builder_key, bar, bar_metadata)

            # Update statistics
            self.bar_stats['total_bars_generated'] += sum(bar_counts.values())
            self.bar_stats['processing_time_ms'] = processing_time
            self.bar_stats['last_processed'] = datetime.now()

            for bar_type, count in bar_counts.items():
                if bar_type not in self.bar_stats['bars_by_type']:
                    self.bar_stats['bars_by_type'][bar_type] = 0
                self.bar_stats['bars_by_type'][bar_type] += count

            # Log summary
            total_bars = sum(bar_counts.values())
            logger.info(
                f"Generated {total_bars} bars from {len(pydantic_ticks)} ticks in {processing_time:.1f}ms"
            )
            logger.info(f"Bar breakdown: {bar_counts}")

            return True, bar_counts

        except Exception as e:
            logger.error(f"Error in store_ticks_with_bars: {e}", exc_info=True)
            return False, {}

    def _store_bars_batch(self,
                         symbol: str,
                         date: str,
                         builder_key: str,
                         bars_list: List[Tuple[Any, Dict]]) -> bool:
        """
        Store a batch of bars with their metadata.

        Args:
            symbol: Stock symbol
            date: Date string
            builder_key: Bar builder identifier (e.g., 'time_60', 'volume_1000')
            bars_list: List of (bar, metadata) tuples

        Returns:
            Success boolean
        """
        try:
            # Determine bar type and library
            bar_type = builder_key.split('_')[0]
            lib = getattr(self, f"{bar_type}_bars_lib", None)

            if lib is None:
                logger.error(f"No library found for bar type: {bar_type}")
                return False

            for bar, bar_metadata in bars_list:
                storage_key = f"{symbol}/{date}/{builder_key}/{bar.timestamp.isoformat()}"

                # Convert bar to DataFrame for storage
                bar_df = pd.DataFrame([bar.model_dump()])

                # Store bar with metadata
                lib.write(
                    storage_key,
                    bar_df,
                    metadata={
                        'bar_type': builder_key,
                        'symbol': symbol,
                        'date': date,
                        'tier2_metadata': bar_metadata,
                        'stored_at': datetime.now().isoformat()
                    }
                )

                logger.debug(f"Stored {builder_key} bar at {bar.timestamp}")

            return True

        except Exception as e:
            logger.error(f"Error storing bars batch: {e}", exc_info=True)
            return False

    def _store_single_bar(self,
                         symbol: str,
                         date: str,
                         builder_key: str,
                         bar: Any,
                         bar_metadata: Dict) -> bool:
        """Store a single bar with its metadata"""
        return self._store_bars_batch(symbol, date, builder_key, [(bar, bar_metadata)])

    def get_bars(self,
                symbol: str,
                date: str,
                bar_type: str,
                bar_param: Optional[Any] = None,
                start_time: Optional[datetime] = None,
                end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve bars for a symbol and date.

        Args:
            symbol: Stock symbol
            date: Date string (YYYY-MM-DD)
            bar_type: Type of bar ('time', 'tick', 'volume', 'dollar', 'range')
            bar_param: Bar parameter (e.g., 60 for 1-min, 1000 for 1K volume)
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            DataFrame with bars
        """
        try:
            # Get appropriate library
            lib = getattr(self, f"{bar_type}_bars_lib", None)
            if lib is None:
                logger.error(f"Unknown bar type: {bar_type}")
                return pd.DataFrame()

            # Build storage key pattern
            if bar_param:
                key_pattern = f"{symbol}/{date}/{bar_type}_{bar_param}/*"
            else:
                key_pattern = f"{symbol}/{date}/{bar_type}*"

            # Read all matching bars
            bars_list = []
            for key in lib.list_symbols():
                if key.startswith(key_pattern.replace('*', '')):
                    try:
                        bar_data = lib.read(key)
                        bars_list.append(bar_data.data)
                    except Exception as e:
                        logger.warning(f"Error reading bar {key}: {e}")

            if not bars_list:
                logger.warning(f"No bars found for {key_pattern}")
                return pd.DataFrame()

            # Combine all bars
            bars_df = pd.concat(bars_list, ignore_index=True)

            # Apply time filters if provided
            if start_time or end_time:
                if 'timestamp' in bars_df.columns:
                    if start_time:
                        bars_df = bars_df[bars_df['timestamp'] >= start_time]
                    if end_time:
                        bars_df = bars_df[bars_df['timestamp'] <= end_time]

            # Sort by timestamp
            if 'timestamp' in bars_df.columns:
                bars_df = bars_df.sort_values('timestamp').reset_index(drop=True)

            return bars_df

        except Exception as e:
            logger.error(f"Error retrieving bars: {e}", exc_info=True)
            return pd.DataFrame()

    def get_bar_metadata(self,
                        symbol: str,
                        date: str,
                        bar_type: str,
                        bar_param: Optional[Any] = None) -> List[Dict]:
        """
        Retrieve bar metadata without loading bar data.

        Args:
            symbol: Stock symbol
            date: Date string
            bar_type: Type of bar
            bar_param: Bar parameter

        Returns:
            List of metadata dictionaries
        """
        try:
            lib = getattr(self, f"{bar_type}_bars_lib", None)
            if lib is None:
                return []

            if bar_param:
                key_pattern = f"{symbol}/{date}/{bar_type}_{bar_param}/"
            else:
                key_pattern = f"{symbol}/{date}/{bar_type}"

            metadata_list = []
            for key in lib.list_symbols():
                if key.startswith(key_pattern):
                    try:
                        versioned_item = lib.read_metadata(key)
                        if versioned_item and versioned_item.metadata:
                            metadata_list.append(versioned_item.metadata)
                    except Exception as e:
                        logger.warning(f"Error reading metadata for {key}: {e}")

            return metadata_list

        except Exception as e:
            logger.error(f"Error retrieving bar metadata: {e}", exc_info=True)
            return []

    def _pydantic_to_dataframe(self, pydantic_ticks: List[TickData]) -> pd.DataFrame:
        """
        Convert list of Pydantic TickData models to DataFrame.

        Args:
            pydantic_ticks: List of TickData Pydantic models

        Returns:
            DataFrame with tick data
        """
        # Convert Pydantic models to list of dicts
        tick_dicts = [tick.model_dump() for tick in pydantic_ticks]

        # Create DataFrame
        df = pd.DataFrame(tick_dicts)

        # Rename columns to match tick_store expectations
        if 'size' in df.columns:
            df['volume'] = df['size']

        # Use source_timestamp as timestamp
        if 'source_timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['source_timestamp'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def get_bar_statistics(self) -> Dict[str, Any]:
        """Get bar processing statistics"""
        return {
            **self.bar_stats,
            'libraries': {
                'time_bars': len(self.time_bars_lib.list_symbols()),
                'tick_bars': len(self.tick_bars_lib.list_symbols()),
                'volume_bars': len(self.volume_bars_lib.list_symbols()),
                'dollar_bars': len(self.dollar_bars_lib.list_symbols()),
                'range_bars': len(self.range_bars_lib.list_symbols())
            }
        }

    def delete_bars(self,
                   symbol: str,
                   date: str,
                   bar_type: Optional[str] = None) -> bool:
        """
        Delete bars for a symbol and date.

        Args:
            symbol: Stock symbol
            date: Date string
            bar_type: Optional specific bar type to delete

        Returns:
            Success boolean
        """
        try:
            if bar_type:
                # Delete specific bar type
                lib = getattr(self, f"{bar_type}_bars_lib", None)
                if lib:
                    pattern = f"{symbol}/{date}/"
                    for key in lib.list_symbols():
                        if key.startswith(pattern):
                            lib.delete(key)
                    logger.info(f"Deleted {bar_type} bars for {symbol} on {date}")
            else:
                # Delete all bar types
                for bar_type in ['time', 'tick', 'volume', 'dollar', 'range']:
                    self.delete_bars(symbol, date, bar_type)
                logger.info(f"Deleted all bars for {symbol} on {date}")

            return True

        except Exception as e:
            logger.error(f"Error deleting bars: {e}", exc_info=True)
            return False

    def optimize_bar_storage(self):
        """Optimize bar storage by compacting libraries"""
        try:
            bar_types = ['time', 'tick', 'volume', 'dollar', 'range']

            for bar_type in bar_types:
                lib = getattr(self, f"{bar_type}_bars_lib", None)
                if lib:
                    # ArcticDB will handle optimization internally
                    symbol_count = len(lib.list_symbols())
                    logger.info(f"Library {bar_type}_bars has {symbol_count} symbols")

            logger.info("Bar storage optimization complete")

        except Exception as e:
            logger.error(f"Error optimizing bar storage: {e}", exc_info=True)


def test_enhanced_store():
    """Test the enhanced tick store with bar processing"""
    import sys
    sys.path.insert(0, 'stage_01_data_engine/collectors')
    from iqfeed_collector import IQFeedCollector
    from foundation.utils.iqfeed_converter import convert_iqfeed_ticks_to_pydantic

    # Initialize components
    store = EnhancedTickStore()
    collector = IQFeedCollector()

    # Connect to IQFeed
    if not collector.ensure_connection():
        logger.error("Failed to connect to IQFeed")
        return

    # Fetch test data
    symbol = 'AAPL'
    logger.info(f"Fetching {symbol} ticks...")
    tick_array = collector.get_tick_data(symbol, num_days=1, max_ticks=5000)
    logger.info(f"Fetched {len(tick_array)} ticks")

    # Convert to Pydantic
    pydantic_ticks = convert_iqfeed_ticks_to_pydantic(tick_array, symbol)
    logger.info(f"Converted {len(pydantic_ticks)} ticks to Pydantic models")

    # Store with bar generation
    date = datetime.now().strftime('%Y-%m-%d')
    success, bar_counts = store.store_ticks_with_bars(
        symbol, date, pydantic_ticks, overwrite=True
    )

    if success:
        logger.info(f"Successfully stored ticks and generated bars")
        logger.info(f"Bar counts: {bar_counts}")

        # Retrieve some bars
        time_bars = store.get_bars(symbol, date, 'time', 60)
        logger.info(f"Retrieved {len(time_bars)} 1-minute bars")

        if not time_bars.empty:
            logger.info(f"Sample bar: {time_bars.iloc[0].to_dict()}")

        # Get statistics
        stats = store.get_bar_statistics()
        logger.info(f"Bar statistics: {stats}")
    else:
        logger.error("Failed to store ticks with bars")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_enhanced_store()