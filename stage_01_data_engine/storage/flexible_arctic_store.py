"""
Flexible ArcticDB Storage for Exploratory Quantitative Research
Dynamic library creation and smart routing based on symbol categorization
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
from ..parsers.dtn_symbol_parser import DTNSymbolParser, SymbolInfo
from ..core.base_collector import BaseStore

logger = logging.getLogger(__name__)

class FlexibleArcticStore(BaseStore):
    """
    Flexible ArcticDB storage that adapts to ANY symbol without preconfiguration.

    Features:
    - Dynamic library creation based on symbol categories
    - Smart namespace routing using DTNSymbolParser
    - Lazy initialization (create libraries on first use)
    - Automatic organization of discovered symbols
    - Full metadata tracking for exploratory research
    """

    def __init__(self,
                 arctic_uri: str = "lmdb://./data/arctic_storage",
                 enable_compression: bool = True,
                 cache_size_mb: int = 1000,
                 config: Optional[Dict] = None):
        """
        Initialize flexible ArcticDB storage.

        Args:
            arctic_uri: ArcticDB connection string
            enable_compression: Enable LZ4 compression
            cache_size_mb: Memory cache size in MB
            config: Optional configuration overrides
        """
        super().__init__("FlexibleArcticStore", config)

        self.arctic_uri = arctic_uri
        self.enable_compression = enable_compression
        self.cache_size_mb = cache_size_mb

        # Initialize symbol parser for smart routing
        self.symbol_parser = DTNSymbolParser()

        # Track discovered libraries and namespaces (must be before _init_arctic)
        self.discovered_libraries = set()

        # Enhanced stats for exploratory research (must be before _init_arctic)
        self.stats.update({
            'symbols_stored': 0,
            'libraries_created': 0,
            'namespaces_discovered': 0,
            'auto_categorizations': 0
        })

        # Initialize ArcticDB connection
        self._init_arctic()

        # Initialize timezone handler
        self.timezone_handler = TimezoneHandler()
        self.namespace_cache = {}

        logger.info(f"FlexibleArcticStore initialized for exploratory research")

    def _init_arctic(self):
        """Initialize ArcticDB connection"""
        try:
            # Ensure data directory exists
            if self.arctic_uri.startswith("lmdb://"):
                data_path = Path(self.arctic_uri.replace("lmdb://", ""))
                data_path.mkdir(parents=True, exist_ok=True)

            # Connect to ArcticDB
            self.arctic = Arctic(self.arctic_uri)

            # Initialize core libraries (minimal set)
            self._init_core_libraries()

            logger.info("ArcticDB connection established for flexible storage")

        except Exception as e:
            logger.error(f"Failed to initialize ArcticDB: {e}")
            raise

    def _init_core_libraries(self):
        """Initialize only essential libraries - others created on demand"""
        core_libraries = {
            'metadata': 'System metadata and symbol registry',
            'discovery_log': 'Symbol discovery and categorization log'
        }

        for lib_name, description in core_libraries.items():
            try:
                self.arctic.get_library(lib_name)
                logger.debug(f"Core library '{lib_name}' already exists")
            except LibraryNotFound:
                self.arctic.create_library(lib_name)
                logger.info(f"Created core library: {lib_name}")
                self.stats['libraries_created'] += 1

            self.discovered_libraries.add(lib_name)

    def store_symbol_data(self,
                         symbol: str,
                         data: pd.DataFrame,
                         data_type: str = 'auto',
                         date: Optional[str] = None,
                         metadata: Optional[Dict] = None,
                         overwrite: bool = True) -> bool:
        """
        Store data for any symbol with automatic categorization and routing.

        Args:
            symbol: Any symbol (stocks, options, futures, DTN indicators)
            data: DataFrame to store
            data_type: 'auto' (detect), 'ticks', 'bars', 'quotes', etc.
            date: Date string (YYYY-MM-DD), auto-detected if None
            metadata: Additional metadata to store
            overwrite: Whether to overwrite existing data

        Returns:
            True if stored successfully
        """
        try:
            # Parse symbol for smart routing
            symbol_info = self.symbol_parser.parse_symbol(symbol)

            # Auto-detect date if not provided
            if date is None and 'timestamp' in data.columns and not data.empty:
                date = data['timestamp'].dt.date.iloc[0].strftime('%Y-%m-%d')
            elif date is None:
                date = datetime.now().strftime('%Y-%m-%d')

            # Generate storage key and library
            storage_key, library_name = self._get_storage_location(symbol_info, data_type, date)

            # Ensure library exists
            self._ensure_library_exists(library_name)

            # Prepare comprehensive metadata
            full_metadata = self._prepare_metadata(symbol_info, data_type, data, metadata)

            # Get library and store data
            library = self.arctic.get_library(library_name)

            # Normalize data for ArcticDB
            normalized_data = self._normalize_data(data)

            if overwrite or not self._symbol_exists(library, storage_key):
                library.write(storage_key, normalized_data, metadata=full_metadata)

                # Update stats
                self.update_stats('store', self._estimate_size_mb(data), success=True)
                self.stats['symbols_stored'] += 1
                self.stats['auto_categorizations'] += 1

                # Log discovery
                self._log_symbol_discovery(symbol_info, library_name, storage_key)

                logger.info(f"Stored {symbol} in {library_name}/{storage_key} ({len(data)} records)")
                return True
            else:
                logger.info(f"Symbol {symbol} already exists in storage (use overwrite=True to replace)")
                return True

        except Exception as e:
            logger.error(f"Failed to store {symbol}: {e}")
            self.update_stats('store', success=False)
            return False

    def load_symbol_data(self,
                        symbol: str,
                        date: Optional[str] = None,
                        data_type: str = 'auto',
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load data for any symbol with automatic routing.

        Args:
            symbol: Symbol to load
            date: Specific date (YYYY-MM-DD)
            data_type: Data type to load
            start_date: Start date for range queries
            end_date: End date for range queries

        Returns:
            DataFrame with loaded data or None if not found
        """
        try:
            # Parse symbol for routing
            symbol_info = self.symbol_parser.parse_symbol(symbol)

            if date:
                # Load specific date
                storage_key, library_name = self._get_storage_location(symbol_info, data_type, date)
                return self._load_from_library(library_name, storage_key)

            elif start_date and end_date:
                # Load date range
                return self._load_date_range(symbol_info, data_type, start_date, end_date)

            else:
                # Load most recent data
                return self._load_latest(symbol_info, data_type)

        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
            self.update_stats('load', success=False)
            return None

    def discover_stored_symbols(self,
                               category: Optional[str] = None,
                               subcategory: Optional[str] = None) -> List[Dict]:
        """
        Discover all symbols stored in the system.

        Args:
            category: Filter by symbol category
            subcategory: Filter by symbol subcategory

        Returns:
            List of symbol information dictionaries
        """
        try:
            discovered_symbols = []

            # Query discovery log
            try:
                discovery_lib = self.arctic.get_library('discovery_log')

                # List all discovery entries
                symbol_list = discovery_lib.list_symbols()

                for symbol_entry in symbol_list:
                    try:
                        entry_data = discovery_lib.read(symbol_entry)
                        symbol_info = entry_data.metadata

                        # Apply filters
                        if category and symbol_info.get('category') != category:
                            continue
                        if subcategory and symbol_info.get('subcategory') != subcategory:
                            continue

                        discovered_symbols.append(symbol_info)

                    except Exception as e:
                        logger.warning(f"Error reading discovery entry {symbol_entry}: {e}")
                        continue

            except LibraryNotFound:
                logger.warning("Discovery log not found - no symbols discovered yet")

            logger.info(f"Discovered {len(discovered_symbols)} stored symbols")
            return discovered_symbols

        except Exception as e:
            logger.error(f"Error discovering stored symbols: {e}")
            return []

    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get comprehensive information about a stored symbol.

        Args:
            symbol: Symbol to get info for

        Returns:
            Dictionary with symbol information and storage details
        """
        try:
            symbol_info = self.symbol_parser.parse_symbol(symbol)

            info = {
                'symbol': symbol,
                'parsed_info': {
                    'category': symbol_info.category,
                    'subcategory': symbol_info.subcategory,
                    'exchange': symbol_info.exchange,
                    'underlying': symbol_info.underlying,
                    'storage_namespace': symbol_info.storage_namespace
                },
                'storage_locations': [],
                'available_data_types': [],
                'date_range': {'earliest': None, 'latest': None},
                'total_records': 0
            }

            # Find storage locations
            namespace_pattern = symbol_info.storage_namespace.replace('/', '_')

            for lib_name in self.discovered_libraries:
                if namespace_pattern in lib_name:
                    try:
                        library = self.arctic.get_library(lib_name)
                        symbols = library.list_symbols()

                        for stored_symbol in symbols:
                            if symbol in stored_symbol:
                                info['storage_locations'].append(f"{lib_name}/{stored_symbol}")

                                # Get data info
                                try:
                                    data_info = library.get_info(stored_symbol)
                                    info['total_records'] += data_info.get('rows', 0)
                                except:
                                    pass

                    except LibraryNotFound:
                        continue

            return info

        except Exception as e:
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def _get_storage_location(self,
                             symbol_info: SymbolInfo,
                             data_type: str,
                             date: str) -> Tuple[str, str]:
        """Generate storage key and library name based on symbol info"""

        # Generate library name from namespace
        library_name = symbol_info.storage_namespace.replace('/', '_')

        # Generate storage key
        if data_type == 'auto':
            # Use symbol's natural data type
            if symbol_info.category == 'dtn_calculated':
                data_type = 'indicators'
            elif symbol_info.category == 'options':
                data_type = 'quotes'
            else:
                data_type = 'ticks'

        storage_key = f"{symbol_info.symbol}_{data_type}_{date}"

        return storage_key, library_name

    def _ensure_library_exists(self, library_name: str):
        """Ensure library exists, create if needed"""
        if library_name not in self.discovered_libraries:
            try:
                self.arctic.get_library(library_name)
                logger.debug(f"Library '{library_name}' already exists")
            except LibraryNotFound:
                self.arctic.create_library(library_name)
                logger.info(f"Created new library: {library_name}")
                self.stats['libraries_created'] += 1

            self.discovered_libraries.add(library_name)

    def _prepare_metadata(self,
                         symbol_info: SymbolInfo,
                         data_type: str,
                         data: pd.DataFrame,
                         user_metadata: Optional[Dict]) -> Dict:
        """Prepare comprehensive metadata for storage"""
        metadata = {
            'symbol': symbol_info.symbol,
            'category': symbol_info.category,
            'subcategory': symbol_info.subcategory,
            'data_type': data_type,
            'storage_namespace': symbol_info.storage_namespace,
            'stored_at': datetime.now().isoformat(),
            'records_count': len(data),
            'columns': list(data.columns),
            'exploratory_mode': True
        }

        # Add symbol-specific metadata
        if symbol_info.metadata:
            metadata.update(symbol_info.metadata)

        # Add optional fields
        for field in ['exchange', 'underlying', 'expiration', 'strike_price', 'option_type']:
            value = getattr(symbol_info, field, None)
            if value is not None:
                metadata[field] = value

        # Add data statistics
        if not data.empty:
            if 'timestamp' in data.columns:
                metadata['date_range'] = {
                    'start': data['timestamp'].min().isoformat(),
                    'end': data['timestamp'].max().isoformat()
                }

            if 'price' in data.columns:
                metadata['price_stats'] = {
                    'min': float(data['price'].min()),
                    'max': float(data['price'].max()),
                    'mean': float(data['price'].mean())
                }

        # Add user metadata
        if user_metadata:
            metadata['user_metadata'] = user_metadata

        return metadata

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data for ArcticDB storage"""
        normalized = data.copy()

        # Ensure timestamp is datetime
        if 'timestamp' in normalized.columns:
            normalized['timestamp'] = pd.to_datetime(normalized['timestamp'])

        # Handle any infinity or NaN values
        normalized = normalized.replace([np.inf, -np.inf], np.nan)

        return normalized

    def _symbol_exists(self, library, storage_key: str) -> bool:
        """Check if symbol already exists in library"""
        try:
            symbols = library.list_symbols()
            return storage_key in symbols
        except:
            return False

    def _estimate_size_mb(self, data: pd.DataFrame) -> float:
        """Estimate data size in MB"""
        try:
            return data.memory_usage(deep=True).sum() / (1024 * 1024)
        except:
            return 0.1

    def _log_symbol_discovery(self, symbol_info: SymbolInfo, library_name: str, storage_key: str):
        """Log symbol discovery for future reference"""
        try:
            discovery_lib = self.arctic.get_library('discovery_log')

            discovery_entry = pd.DataFrame([{
                'symbol': symbol_info.symbol,
                'discovered_at': datetime.now(),
                'category': symbol_info.category,
                'subcategory': symbol_info.subcategory,
                'library_name': library_name,
                'storage_key': storage_key
            }])

            # Store with symbol as key
            discovery_key = f"discovery_{symbol_info.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            metadata = {
                'symbol': symbol_info.symbol,
                'category': symbol_info.category,
                'subcategory': symbol_info.subcategory,
                'storage_namespace': symbol_info.storage_namespace,
                'discovery_timestamp': datetime.now().isoformat()
            }

            discovery_lib.write(discovery_key, discovery_entry, metadata=metadata)
            self.stats['namespaces_discovered'] += 1

        except Exception as e:
            logger.warning(f"Failed to log symbol discovery: {e}")

    def _load_from_library(self, library_name: str, storage_key: str) -> Optional[pd.DataFrame]:
        """Load data from specific library and key"""
        try:
            library = self.arctic.get_library(library_name)
            result = library.read(storage_key)

            self.update_stats('load', success=True)
            logger.debug(f"Loaded {storage_key} from {library_name}")
            return result.data

        except (LibraryNotFound, Exception) as e:
            logger.debug(f"Could not load {library_name}/{storage_key}: {e}")
            return None

    def _load_date_range(self, symbol_info: SymbolInfo, data_type: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Load data for a date range"""
        # Implementation for date range loading
        # This would iterate through dates and combine results
        logger.info(f"Date range loading not yet implemented for {symbol_info.symbol}")
        return None

    def _load_latest(self, symbol_info: SymbolInfo, data_type: str) -> Optional[pd.DataFrame]:
        """Load most recent data for symbol"""
        # Implementation for loading latest data
        # This would find the most recent date and load that
        logger.info(f"Latest data loading not yet implemented for {symbol_info.symbol}")
        return None

    # Override BaseStore abstract methods
    def store(self, key: str, data: pd.DataFrame, metadata: Optional[Dict] = None) -> bool:
        """Store data with given key (BaseStore compatibility)"""
        # Extract symbol from key
        symbol = key.split('_')[0] if '_' in key else key
        return self.store_symbol_data(symbol, data, metadata=metadata)

    def load(self, key: str, **kwargs) -> Optional[pd.DataFrame]:
        """Load data by key (BaseStore compatibility)"""
        symbol = key.split('_')[0] if '_' in key else key
        return self.load_symbol_data(symbol, **kwargs)

    def exists(self, key: str) -> bool:
        """Check if data exists for key"""
        symbol = key.split('_')[0] if '_' in key else key
        info = self.get_symbol_info(symbol)
        return len(info.get('storage_locations', [])) > 0

    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List all storage keys"""
        discovered = self.discover_stored_symbols()
        keys = [item['symbol'] for item in discovered]

        if pattern:
            keys = [k for k in keys if pattern in k]

        return keys

    def get_storage_stats(self) -> Dict:
        """Get comprehensive storage statistics for exploratory research"""
        base_stats = self.get_stats()

        enhanced_stats = {
            **base_stats,
            'discovered_libraries': len(self.discovered_libraries),
            'library_names': list(self.discovered_libraries),
            'total_symbols_by_category': {},
            'storage_efficiency': {
                'avg_records_per_symbol': self.stats['total_records'] / max(self.stats['symbols_stored'], 1),
                'libraries_per_category': len(self.discovered_libraries) / max(1, len(set(lib.split('_')[0] for lib in self.discovered_libraries)))
            }
        }

        # Get category breakdown
        discovered = self.discover_stored_symbols()
        category_counts = {}
        for symbol_info in discovered:
            category = symbol_info.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1

        enhanced_stats['total_symbols_by_category'] = category_counts

        return enhanced_stats