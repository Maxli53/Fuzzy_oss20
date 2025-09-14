"""
Base Collector Architecture for Stage 1 Data Engine
Provides abstract base classes for all data collection components
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.

    Ensures consistent interface across all data collection modules:
    - Tick collectors
    - Bar collectors
    - Market internals collectors
    - Options collectors
    - DTN indicator collectors
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.stats = {
            'collections': 0,
            'errors': 0,
            'last_collection': None,
            'total_records': 0
        }
        logger.info(f"Initialized {name} collector")

    @abstractmethod
    def collect(self, symbols: Union[str, List[str]], **kwargs) -> Optional[pd.DataFrame]:
        """
        Collect data for given symbols.

        Args:
            symbols: Symbol or list of symbols to collect
            **kwargs: Collector-specific parameters

        Returns:
            DataFrame with collected data or None if failed
        """
        pass

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        """
        Validate collected data quality.

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_storage_key(self, symbol: str, date: str, **kwargs) -> str:
        """
        Generate storage key for data.

        Args:
            symbol: Stock symbol
            date: Date string (YYYY-MM-DD)
            **kwargs: Additional key parameters

        Returns:
            Storage key string
        """
        pass

    def update_stats(self, records_collected: int, success: bool = True):
        """Update collector statistics"""
        self.stats['collections'] += 1
        self.stats['last_collection'] = datetime.now()

        if success:
            self.stats['total_records'] += records_collected
        else:
            self.stats['errors'] += 1

    def get_stats(self) -> Dict:
        """Get collector statistics"""
        return {
            **self.stats,
            'name': self.name,
            'success_rate': (
                (self.stats['collections'] - self.stats['errors']) /
                max(1, self.stats['collections']) * 100
            )
        }


class BaseProcessor(ABC):
    """
    Abstract base class for all data processors.

    Handles transformation of raw data into processed formats:
    - Bar building
    - Microstructure analysis
    - Options processing
    - Market regime detection
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.stats = {
            'processes': 0,
            'errors': 0,
            'last_process': None
        }
        logger.info(f"Initialized {name} processor")

    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> Optional[pd.DataFrame]:
        """
        Process raw data into transformed format.

        Args:
            data: Raw data to process
            **kwargs: Processor-specific parameters

        Returns:
            Processed DataFrame or None if failed
        """
        pass

    @abstractmethod
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data is suitable for processing.

        Args:
            data: Input data to validate

        Returns:
            True if data is valid for processing
        """
        pass

    def update_stats(self, success: bool = True):
        """Update processor statistics"""
        self.stats['processes'] += 1
        self.stats['last_process'] = datetime.now()

        if not success:
            self.stats['errors'] += 1

    def get_stats(self) -> Dict:
        """Get processor statistics"""
        return {
            **self.stats,
            'name': self.name,
            'success_rate': (
                (self.stats['processes'] - self.stats['errors']) /
                max(1, self.stats['processes']) * 100
            )
        }


class BaseStore(ABC):
    """
    Abstract base class for all data storage components.

    Handles persistence of different data types:
    - Tick storage
    - Bar storage
    - Indicator storage
    - Options storage
    """

    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self.stats = {
            'stores': 0,
            'loads': 0,
            'errors': 0,
            'total_size_mb': 0
        }
        logger.info(f"Initialized {name} store")

    @abstractmethod
    def store(self, key: str, data: pd.DataFrame, metadata: Optional[Dict] = None) -> bool:
        """
        Store data with given key.

        Args:
            key: Storage key
            data: Data to store
            metadata: Optional metadata

        Returns:
            True if stored successfully
        """
        pass

    @abstractmethod
    def load(self, key: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Load data by key.

        Args:
            key: Storage key
            **kwargs: Load-specific parameters

        Returns:
            Loaded DataFrame or None if not found
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if data exists for key.

        Args:
            key: Storage key to check

        Returns:
            True if data exists
        """
        pass

    @abstractmethod
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all storage keys, optionally filtered by pattern.

        Args:
            pattern: Optional key pattern to filter

        Returns:
            List of matching keys
        """
        pass

    def update_stats(self, operation: str, size_mb: float = 0, success: bool = True):
        """Update storage statistics"""
        if operation == 'store':
            self.stats['stores'] += 1
            if success:
                self.stats['total_size_mb'] += size_mb
        elif operation == 'load':
            self.stats['loads'] += 1

        if not success:
            self.stats['errors'] += 1

    def get_stats(self) -> Dict:
        """Get storage statistics"""
        return {
            **self.stats,
            'name': self.name
        }


class DataFrequency:
    """Standard data frequency classifications"""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAILY = "daily"
    STATIC = "static"


class AssetClass:
    """Standard asset class classifications"""
    EQUITY = "equity"
    OPTION = "option"
    INDEX = "index"
    FUTURE = "future"
    INTERNAL = "internal"  # Market internals
    INDICATOR = "indicator"  # DTN indicators


class DataType:
    """Standard data type classifications"""
    RAW = "raw"
    PROCESSED = "processed"
    AGGREGATED = "aggregated"
    DERIVED = "derived"


class StorageNamespace:
    """Centralized namespace management for consistent storage keys with exploratory support"""

    @staticmethod
    def tick_key(symbol: str, date: str) -> str:
        """Generate tick data storage key"""
        return f"ticks/{symbol}/{date}"

    @staticmethod
    def bar_key(symbol: str, frequency: str, date: str) -> str:
        """Generate bar data storage key"""
        return f"bars_{frequency}/{symbol}/{date}"

    @staticmethod
    def indicator_key(indicator: str, date: str) -> str:
        """Generate indicator storage key"""
        return f"indicators/{indicator}/{date}"

    @staticmethod
    def options_key(symbol: str, data_type: str, date: str) -> str:
        """Generate options data storage key"""
        return f"options_{data_type}/{symbol}/{date}"

    @staticmethod
    def metadata_key(symbol: str, data_type: str) -> str:
        """Generate metadata storage key"""
        return f"metadata/{data_type}/{symbol}"

    # ===========================================
    # DYNAMIC ROUTING FOR EXPLORATORY RESEARCH
    # ===========================================

    @staticmethod
    def dynamic_key(symbol_info, data_type: str, date: str) -> str:
        """
        Generate dynamic storage key based on parsed symbol information.
        Perfect for exploratory research with ANY symbol.

        Args:
            symbol_info: SymbolInfo object from DTNSymbolParser
            data_type: Type of data being stored
            date: Date string (YYYY-MM-DD)

        Returns:
            Dynamic storage key based on symbol category
        """
        category = symbol_info.category
        subcategory = symbol_info.subcategory
        symbol = symbol_info.symbol

        # Route based on category
        if category == 'equity':
            if subcategory == 'common_stock':
                return f"equity/stocks/{symbol}/{data_type}/{date}"
            elif subcategory == 'etf':
                return f"equity/etfs/{symbol}/{data_type}/{date}"
            elif subcategory == 'index':
                return f"equity/indices/{symbol}/{data_type}/{date}"
            else:
                return f"equity/{subcategory}/{symbol}/{data_type}/{date}"

        elif category == 'dtn_calculated':
            # Use subcategory for DTN organization
            return f"dtn/{subcategory}/{symbol}/{data_type}/{date}"

        elif category == 'options':
            underlying = symbol_info.underlying or 'unknown'
            return f"options/{underlying}/{symbol}/{data_type}/{date}"

        elif category == 'futures':
            underlying = symbol_info.underlying or 'unknown'
            return f"futures/{underlying}/{symbol}/{data_type}/{date}"

        elif category == 'forex':
            base_currency = symbol_info.metadata.get('base_currency', symbol[:3]) if symbol_info.metadata else symbol[:3]
            return f"forex/{base_currency}/{symbol}/{data_type}/{date}"

        else:
            # Unknown category
            return f"unknown/{category}/{symbol}/{data_type}/{date}"

    @staticmethod
    def dynamic_library_name(symbol_info) -> str:
        """
        Generate dynamic library name for ArcticDB based on symbol category.

        Args:
            symbol_info: SymbolInfo object from DTNSymbolParser

        Returns:
            Library name for ArcticDB storage
        """
        category = symbol_info.category
        subcategory = symbol_info.subcategory

        # Create hierarchical library names
        if category == 'equity':
            return f"iqfeed_equity_{subcategory}"

        elif category == 'dtn_calculated':
            # Group DTN indicators by category from PDF
            category_group = symbol_info.metadata.get('category_group', 'general') if symbol_info.metadata else 'general'
            return f"iqfeed_dtn_{category_group}"

        elif category == 'options':
            underlying = symbol_info.underlying or 'general'
            return f"iqfeed_options_{underlying.lower()}"

        elif category == 'futures':
            underlying = symbol_info.underlying or 'general'
            return f"iqfeed_futures_{underlying.lower()}"

        elif category == 'forex':
            return f"iqfeed_forex"

        else:
            return f"iqfeed_unknown_{category}"

    @staticmethod
    def iqfeed_key(data_type: str, symbol: str, date: str) -> str:
        """Generate IQFeed-specific storage key (legacy compatibility)"""
        return f"iqfeed/{data_type}/{symbol}/{date}"

    @staticmethod
    def polygon_key(data_type: str, symbol: str, date: str) -> str:
        """Generate Polygon-specific storage key"""
        return f"polygon/{data_type}/{symbol}/{date}"

    @staticmethod
    def get_namespace_hierarchy(symbol_info) -> Dict[str, str]:
        """
        Get complete namespace hierarchy for a symbol.

        Args:
            symbol_info: SymbolInfo object

        Returns:
            Dictionary with namespace components
        """
        return {
            'data_source': 'iqfeed',
            'category': symbol_info.category,
            'subcategory': symbol_info.subcategory,
            'symbol': symbol_info.symbol,
            'library_name': StorageNamespace.dynamic_library_name(symbol_info),
            'base_namespace': symbol_info.storage_namespace,
            'exchange': symbol_info.exchange,
            'underlying': symbol_info.underlying
        }

    @staticmethod
    def categorize_storage_keys(keys: List[str]) -> Dict[str, List[str]]:
        """
        Categorize a list of storage keys by their namespace.

        Args:
            keys: List of storage keys

        Returns:
            Dictionary mapping categories to their keys
        """
        categorized = {
            'equity': [],
            'dtn_calculated': [],
            'options': [],
            'futures': [],
            'forex': [],
            'unknown': []
        }

        for key in keys:
            parts = key.split('/')
            if len(parts) > 0:
                category = parts[0]
                if category in categorized:
                    categorized[category].append(key)
                else:
                    categorized['unknown'].append(key)

        return categorized

    @staticmethod
    def suggest_related_keys(symbol_info, existing_keys: List[str]) -> List[str]:
        """
        Suggest related storage keys based on symbol information.

        Args:
            symbol_info: SymbolInfo object
            existing_keys: List of existing keys in storage

        Returns:
            List of suggested related keys
        """
        suggestions = []

        # Find keys with same underlying (for options/futures)
        if symbol_info.underlying:
            underlying = symbol_info.underlying
            for key in existing_keys:
                if underlying in key and key not in suggestions:
                    suggestions.append(key)

        # Find keys in same category
        category = symbol_info.category
        for key in existing_keys:
            if key.startswith(category) and key not in suggestions:
                suggestions.append(key)

        # Find keys with same exchange
        if symbol_info.exchange:
            exchange = symbol_info.exchange.lower()
            for key in existing_keys:
                if exchange in key.lower() and key not in suggestions:
                    suggestions.append(key)

        return suggestions[:10]  # Return top 10 suggestions

    @staticmethod
    def validate_storage_key(key: str) -> bool:
        """
        Validate storage key format.

        Args:
            key: Storage key to validate

        Returns:
            True if key format is valid
        """
        try:
            parts = key.split('/')
            return len(parts) >= 3 and all(part for part in parts)
        except:
            return False

    @staticmethod
    def normalize_symbol_for_storage(symbol: str) -> str:
        """
        Normalize symbol for storage key usage.

        Args:
            symbol: Raw symbol

        Returns:
            Normalized symbol safe for storage keys
        """
        # Replace problematic characters for storage systems
        normalized = symbol.replace('/', '_').replace('\\', '_').replace(':', '_').replace('?', '_')
        normalized = normalized.replace('*', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        normalized = normalized.replace('"', '_').replace(' ', '_')

        return normalized.upper()