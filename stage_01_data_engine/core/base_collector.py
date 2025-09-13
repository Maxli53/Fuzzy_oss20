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
    """Centralized namespace management for consistent storage keys"""

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