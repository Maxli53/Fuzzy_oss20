"""
Centralized Configuration Management for Stage 1 Data Engine
Loads and manages all configuration files
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Centralized configuration management system.
    Loads and provides access to all configuration files.
    """

    def __init__(self, config_dir: str = "stage_01_data_engine/config"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_all_configs()

    def _load_all_configs(self):
        """Load all configuration files"""
        config_files = {
            'symbol': 'symbol_config.yaml',
            'bar': 'bar_config.yaml',
            'indicator': 'indicator_config.yaml',
            'storage': 'storage_config.yaml',
            'stream': 'stream_config.yaml'
        }

        for config_name, filename in config_files.items():
            config_path = self.config_dir / filename
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        self.configs[config_name] = yaml.safe_load(f)
                    logger.info(f"Loaded {config_name} configuration")
                except Exception as e:
                    logger.error(f"Error loading {config_name} config: {e}")
                    self.configs[config_name] = {}
            else:
                logger.warning(f"Config file not found: {config_path}")
                self.configs[config_name] = {}

    def get(self, config_name: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            config_name: Name of configuration (symbol, bar, etc.)
            key: Specific key to retrieve (None for entire config)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if config_name not in self.configs:
            logger.warning(f"Configuration '{config_name}' not found")
            return default

        config = self.configs[config_name]

        if key is None:
            return config

        return config.get(key, default)

    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """
        Get configuration for specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Symbol configuration dict
        """
        symbol_configs = self.get('symbol', default={})

        # Return symbol-specific config or DEFAULT
        return symbol_configs.get(symbol, symbol_configs.get('DEFAULT', {}))

    def get_bar_config(self, bar_type: str) -> Dict[str, Any]:
        """
        Get configuration for specific bar type.

        Args:
            bar_type: Type of bar (volume, dollar, etc.)

        Returns:
            Bar configuration dict
        """
        bar_configs = self.get('bar', default={})
        return bar_configs.get(bar_type, {})

    def get_indicator_mapping(self, indicator_type: str) -> Dict[str, str]:
        """
        Get DTN indicator symbol mappings.

        Args:
            indicator_type: Type of indicator (breadth, sentiment, etc.)

        Returns:
            Dictionary mapping indicator names to symbols
        """
        indicator_configs = self.get('indicator', default={})
        return indicator_configs.get(indicator_type, {})

    def get_storage_config(self, store_type: str) -> Dict[str, Any]:
        """
        Get configuration for storage type.

        Args:
            store_type: Type of storage (tick, bar, etc.)

        Returns:
            Storage configuration dict
        """
        storage_configs = self.get('storage', default={})
        return storage_configs.get(store_type, {})

    def get_stream_config(self, stream_type: str) -> Dict[str, Any]:
        """
        Get configuration for streaming type.

        Args:
            stream_type: Type of stream (level1, internals, etc.)

        Returns:
            Stream configuration dict
        """
        stream_configs = self.get('stream', default={})
        return stream_configs.get(stream_type, {})

    def reload_configs(self):
        """Reload all configuration files"""
        logger.info("Reloading all configurations")
        self._load_all_configs()

    def list_configs(self) -> Dict[str, Dict]:
        """List all loaded configurations"""
        return {name: list(config.keys()) for name, config in self.configs.items()}


# Global configuration instance
_config_loader = None


def get_config_loader() -> ConfigLoader:
    """Get global configuration loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def get_config(config_name: str, key: Optional[str] = None, default: Any = None) -> Any:
    """Convenience function to get configuration"""
    return get_config_loader().get(config_name, key, default)


def get_symbol_config(symbol: str) -> Dict[str, Any]:
    """Convenience function to get symbol configuration"""
    return get_config_loader().get_symbol_config(symbol)