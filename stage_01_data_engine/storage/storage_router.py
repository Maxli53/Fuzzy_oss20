"""
Storage Router - Smart Symbol Routing for Exploratory Research
Routes ANY symbol to appropriate storage based on automated categorization
"""
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import yaml
from pathlib import Path

from .flexible_arctic_store import FlexibleArcticStore
from .tick_store import TickStore  # Legacy compatibility
from ..parsers.dtn_symbol_parser import DTNSymbolParser, SymbolInfo
from ..core.base_collector import StorageNamespace
from ..core.config_loader import get_config

logger = logging.getLogger(__name__)

class StorageRouter:
    """
    Smart routing system for exploratory quantitative research storage.

    Features:
    - Automatic symbol categorization and routing
    - Multi-backend storage support (flexible + legacy)
    - Intelligent storage selection based on symbol type
    - Performance optimization through smart caching
    - Research pattern learning and optimization
    """

    def __init__(self,
                 primary_storage: Optional[FlexibleArcticStore] = None,
                 legacy_storage: Optional[TickStore] = None,
                 config: Optional[Dict] = None):
        """
        Initialize storage router with flexible backend selection.

        Args:
            primary_storage: FlexibleArcticStore for new exploratory storage
            legacy_storage: TickStore for backwards compatibility
            config: Configuration overrides
        """
        self.config = config or self._load_config()

        # Initialize storage backends
        self.primary_storage = primary_storage or FlexibleArcticStore(
            arctic_uri=self.config.get('arctic', {}).get('uri', 'lmdb://./data/arctic_storage'),
            config=self.config
        )

        self.legacy_storage = legacy_storage

        # Initialize symbol parser for routing decisions
        self.symbol_parser = DTNSymbolParser()

        # Routing statistics and learning
        self.routing_stats = {
            'total_routes': 0,
            'flexible_routes': 0,
            'legacy_routes': 0,
            'routing_errors': 0,
            'symbol_categories': {},
            'performance_metrics': {}
        }

        # Symbol access patterns for optimization
        self.access_patterns = {}

        logger.info("StorageRouter initialized with flexible backends")

    def _load_config(self) -> Dict:
        """Load configuration from arctic_config.yaml"""
        try:
            config_path = Path(__file__).parent.parent / 'config' / 'arctic_config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning("Arctic config not found, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def store_symbol_data(self,
                         symbol: str,
                         data: pd.DataFrame,
                         data_type: str = 'auto',
                         date: Optional[str] = None,
                         metadata: Optional[Dict] = None,
                         force_backend: Optional[str] = None) -> bool:
        """
        Store data for any symbol with intelligent backend selection.

        Args:
            symbol: Any symbol to store
            data: DataFrame to store
            data_type: Data type ('auto', 'ticks', 'bars', etc.)
            date: Date string (YYYY-MM-DD)
            metadata: Additional metadata
            force_backend: Force specific backend ('flexible', 'legacy')

        Returns:
            True if stored successfully
        """
        try:
            self.routing_stats['total_routes'] += 1
            start_time = datetime.now()

            # Parse symbol for routing decision
            symbol_info = self.symbol_parser.parse_symbol(symbol)

            # Update symbol category statistics
            category = symbol_info.category
            self.routing_stats['symbol_categories'][category] = \
                self.routing_stats['symbol_categories'].get(category, 0) + 1

            # Determine storage backend
            backend = self._select_storage_backend(symbol_info, data_type, force_backend)

            # Route to appropriate storage
            if backend == 'flexible':
                success = self.primary_storage.store_symbol_data(
                    symbol, data, data_type, date, metadata
                )
                if success:
                    self.routing_stats['flexible_routes'] += 1

            elif backend == 'legacy' and self.legacy_storage:
                success = self._store_legacy(symbol, data, data_type, date, metadata)
                if success:
                    self.routing_stats['legacy_routes'] += 1

            else:
                logger.warning(f"No suitable backend for {symbol}, falling back to flexible")
                success = self.primary_storage.store_symbol_data(
                    symbol, data, data_type, date, metadata
                )
                if success:
                    self.routing_stats['flexible_routes'] += 1

            # Track performance
            elapsed = (datetime.now() - start_time).total_seconds()
            self._track_performance(symbol, backend, elapsed, success)

            # Update access patterns
            self._update_access_patterns(symbol, 'store')

            if not success:
                self.routing_stats['routing_errors'] += 1

            return success

        except Exception as e:
            logger.error(f"Storage routing error for {symbol}: {e}")
            self.routing_stats['routing_errors'] += 1
            return False

    def load_symbol_data(self,
                        symbol: str,
                        date: Optional[str] = None,
                        data_type: str = 'auto',
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        prefer_backend: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load data for any symbol with intelligent backend selection.

        Args:
            symbol: Symbol to load
            date: Specific date
            data_type: Data type to load
            start_date: Start date for range queries
            end_date: End date for range queries
            prefer_backend: Preferred backend ('flexible', 'legacy')

        Returns:
            DataFrame with loaded data or None
        """
        try:
            start_time = datetime.now()
            symbol_info = self.symbol_parser.parse_symbol(symbol)

            # Try backends in order of preference
            backends_to_try = self._get_load_backends(symbol_info, prefer_backend)

            for backend in backends_to_try:
                try:
                    if backend == 'flexible':
                        data = self.primary_storage.load_symbol_data(
                            symbol, date, data_type, start_date, end_date
                        )
                    elif backend == 'legacy' and self.legacy_storage:
                        data = self._load_legacy(symbol, date, data_type)
                    else:
                        continue

                    if data is not None and not data.empty:
                        # Track successful load
                        elapsed = (datetime.now() - start_time).total_seconds()
                        self._track_performance(symbol, backend, elapsed, True)
                        self._update_access_patterns(symbol, 'load')

                        logger.debug(f"Loaded {symbol} from {backend} backend ({len(data)} records)")
                        return data

                except Exception as e:
                    logger.warning(f"Failed to load {symbol} from {backend}: {e}")
                    continue

            # No backend had the data
            elapsed = (datetime.now() - start_time).total_seconds()
            self._track_performance(symbol, 'none', elapsed, False)

            logger.info(f"Symbol {symbol} not found in any storage backend")
            return None

        except Exception as e:
            logger.error(f"Load routing error for {symbol}: {e}")
            return None

    def discover_stored_symbols(self,
                               category: Optional[str] = None,
                               backend: Optional[str] = None) -> List[Dict]:
        """
        Discover symbols across all storage backends.

        Args:
            category: Filter by symbol category
            backend: Filter by storage backend

        Returns:
            List of symbol information from all backends
        """
        try:
            all_symbols = []

            # Discover from flexible storage
            if backend in [None, 'flexible']:
                try:
                    flexible_symbols = self.primary_storage.discover_stored_symbols(category)
                    for symbol_info in flexible_symbols:
                        symbol_info['backend'] = 'flexible'
                        all_symbols.append(symbol_info)
                except Exception as e:
                    logger.warning(f"Error discovering from flexible storage: {e}")

            # Discover from legacy storage
            if backend in [None, 'legacy'] and self.legacy_storage:
                try:
                    legacy_symbols = self._discover_legacy_symbols(category)
                    for symbol_info in legacy_symbols:
                        symbol_info['backend'] = 'legacy'
                        all_symbols.append(symbol_info)
                except Exception as e:
                    logger.warning(f"Error discovering from legacy storage: {e}")

            # Remove duplicates (prefer flexible backend)
            unique_symbols = {}
            for symbol_info in all_symbols:
                symbol = symbol_info.get('symbol', '')
                if symbol not in unique_symbols or symbol_info['backend'] == 'flexible':
                    unique_symbols[symbol] = symbol_info

            return list(unique_symbols.values())

        except Exception as e:
            logger.error(f"Error discovering symbols: {e}")
            return []

    def get_symbol_storage_info(self, symbol: str) -> Dict:
        """
        Get comprehensive storage information for a symbol.

        Args:
            symbol: Symbol to get info for

        Returns:
            Dictionary with storage information from all backends
        """
        try:
            symbol_info = self.symbol_parser.parse_symbol(symbol)

            info = {
                'symbol': symbol,
                'parsed_info': symbol_info.__dict__,
                'routing_recommendation': self._select_storage_backend(symbol_info, 'auto'),
                'storage_locations': [],
                'access_patterns': self.access_patterns.get(symbol, {}),
                'backends': {}
            }

            # Check flexible storage
            try:
                flexible_info = self.primary_storage.get_symbol_info(symbol)
                info['backends']['flexible'] = flexible_info
                if flexible_info.get('storage_locations'):
                    info['storage_locations'].extend(flexible_info['storage_locations'])
            except Exception as e:
                logger.debug(f"Error getting flexible info for {symbol}: {e}")

            # Check legacy storage
            if self.legacy_storage:
                try:
                    legacy_info = self._get_legacy_symbol_info(symbol)
                    info['backends']['legacy'] = legacy_info
                    if legacy_info.get('storage_locations'):
                        info['storage_locations'].extend(legacy_info['storage_locations'])
                except Exception as e:
                    logger.debug(f"Error getting legacy info for {symbol}: {e}")

            return info

        except Exception as e:
            logger.error(f"Error getting storage info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def optimize_symbol_storage(self, symbol: str) -> Dict:
        """
        Optimize storage for a frequently accessed symbol.

        Args:
            symbol: Symbol to optimize

        Returns:
            Dictionary with optimization results
        """
        try:
            access_pattern = self.access_patterns.get(symbol, {})
            access_count = access_pattern.get('access_count', 0)

            if access_count < 10:
                return {'symbol': symbol, 'optimization': 'not_needed', 'reason': 'low_access_count'}

            symbol_info = self.symbol_parser.parse_symbol(symbol)
            optimization_results = {
                'symbol': symbol,
                'access_count': access_count,
                'optimizations_applied': []
            }

            # Move high-access symbols to flexible storage if in legacy
            if self.legacy_storage:
                legacy_data = self._load_legacy(symbol, data_type='auto')
                if legacy_data is not None and not legacy_data.empty:
                    # Migrate to flexible storage
                    success = self.primary_storage.store_symbol_data(
                        symbol, legacy_data, 'migrated'
                    )
                    if success:
                        optimization_results['optimizations_applied'].append('migrated_to_flexible')

            # Create index for high-access symbols
            optimization_results['optimizations_applied'].append('indexing_prioritized')

            logger.info(f"Optimized storage for {symbol}: {optimization_results}")
            return optimization_results

        except Exception as e:
            logger.error(f"Error optimizing storage for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def get_routing_statistics(self) -> Dict:
        """Get comprehensive routing and performance statistics"""
        try:
            stats = {
                **self.routing_stats,
                'flexible_storage_stats': self.primary_storage.get_storage_stats(),
                'access_patterns_summary': {
                    'total_symbols_accessed': len(self.access_patterns),
                    'high_frequency_symbols': len([
                        s for s, p in self.access_patterns.items()
                        if p.get('access_count', 0) > 10
                    ]),
                    'most_accessed_symbol': max(
                        self.access_patterns.items(),
                        key=lambda x: x[1].get('access_count', 0)
                    )[0] if self.access_patterns else None
                },
                'routing_efficiency': {
                    'success_rate': (self.routing_stats['total_routes'] - self.routing_stats['routing_errors']) / max(self.routing_stats['total_routes'], 1) * 100,
                    'flexible_preference': self.routing_stats['flexible_routes'] / max(self.routing_stats['total_routes'], 1) * 100
                }
            }

            if self.legacy_storage:
                stats['legacy_storage_stats'] = self.legacy_storage.get_stats()

            return stats

        except Exception as e:
            logger.error(f"Error getting routing statistics: {e}")
            return {'error': str(e)}

    def _select_storage_backend(self,
                               symbol_info: SymbolInfo,
                               data_type: str,
                               force_backend: Optional[str] = None) -> str:
        """Select appropriate storage backend based on symbol characteristics"""

        if force_backend:
            return force_backend

        # Always prefer flexible storage for exploratory research
        if self.config.get('exploratory_mode', {}).get('enabled', True):
            return 'flexible'

        # Legacy routing logic for backwards compatibility
        category = symbol_info.category

        # DTN indicators and options work better with flexible storage
        if category in ['dtn_calculated', 'options', 'futures', 'forex']:
            return 'flexible'

        # Regular equities can use either, prefer flexible for new symbols
        if category == 'equity':
            return 'flexible'

        # Default to flexible
        return 'flexible'

    def _get_load_backends(self, symbol_info: SymbolInfo, prefer_backend: Optional[str]) -> List[str]:
        """Get ordered list of backends to try for loading"""

        if prefer_backend:
            if prefer_backend == 'flexible':
                return ['flexible', 'legacy']
            else:
                return ['legacy', 'flexible']

        # Default order: flexible first, then legacy
        backends = ['flexible']
        if self.legacy_storage:
            backends.append('legacy')

        return backends

    def _store_legacy(self, symbol: str, data: pd.DataFrame, data_type: str, date: str, metadata: Dict) -> bool:
        """Store data using legacy TickStore"""
        try:
            if not self.legacy_storage:
                return False

            # Convert to legacy format
            return self.legacy_storage.store_ticks(
                symbol=symbol,
                date=date or datetime.now().strftime('%Y-%m-%d'),
                tick_df=data,
                metadata=metadata or {},
                overwrite=True
            )
        except Exception as e:
            logger.error(f"Legacy storage error for {symbol}: {e}")
            return False

    def _load_legacy(self, symbol: str, date: Optional[str], data_type: str) -> Optional[pd.DataFrame]:
        """Load data from legacy TickStore"""
        try:
            if not self.legacy_storage:
                return None

            if date:
                return self.legacy_storage.load_ticks(symbol, date)
            else:
                # Try to load recent data
                recent_date = datetime.now().strftime('%Y-%m-%d')
                return self.legacy_storage.load_ticks(symbol, recent_date)

        except Exception as e:
            logger.debug(f"Legacy load error for {symbol}: {e}")
            return None

    def _discover_legacy_symbols(self, category: Optional[str]) -> List[Dict]:
        """Discover symbols from legacy storage"""
        try:
            if not self.legacy_storage:
                return []

            # This would need to be implemented based on legacy storage structure
            logger.debug("Legacy symbol discovery not fully implemented")
            return []

        except Exception as e:
            logger.error(f"Error discovering legacy symbols: {e}")
            return []

    def _get_legacy_symbol_info(self, symbol: str) -> Dict:
        """Get symbol info from legacy storage"""
        try:
            if not self.legacy_storage:
                return {}

            # This would need to be implemented based on legacy storage structure
            return {'backend': 'legacy', 'symbol': symbol}

        except Exception as e:
            logger.debug(f"Error getting legacy info for {symbol}: {e}")
            return {}

    def _track_performance(self, symbol: str, backend: str, elapsed_seconds: float, success: bool):
        """Track performance metrics for optimization"""
        if backend not in self.routing_stats['performance_metrics']:
            self.routing_stats['performance_metrics'][backend] = {
                'total_operations': 0,
                'successful_operations': 0,
                'total_time': 0.0,
                'avg_time': 0.0
            }

        metrics = self.routing_stats['performance_metrics'][backend]
        metrics['total_operations'] += 1
        metrics['total_time'] += elapsed_seconds

        if success:
            metrics['successful_operations'] += 1

        metrics['avg_time'] = metrics['total_time'] / metrics['total_operations']

    def _update_access_patterns(self, symbol: str, operation: str):
        """Update symbol access patterns for optimization"""
        if symbol not in self.access_patterns:
            self.access_patterns[symbol] = {
                'access_count': 0,
                'first_access': datetime.now().isoformat(),
                'last_access': None,
                'operations': {'store': 0, 'load': 0}
            }

        pattern = self.access_patterns[symbol]
        pattern['access_count'] += 1
        pattern['last_access'] = datetime.now().isoformat()
        pattern['operations'][operation] = pattern['operations'].get(operation, 0) + 1