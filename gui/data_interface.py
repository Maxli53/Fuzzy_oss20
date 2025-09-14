"""
DataEngine Interface for Streamlit GUI
Provides simplified interface for GUI operations with real data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import traceback

logger = logging.getLogger(__name__)

try:
    from stage_01_data_engine.core.data_engine import DataEngine
    from stage_01_data_engine.parsers.dtn_symbol_parser import DTNSymbolParser
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import data engine components: {e}")
    DataEngine = None
    DTNSymbolParser = None
    IMPORTS_AVAILABLE = False

class GUIDataInterface:
    """
    Simplified interface to DataEngine for GUI operations
    Handles error management and provides user-friendly responses
    """

    def __init__(self):
        """Initialize the data interface"""
        self.data_engine = None
        self.symbol_parser = DTNSymbolParser() if DTNSymbolParser else None
        self.last_error = None

        if not IMPORTS_AVAILABLE:
            self.last_error = "Data engine components not available. Check dependencies."
        self.connection_status = {
            'data_engine': False,
            'iqfeed': False,
            'polygon': False,
            'storage': False
        }

        self._initialize_data_engine()

    def _initialize_data_engine(self) -> bool:
        """Initialize DataEngine with error handling"""
        if not IMPORTS_AVAILABLE or not DataEngine:
            self.last_error = "DataEngine class not available"
            return False

        try:
            self.data_engine = DataEngine()
            self.connection_status['data_engine'] = True
            self.connection_status['storage'] = True

            # Test collector connectivity
            try:
                self.connection_status['iqfeed'] = self.data_engine.iqfeed_collector is not None
            except Exception:
                self.connection_status['iqfeed'] = False

            try:
                self.connection_status['polygon'] = self.data_engine.polygon_collector is not None
            except Exception:
                self.connection_status['polygon'] = False

            logger.info("DataEngine initialized successfully for GUI")
            return True

        except Exception as e:
            self.last_error = f"Failed to initialize DataEngine: {str(e)}"
            logger.error(f"DataEngine initialization failed: {e}")
            self.connection_status['data_engine'] = False
            return False

    def get_connection_status(self) -> Dict[str, bool]:
        """Get current connection status for all components"""
        return self.connection_status.copy()

    def parse_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Parse symbol and return structured information

        Args:
            symbol: Symbol to parse

        Returns:
            Dictionary with parsing results and metadata
        """
        try:
            symbol_info = self.symbol_parser.parse_symbol(symbol)

            return {
                'success': True,
                'symbol': symbol,
                'parsed_info': {
                    'category': symbol_info.category,
                    'subcategory': symbol_info.subcategory,
                    'base_symbol': symbol_info.base_symbol,
                    'exchange': symbol_info.exchange,
                    'instrument_type': symbol_info.instrument_type,
                    'is_valid': symbol_info.is_valid
                },
                'routing_recommendation': self._get_storage_recommendation(symbol_info),
                'error': None
            }

        except Exception as e:
            error_msg = f"Symbol parsing failed: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'symbol': symbol,
                'parsed_info': None,
                'routing_recommendation': None,
                'error': error_msg
            }

    def fetch_real_data(self,
                       symbol: str,
                       data_type: str = 'ticks',
                       lookback_days: int = 1,
                       max_records: int = 1000) -> Dict[str, Any]:
        """
        Fetch real data for symbol using available collectors

        Args:
            symbol: Symbol to fetch
            data_type: Type of data ('ticks', 'bars', etc.)
            lookback_days: Days to look back
            max_records: Maximum records to fetch

        Returns:
            Dictionary with fetch results
        """
        if not self.data_engine:
            return {
                'success': False,
                'data': None,
                'error': 'DataEngine not initialized',
                'source': None,
                'metadata': {}
            }

        try:
            # Try IQFeed first
            if self.connection_status['iqfeed']:
                try:
                    logger.info(f"Attempting to fetch {symbol} from IQFeed")
                    data = self.data_engine.iqfeed_collector.fetch(
                        symbols=[symbol],
                        data_type=data_type,
                        lookback_days=lookback_days,
                        max_ticks=max_records if data_type == 'ticks' else None
                    )

                    if data and symbol in data and not data[symbol].empty:
                        return {
                            'success': True,
                            'data': data[symbol],
                            'error': None,
                            'source': 'IQFeed',
                            'metadata': {
                                'records_count': len(data[symbol]),
                                'date_range': f"{data[symbol].index.min()} to {data[symbol].index.max()}",
                                'columns': list(data[symbol].columns)
                            }
                        }
                except Exception as e:
                    logger.warning(f"IQFeed fetch failed for {symbol}: {e}")

            # Try Polygon as fallback
            if self.connection_status['polygon']:
                try:
                    logger.info(f"Attempting to fetch {symbol} from Polygon")
                    data = self.data_engine.polygon_collector.collect(
                        symbols=[symbol],
                        data_type=data_type,
                        lookback_days=lookback_days
                    )

                    if data and symbol in data and not data[symbol].empty:
                        return {
                            'success': True,
                            'data': data[symbol],
                            'error': None,
                            'source': 'Polygon',
                            'metadata': {
                                'records_count': len(data[symbol]),
                                'date_range': f"{data[symbol].index.min()} to {data[symbol].index.max()}",
                                'columns': list(data[symbol].columns)
                            }
                        }
                except Exception as e:
                    logger.warning(f"Polygon fetch failed for {symbol}: {e}")

            # No data from any source
            return {
                'success': False,
                'data': None,
                'error': 'No data available from any source',
                'source': None,
                'metadata': {}
            }

        except Exception as e:
            error_msg = f"Data fetch failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                'success': False,
                'data': None,
                'error': error_msg,
                'source': None,
                'metadata': {}
            }

    def store_data(self, symbol: str, data: pd.DataFrame, data_type: str = 'ticks') -> Dict[str, Any]:
        """
        Store data using the flexible storage system

        Args:
            symbol: Symbol being stored
            data: DataFrame to store
            data_type: Type of data

        Returns:
            Dictionary with storage results
        """
        if not self.data_engine:
            return {
                'success': False,
                'storage_location': None,
                'backend_used': None,
                'error': 'DataEngine not initialized'
            }

        try:
            # Store using storage router
            success = self.data_engine.storage_router.store_symbol_data(
                symbol=symbol,
                data=data,
                data_type=data_type,
                date=datetime.now().strftime('%Y-%m-%d')
            )

            if success:
                # Get storage info to show where it was stored
                storage_info = self.data_engine.storage_router.get_symbol_storage_info(symbol)

                return {
                    'success': True,
                    'storage_location': storage_info.get('storage_locations', []),
                    'backend_used': storage_info.get('routing_recommendation', 'flexible'),
                    'error': None,
                    'metadata': {
                        'records_stored': len(data),
                        'storage_info': storage_info
                    }
                }
            else:
                return {
                    'success': False,
                    'storage_location': None,
                    'backend_used': None,
                    'error': 'Storage operation failed'
                }

        except Exception as e:
            error_msg = f"Storage failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                'success': False,
                'storage_location': None,
                'backend_used': None,
                'error': error_msg
            }

    def retrieve_stored_data(self,
                           symbol: str,
                           date: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve stored data for verification

        Args:
            symbol: Symbol to retrieve
            date: Specific date
            start_date: Start date for range
            end_date: End date for range

        Returns:
            Dictionary with retrieval results
        """
        if not self.data_engine:
            return {
                'success': False,
                'data': None,
                'error': 'DataEngine not initialized',
                'source_backend': None
            }

        try:
            data = self.data_engine.storage_router.load_symbol_data(
                symbol=symbol,
                date=date,
                start_date=start_date,
                end_date=end_date
            )

            if data is not None and not data.empty:
                return {
                    'success': True,
                    'data': data,
                    'error': None,
                    'source_backend': 'determined_by_router',
                    'metadata': {
                        'records_retrieved': len(data),
                        'date_range': f"{data.index.min()} to {data.index.max()}",
                        'columns': list(data.columns)
                    }
                }
            else:
                return {
                    'success': False,
                    'data': None,
                    'error': 'No stored data found for symbol',
                    'source_backend': None
                }

        except Exception as e:
            error_msg = f"Data retrieval failed: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {
                'success': False,
                'data': None,
                'error': error_msg,
                'source_backend': None
            }

    def discover_stored_symbols(self) -> Dict[str, Any]:
        """
        Discover all symbols stored in the system

        Returns:
            Dictionary with discovered symbols
        """
        if not self.data_engine:
            return {
                'success': False,
                'symbols': [],
                'error': 'DataEngine not initialized'
            }

        try:
            symbols = self.data_engine.storage_router.discover_stored_symbols()

            return {
                'success': True,
                'symbols': symbols,
                'error': None,
                'metadata': {
                    'total_symbols': len(symbols),
                    'categories': self._categorize_symbols(symbols)
                }
            }

        except Exception as e:
            error_msg = f"Symbol discovery failed: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'symbols': [],
                'error': error_msg
            }

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        if not self.data_engine:
            return {'error': 'DataEngine not initialized'}

        try:
            stats = self.data_engine.storage_router.get_routing_statistics()
            engine_stats = self.data_engine.stats

            return {
                'success': True,
                'routing_stats': stats,
                'engine_stats': engine_stats,
                'connection_status': self.connection_status,
                'error': None
            }

        except Exception as e:
            error_msg = f"Statistics retrieval failed: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg
            }

    def _get_storage_recommendation(self, symbol_info) -> str:
        """Get storage backend recommendation for symbol"""
        if not self.data_engine:
            return 'flexible'  # Default recommendation

        try:
            return self.data_engine.storage_router._select_storage_backend(
                symbol_info, 'auto'
            )
        except Exception:
            return 'flexible'

    def _categorize_symbols(self, symbols: List[Dict]) -> Dict[str, int]:
        """Categorize discovered symbols by type"""
        categories = {}
        for symbol_info in symbols:
            category = symbol_info.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        return categories

    def perform_round_trip_test(self, symbol: str) -> Dict[str, Any]:
        """
        Perform complete round-trip test: fetch → store → retrieve → verify

        Args:
            symbol: Symbol to test

        Returns:
            Dictionary with complete test results
        """
        test_results = {
            'symbol': symbol,
            'test_timestamp': datetime.now().isoformat(),
            'steps': {},
            'overall_success': False,
            'error': None
        }

        try:
            # Step 1: Parse symbol
            parse_result = self.parse_symbol(symbol)
            test_results['steps']['parse'] = parse_result

            if not parse_result['success']:
                test_results['error'] = 'Symbol parsing failed'
                return test_results

            # Step 2: Fetch data
            fetch_result = self.fetch_real_data(symbol)
            test_results['steps']['fetch'] = fetch_result

            if not fetch_result['success']:
                test_results['error'] = 'Data fetching failed'
                return test_results

            # Step 3: Store data
            store_result = self.store_data(symbol, fetch_result['data'])
            test_results['steps']['store'] = store_result

            if not store_result['success']:
                test_results['error'] = 'Data storage failed'
                return test_results

            # Step 4: Retrieve data
            retrieve_result = self.retrieve_stored_data(symbol)
            test_results['steps']['retrieve'] = retrieve_result

            if not retrieve_result['success']:
                test_results['error'] = 'Data retrieval failed'
                return test_results

            # Step 5: Verify data integrity
            original_data = fetch_result['data']
            retrieved_data = retrieve_result['data']

            verification = {
                'success': True,
                'original_records': len(original_data),
                'retrieved_records': len(retrieved_data),
                'columns_match': list(original_data.columns) == list(retrieved_data.columns),
                'data_types_match': str(original_data.dtypes) == str(retrieved_data.dtypes)
            }

            test_results['steps']['verify'] = verification
            test_results['overall_success'] = verification['success']

            return test_results

        except Exception as e:
            test_results['error'] = f"Round-trip test failed: {str(e)}"
            logger.error(f"Round-trip test error: {e}\n{traceback.format_exc()}")
            return test_results