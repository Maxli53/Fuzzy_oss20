"""
Unified Data Engine Interface
Single point of access for all Stage 1 data collection and processing
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import logging

from stage_01_data_engine.core.config_loader import ConfigLoader, get_config_loader
from stage_01_data_engine.collectors.iqfeed_collector import IQFeedCollector
from stage_01_data_engine.collectors.polygon_collector import PolygonCollector
from stage_01_data_engine.parsers.dtn_symbol_parser import DTNSymbolParser
from stage_01_data_engine.storage.flexible_arctic_store import FlexibleArcticStore
from stage_01_data_engine.storage.storage_router import StorageRouter
from stage_01_data_engine.storage.tick_store import TickStore  # Legacy support

logger = logging.getLogger(__name__)

class DataEngine:
    """
    Unified interface for Stage 1 data collection and access.

    Provides single entry point for:
    - Tick data collection and retrieval
    - DTN indicators collection
    - Bar construction and advanced analytics
    - Market regime detection
    - Data storage and caching
    """

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize data engine with flexible collectors and storage"""

        # Initialize configuration
        self.config = get_config_loader()
        if config_dir:
            self.config.config_dir = config_dir
            self.config.reload_configs()

        # Initialize flexible collectors
        self.iqfeed_collector = IQFeedCollector()
        self.polygon_collector = PolygonCollector()
        self.symbol_parser = DTNSymbolParser()

        # Initialize flexible storage with routing
        self.flexible_store = FlexibleArcticStore(config=self.config.config if hasattr(self.config, 'config') else {})
        self.legacy_store = TickStore()  # Keep for backwards compatibility
        self.storage_router = StorageRouter(
            primary_storage=self.flexible_store,
            legacy_storage=self.legacy_store,
            config=self.config.config if hasattr(self.config, 'config') else {}
        )

        # Symbol universe management for exploratory research
        self.discovered_symbols = set()
        self.symbol_metadata_cache = {}

        # Performance tracking
        self.stats = {
            'collections_today': 0,
            'data_points_collected': 0,
            'last_collection': None,
            'active_symbols': set(),
            'discovered_symbols_count': 0,
            'errors_today': 0,
            'fetch_requests': 0,
            'exploration_requests': 0
        }

        logger.info("DataEngine initialized with flexible architecture for exploratory research")

    def collect_market_snapshot(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Collect complete market snapshot for given symbols.

        Args:
            symbols: List of symbols to collect

        Returns:
            Dictionary with 'ticks', 'indicators', and 'sentiment' DataFrames
        """
        snapshot = {}
        timestamp = datetime.now()

        try:
            logger.info(f"Collecting market snapshot for {len(symbols)} symbols")

            # Collect tick data using flexible fetch
            logger.info("Collecting tick data...")
            tick_results = self.iqfeed_collector.fetch(
                symbols,
                data_type='ticks',
                lookback_days=1,
                max_ticks=1000
            )

            # Combine all tick data
            all_tick_data = []
            for symbol, df in tick_results.items():
                if df is not None and not df.empty:
                    all_tick_data.append(df)

            if all_tick_data:
                snapshot['ticks'] = pd.concat(all_tick_data, ignore_index=True)
                logger.info(f"Collected {len(snapshot['ticks'])} ticks")
            else:
                snapshot['ticks'] = pd.DataFrame()
                logger.warning("No tick data collected")

            # Collect DTN indicators
            logger.info("Collecting market indicators...")
            key_indicators = ['JTNT.Z', 'RINT.Z', 'TCOEA.Z', 'VCOET.Z']

            indicator_results = self.iqfeed_collector.fetch(
                key_indicators,
                data_type='auto',
                auto_categorize=True
            )

            # Combine indicator data
            all_indicator_data = []
            for symbol, df in indicator_results.items():
                if df is not None and not df.empty:
                    all_indicator_data.append(df)

            if all_indicator_data:
                snapshot['indicators'] = pd.concat(all_indicator_data, ignore_index=True)
                logger.info(f"Collected {len(snapshot['indicators'])} indicators")
            else:
                snapshot['indicators'] = pd.DataFrame()
                logger.warning("No indicator data collected")

            # Get sentiment snapshot
            logger.info("Calculating sentiment metrics...")
            sentiment_data = self.iqfeed_collector.get_market_sentiment_snapshot()
            if sentiment_data:
                sentiment_df = pd.DataFrame([{
                    'timestamp': timestamp,
                    'metric': k,
                    'value': v
                } for k, v in sentiment_data.items()])
                snapshot['sentiment'] = sentiment_df
                logger.info(f"Calculated {len(sentiment_data)} sentiment metrics")
            else:
                snapshot['sentiment'] = pd.DataFrame()

            # Collect news sentiment from Polygon (if available)
            logger.info("Collecting news sentiment...")
            try:
                news_sentiment = self.polygon_collector.get_market_wide_sentiment()
                if news_sentiment and news_sentiment.get('overall_sentiment', 0) != 0:
                    news_df = pd.DataFrame([{
                        'timestamp': timestamp,
                        'metric': f'polygon_{k}',
                        'value': v
                    } for k, v in news_sentiment.items() if isinstance(v, (int, float))])

                    if not news_df.empty:
                        if not snapshot['sentiment'].empty:
                            snapshot['sentiment'] = pd.concat([snapshot['sentiment'], news_df], ignore_index=True)
                        else:
                            snapshot['sentiment'] = news_df
                        logger.info(f"Added {len(news_df)} news sentiment metrics")
            except Exception as e:
                logger.warning(f"News sentiment collection failed (expected for free tier): {e}")

            # Update stats
            self._update_stats(symbols, snapshot)

            return snapshot

        except Exception as e:
            logger.error(f"Error collecting market snapshot: {e}")
            self.stats['errors_today'] += 1
            return {
                'ticks': pd.DataFrame(),
                'indicators': pd.DataFrame(),
                'sentiment': pd.DataFrame()
            }

    def collect_historical_data(self, symbols: List[str],
                              days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Collect historical data for backtesting and analysis.

        Args:
            symbols: Symbols to collect
            days: Number of days of historical data

        Returns:
            Dictionary with historical data
        """
        try:
            logger.info(f"Collecting {days} days of historical data for {len(symbols)} symbols")

            historical = {}

            # Collect historical indicators using flexible fetch
            all_indicators = ['JTNT.Z', 'RINT.Z', 'TCOEA.Z', 'VCOET.Z']

            indicator_results = self.iqfeed_collector.fetch(
                all_indicators,
                data_type='auto',
                lookback_days=min(days, 5),  # Limit lookback
                auto_categorize=True
            )

            all_indicator_data = []
            for symbol, df in indicator_results.items():
                if df is not None and not df.empty:
                    all_indicator_data.append(df)

            if all_indicator_data:
                historical['indicators'] = pd.concat(all_indicator_data, ignore_index=True)
                logger.info(f"Collected historical indicators: {len(historical['indicators'])} records")

            # For each symbol, try to get some tick data using flexible fetch
            tick_results = self.iqfeed_collector.fetch(
                symbols,
                data_type='ticks',
                lookback_days=min(days, 5),  # Limit tick data collection
                max_ticks=10000
            )

            all_tick_data = []
            for symbol, df in tick_results.items():
                if df is not None and not df.empty:
                    all_tick_data.append(df)

            if all_tick_data:
                historical['ticks'] = pd.concat(all_tick_data, ignore_index=True)
                logger.info(f"Collected historical ticks: {len(historical['ticks'])} records")

            return historical

        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
            return {}

    def get_market_regime(self) -> Dict[str, str]:
        """
        Analyze current market regime based on indicators.

        Returns:
            Dictionary with regime classifications
        """
        try:
            # Get current sentiment snapshot
            sentiment = self.iqfeed_collector.get_market_sentiment_snapshot()

            if not sentiment:
                return {'regime': 'UNKNOWN', 'confidence': 'LOW'}

            regime = {}

            # Analyze momentum regime (based on TICK)
            tick_value = sentiment.get('NYSE_TICK', 0)
            if tick_value > 1000:
                regime['momentum'] = 'BULLISH_STRONG'
            elif tick_value > 500:
                regime['momentum'] = 'BULLISH_MODERATE'
            elif tick_value < -1000:
                regime['momentum'] = 'BEARISH_STRONG'
            elif tick_value < -500:
                regime['momentum'] = 'BEARISH_MODERATE'
            else:
                regime['momentum'] = 'NEUTRAL'

            # Analyze liquidity regime (based on TRIN)
            trin_value = sentiment.get('NYSE_TRIN', 1.0)
            if trin_value > 2.0:
                regime['liquidity'] = 'STRESSED'
            elif trin_value > 1.5:
                regime['liquidity'] = 'TIGHT'
            elif trin_value < 0.5:
                regime['liquidity'] = 'ABUNDANT'
            elif trin_value < 0.8:
                regime['liquidity'] = 'LOOSE'
            else:
                regime['liquidity'] = 'NORMAL'

            # Analyze sentiment regime (based on Put/Call)
            pc_ratio = sentiment.get('TOTAL_PC_RATIO', 1.0)
            if pc_ratio > 1.3:
                regime['sentiment'] = 'FEARFUL'
            elif pc_ratio > 1.1:
                regime['sentiment'] = 'CAUTIOUS'
            elif pc_ratio < 0.7:
                regime['sentiment'] = 'GREEDY'
            elif pc_ratio < 0.9:
                regime['sentiment'] = 'OPTIMISTIC'
            else:
                regime['sentiment'] = 'NEUTRAL'

            # Calculate composite regime
            regime['composite'] = self._calculate_composite_regime(regime)

            # Calculate confidence based on signal strength
            regime['confidence'] = self._calculate_confidence(sentiment)

            logger.info(f"Market regime: {regime}")
            return regime

        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return {'regime': 'ERROR', 'confidence': 'LOW'}

    def _calculate_composite_regime(self, regime: Dict[str, str]) -> str:
        """Calculate composite market regime from component regimes"""
        try:
            momentum = regime.get('momentum', 'NEUTRAL')
            liquidity = regime.get('liquidity', 'NORMAL')
            sentiment = regime.get('sentiment', 'NEUTRAL')

            # Simple composite logic
            bullish_signals = sum([
                'BULLISH' in momentum,
                liquidity in ['ABUNDANT', 'LOOSE'],
                sentiment in ['GREEDY', 'OPTIMISTIC']
            ])

            bearish_signals = sum([
                'BEARISH' in momentum,
                liquidity in ['STRESSED', 'TIGHT'],
                sentiment in ['FEARFUL', 'CAUTIOUS']
            ])

            if bullish_signals >= 2:
                return 'RISK_ON'
            elif bearish_signals >= 2:
                return 'RISK_OFF'
            else:
                return 'MIXED'

        except Exception as e:
            logger.error(f"Error calculating composite regime: {e}")
            return 'UNKNOWN'

    def _calculate_confidence(self, sentiment: Dict[str, float]) -> str:
        """Calculate confidence in regime assessment"""
        try:
            # Calculate signal strength
            tick_strength = abs(sentiment.get('NYSE_TICK', 0)) / 1500
            trin_deviation = abs(sentiment.get('NYSE_TRIN', 1.0) - 1.0) / 1.0
            pc_deviation = abs(sentiment.get('TOTAL_PC_RATIO', 1.0) - 1.0) / 0.3

            avg_strength = np.mean([tick_strength, trin_deviation, pc_deviation])

            if avg_strength > 1.0:
                return 'HIGH'
            elif avg_strength > 0.5:
                return 'MEDIUM'
            else:
                return 'LOW'

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 'LOW'

    def store_snapshot(self, snapshot: Dict[str, pd.DataFrame]) -> bool:
        """
        Store market snapshot to persistent storage using flexible routing.

        Args:
            snapshot: Market snapshot data

        Returns:
            True if successful
        """
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            success = True
            stored_symbols = []

            # Store tick data using flexible storage
            if 'ticks' in snapshot and not snapshot['ticks'].empty:
                for symbol in snapshot['ticks']['symbol'].unique():
                    symbol_ticks = snapshot['ticks'][
                        snapshot['ticks']['symbol'] == symbol
                    ].copy()

                    stored = self.storage_router.store_symbol_data(
                        symbol=symbol,
                        data=symbol_ticks,
                        data_type='ticks',
                        date=today,
                        metadata={'source': 'data_engine_snapshot'}
                    )
                    if stored:
                        stored_symbols.append(symbol)
                    else:
                        success = False

            # Store indicator data
            if 'indicators' in snapshot and not snapshot['indicators'].empty:
                for _, indicator_row in snapshot['indicators'].iterrows():
                    symbol = indicator_row.get('symbol')
                    if symbol:
                        # Create single-row DataFrame for storage
                        indicator_data = pd.DataFrame([indicator_row])

                        stored = self.storage_router.store_symbol_data(
                            symbol=symbol,
                            data=indicator_data,
                            data_type='indicators',
                            date=today,
                            metadata={'source': 'data_engine_snapshot', 'data_type': 'dtn_calculated'}
                        )
                        if stored:
                            stored_symbols.append(symbol)
                        else:
                            success = False

            # Store sentiment data
            if 'sentiment' in snapshot and not snapshot['sentiment'].empty:
                # Store sentiment as market-wide indicator
                stored = self.storage_router.store_symbol_data(
                    symbol='MARKET_SENTIMENT',
                    data=snapshot['sentiment'],
                    data_type='sentiment',
                    date=today,
                    metadata={'source': 'data_engine_snapshot', 'data_type': 'market_sentiment'}
                )
                if stored:
                    stored_symbols.append('MARKET_SENTIMENT')
                else:
                    success = False

            # Update discovery cache
            for symbol in stored_symbols:
                self.discovered_symbols.add(symbol)

            logger.info(f"Snapshot storage {'successful' if success else 'partially failed'} - stored {len(stored_symbols)} symbols")
            return success

        except Exception as e:
            logger.error(f"Error storing snapshot: {e}")
            return False

    def _update_stats(self, symbols: List[str], snapshot: Dict[str, pd.DataFrame]):
        """Update engine statistics"""
        self.stats['collections_today'] += 1
        self.stats['last_collection'] = datetime.now()
        self.stats['active_symbols'].update(symbols)

        # Count data points
        total_points = sum(len(df) for df in snapshot.values() if not df.empty)
        self.stats['data_points_collected'] += total_points

    def get_stats(self) -> Dict[str, Any]:
        """Get data engine statistics"""
        return {
            **self.stats,
            'active_symbols': list(self.stats['active_symbols']),
            'active_symbol_count': len(self.stats['active_symbols'])
        }

    def health_check(self) -> Dict[str, str]:
        """Perform system health check"""
        health = {
            'overall': 'HEALTHY',
            'tick_collector': 'UNKNOWN',
            'dtn_collector': 'UNKNOWN',
            'storage': 'UNKNOWN'
        }

        try:
            # Test tick collector
            test_ticks = self.tick_collector.collect(['AAPL'], num_days=1, max_ticks=10)
            health['tick_collector'] = 'HEALTHY' if test_ticks is not None else 'DEGRADED'

            # Test DTN collector
            test_indicators = self.dtn_collector.collect(['NYSE_TICK'], data_type='snapshot')
            health['dtn_collector'] = 'HEALTHY' if test_indicators is not None else 'DEGRADED'

            # Test storage
            stats = self.tick_store.get_storage_stats()
            health['storage'] = 'HEALTHY' if stats else 'DEGRADED'

            # Overall health
            if all(status == 'HEALTHY' for key, status in health.items() if key != 'overall'):
                health['overall'] = 'HEALTHY'
            elif any(status == 'DEGRADED' for status in health.values()):
                health['overall'] = 'DEGRADED'
            else:
                health['overall'] = 'UNHEALTHY'

        except Exception as e:
            logger.error(f"Health check error: {e}")
            health['overall'] = 'ERROR'

        return health

    # ===========================================
    # EXPLORATORY RESEARCH METHODS
    # ===========================================

    def fetch_any(self, symbols: Union[str, List[str]], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Flexible method to fetch ANY symbols without preconfiguration.
        Perfect for exploratory quantitative research.

        Args:
            symbols: Any symbol(s) - stocks, options, futures, DTN indicators
            data_type: 'auto' (detect), 'ticks', 'bars', 'quotes', 'news'
            lookback_days: Number of days to look back
            include_news: Whether to include news sentiment (Polygon)
            auto_categorize: Whether to use symbol parser for smart routing

        Returns:
            Dict mapping symbol to collected DataFrame with metadata
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        self.stats['fetch_requests'] += 1
        logger.info(f"Flexible fetch requested for {len(symbols)} symbols: {symbols}")

        results = {}
        include_news = kwargs.get('include_news', False)

        try:
            # Use IQFeed collector for primary data
            iqfeed_results = self.iqfeed_collector.fetch(symbols, **kwargs)

            for symbol, df in iqfeed_results.items():
                if df is not None and not df.empty:
                    # Get symbol metadata
                    metadata = self.symbol_parser.get_symbol_metadata(symbol)

                    # Add metadata to results
                    results[symbol] = {
                        'data': df,
                        'metadata': metadata,
                        'source': 'iqfeed',
                        'fetched_at': pd.Timestamp.now().isoformat()
                    }

                    # Update discovered symbols cache
                    self.discovered_symbols.add(symbol)
                    self.symbol_metadata_cache[symbol] = metadata

            # Add news sentiment if requested and available
            if include_news and symbols:
                try:
                    # Get news for symbols that are regular equities
                    equity_symbols = [
                        s for s in symbols
                        if self.symbol_parser.get_symbol_category(s) == 'equity'
                    ]

                    if equity_symbols:
                        news_results = self.polygon_collector.collect(
                            equity_symbols[:5],  # Limit for free tier
                            data_type='news',
                            lookback_days=kwargs.get('lookback_days', 7)
                        )

                        if news_results is not None and not news_results.empty:
                            for symbol in equity_symbols:
                                symbol_news = news_results[news_results['symbol'] == symbol]
                                if not symbol_news.empty and symbol in results:
                                    results[symbol]['news'] = symbol_news
                                    logger.info(f"Added {len(symbol_news)} news articles for {symbol}")

                except Exception as e:
                    logger.warning(f"News collection failed (expected for free tier): {e}")

            # Update stats
            self.stats['data_points_collected'] += sum(
                len(r['data']) for r in results.values() if 'data' in r and r['data'] is not None
            )
            self.stats['discovered_symbols_count'] = len(self.discovered_symbols)

            logger.info(f"Flexible fetch completed: {len(results)} symbols with data")
            return results

        except Exception as e:
            logger.error(f"Error in flexible fetch: {e}")
            self.stats['errors_today'] += 1
            return {}

    def explore(self, symbols: Union[str, List[str]],
                deep_analysis: bool = False) -> Dict[str, Dict]:
        """
        Explore symbols to understand what data is available and get recommendations.
        Perfect for discovering new trading opportunities.

        Args:
            symbols: Symbol(s) to explore
            deep_analysis: Whether to perform deep analysis (fetch sample data)

        Returns:
            Dict with exploration results for each symbol
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        self.stats['exploration_requests'] += 1
        logger.info(f"Exploring {len(symbols)} symbols: {symbols}")

        exploration_results = {}

        for symbol in symbols:
            try:
                # Parse symbol to understand what it is
                exploration = self.iqfeed_collector.explore_symbol(symbol)

                # Add news potential (from Polygon)
                symbol_category = self.symbol_parser.get_symbol_category(symbol)
                exploration['news_availability'] = symbol_category == 'equity'

                # Deep analysis if requested
                if deep_analysis:
                    logger.info(f"Performing deep analysis for {symbol}")

                    # Try to fetch sample data
                    sample_data = self.fetch_any([symbol], lookback_days=1)
                    if symbol in sample_data and 'data' in sample_data[symbol]:
                        data_df = sample_data[symbol]['data']
                        exploration['sample_data_info'] = {
                            'records_available': len(data_df),
                            'columns': list(data_df.columns),
                            'data_quality': 'GOOD' if len(data_df) > 0 else 'LIMITED',
                            'recent_timestamp': data_df['timestamp'].max().isoformat() if 'timestamp' in data_df.columns and not data_df.empty else None
                        }
                    else:
                        exploration['sample_data_info'] = {
                            'records_available': 0,
                            'data_quality': 'UNAVAILABLE'
                        }

                    # Symbol universe expansion suggestions
                    if symbol_category == 'equity':
                        exploration['expansion_suggestions'] = self._get_expansion_suggestions(symbol)

                exploration_results[symbol] = exploration

            except Exception as e:
                logger.error(f"Error exploring {symbol}: {e}")
                exploration_results[symbol] = {
                    'symbol': symbol,
                    'error': str(e),
                    'exploration_status': 'FAILED'
                }

        logger.info(f"Exploration completed for {len(exploration_results)} symbols")
        return exploration_results

    def discover_new_symbols(self, method: str = 'news',
                           min_mentions: int = 3) -> List[str]:
        """
        Discover new symbols for exploratory research.

        Args:
            method: 'news' (from news mentions), 'sector' (sector ETFs), 'related' (related symbols)
            min_mentions: Minimum mentions for news method

        Returns:
            List of discovered symbols
        """
        logger.info(f"Discovering new symbols using method: {method}")

        try:
            if method == 'news':
                # Use Polygon to discover symbols from news
                discovered = self.polygon_collector.discover_symbols_from_news(
                    lookback_days=7,
                    min_mentions=min_mentions
                )

                # Filter already known symbols
                new_symbols = [s for s in discovered if s not in self.discovered_symbols]

                logger.info(f"Discovered {len(new_symbols)} new symbols from news")
                return new_symbols

            elif method == 'sector':
                # Predefined sector ETFs and major indices
                sector_symbols = [
                    'SPY', 'QQQ', 'IWM',  # Market indices
                    'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU',  # Sector ETFs
                    'GLD', 'SLV', 'TLT', 'VIX'  # Alternative assets
                ]

                new_symbols = [s for s in sector_symbols if s not in self.discovered_symbols]
                logger.info(f"Suggested {len(new_symbols)} sector symbols")
                return new_symbols

            elif method == 'related':
                # Get symbols related to currently active symbols
                related_symbols = []

                # Simple related symbol logic (could be enhanced)
                for symbol in list(self.stats['active_symbols']):
                    if symbol == 'AAPL':
                        related_symbols.extend(['MSFT', 'GOOGL', 'AMZN', 'TSLA'])
                    elif symbol == 'SPY':
                        related_symbols.extend(['QQQ', 'IWM', 'VTI'])

                new_symbols = [s for s in related_symbols if s not in self.discovered_symbols]
                logger.info(f"Suggested {len(new_symbols)} related symbols")
                return new_symbols

            else:
                logger.error(f"Unknown discovery method: {method}")
                return []

        except Exception as e:
            logger.error(f"Error discovering symbols: {e}")
            return []

    def get_universe_snapshot(self) -> Dict[str, Any]:
        """
        Get snapshot of current symbol universe for exploratory research.

        Returns:
            Dict with universe statistics and symbol breakdown
        """
        try:
            # Categorize discovered symbols
            categorized = self.symbol_parser.categorize_symbols(list(self.discovered_symbols))

            universe_snapshot = {
                'total_discovered_symbols': len(self.discovered_symbols),
                'categories': {category: len(symbols) for category, symbols in categorized.items()},
                'category_breakdown': categorized,
                'most_recent_discoveries': list(self.discovered_symbols)[-10:] if len(self.discovered_symbols) > 10 else list(self.discovered_symbols),
                'exploration_stats': {
                    'total_explorations': self.stats['exploration_requests'],
                    'total_fetches': self.stats['fetch_requests'],
                    'active_symbols': len(self.stats['active_symbols']),
                    'data_points_collected': self.stats['data_points_collected']
                },
                'recommendations': self._get_universe_recommendations()
            }

            return universe_snapshot

        except Exception as e:
            logger.error(f"Error getting universe snapshot: {e}")
            return {}

    def _get_expansion_suggestions(self, symbol: str) -> List[str]:
        """Get suggestions for expanding symbol universe based on a given symbol"""
        suggestions = []

        # Industry peers (simplified)
        peers_map = {
            'AAPL': ['MSFT', 'GOOGL', 'AMZN'],
            'TSLA': ['GM', 'F', 'NIO'],
            'JPM': ['BAC', 'WFC', 'C'],
            'MSFT': ['AAPL', 'GOOGL', 'AMZN'],
        }

        base_symbol = symbol.upper()
        if base_symbol in peers_map:
            suggestions.extend(peers_map[base_symbol])

        # ETF suggestions for individual stocks
        if self.symbol_parser.get_symbol_category(symbol) == 'equity':
            suggestions.extend(['SPY', 'QQQ', 'XLK'])  # Broad market and tech

        return suggestions[:5]  # Limit suggestions

    def _get_universe_recommendations(self) -> List[str]:
        """Get recommendations for universe expansion"""
        recommendations = []

        if len(self.discovered_symbols) < 10:
            recommendations.append("Consider discovering more symbols using discover_new_symbols()")

        if 'equity' in self.symbol_parser.categorize_symbols(list(self.discovered_symbols)):
            recommendations.append("Add sector ETFs for sector rotation analysis")

        if self.stats['fetch_requests'] > self.stats['exploration_requests'] * 2:
            recommendations.append("Use explore() method to understand symbol characteristics before fetching")

        if not any('dtn_calculated' in self.symbol_metadata_cache.get(s, {}).get('category', '') for s in self.discovered_symbols):
            recommendations.append("Add DTN calculated indicators (*.Z symbols) for market sentiment")

        return recommendations

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics including exploratory research metrics"""
        base_stats = self.get_stats()

        enhanced_stats = {
            **base_stats,
            'discovered_symbols': list(self.discovered_symbols),
            'symbol_categories': self.symbol_parser.categorize_symbols(list(self.discovered_symbols)),
            'exploration_efficiency': {
                'explorations_per_symbol': self.stats['exploration_requests'] / max(len(self.discovered_symbols), 1),
                'fetches_per_symbol': self.stats['fetch_requests'] / max(len(self.discovered_symbols), 1),
                'data_points_per_fetch': self.stats['data_points_collected'] / max(self.stats['fetch_requests'], 1)
            },
            'universe_health': {
                'diversity_score': len(self.symbol_parser.categorize_symbols(list(self.discovered_symbols))),
                'discovery_rate': len(self.discovered_symbols) / max(self.stats['exploration_requests'], 1)
            }
        }

        return enhanced_stats