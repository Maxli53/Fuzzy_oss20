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
from stage_01_data_engine.collectors.tick_collector import TickCollector
from stage_01_data_engine.collectors.dtn_indicators_collector import DTNIndicatorCollector
from stage_01_data_engine.storage.tick_store import TickStore

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
        """Initialize data engine with collectors and storage"""

        # Initialize configuration
        self.config = get_config_loader()
        if config_dir:
            self.config.config_dir = config_dir
            self.config.reload_configs()

        # Initialize collectors
        self.tick_collector = TickCollector()
        self.dtn_collector = DTNIndicatorCollector()

        # Initialize storage
        self.tick_store = TickStore()

        # Performance tracking
        self.stats = {
            'collections_today': 0,
            'data_points_collected': 0,
            'last_collection': None,
            'active_symbols': set(),
            'errors_today': 0
        }

        logger.info("DataEngine initialized successfully")

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

            # Collect tick data
            logger.info("Collecting tick data...")
            tick_data = self.tick_collector.collect(
                symbols,
                num_days=1,
                max_ticks=1000
            )
            if tick_data is not None:
                snapshot['ticks'] = tick_data
                logger.info(f"Collected {len(tick_data)} ticks")
            else:
                snapshot['ticks'] = pd.DataFrame()
                logger.warning("No tick data collected")

            # Collect key market indicators
            logger.info("Collecting market indicators...")
            key_indicators = [
                'NYSE_TICK', 'NYSE_TRIN', 'NYSE_ADD',
                'TOTAL_PC_RATIO', 'CALL_VOLUME_TOTAL', 'PUT_VOLUME_TOTAL'
            ]

            indicator_data = self.dtn_collector.collect(
                key_indicators,
                data_type='snapshot'
            )
            if indicator_data is not None:
                snapshot['indicators'] = indicator_data
                logger.info(f"Collected {len(indicator_data)} indicators")
            else:
                snapshot['indicators'] = pd.DataFrame()
                logger.warning("No indicator data collected")

            # Get sentiment snapshot
            logger.info("Calculating sentiment metrics...")
            sentiment_data = self.dtn_collector.get_market_sentiment_snapshot()
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

            # Collect historical indicators
            all_indicators = ['NYSE_TICK', 'NYSE_TRIN', 'NYSE_ADD', 'TOTAL_PC_RATIO']

            indicator_data = self.dtn_collector.collect(
                all_indicators,
                data_type='historical',
                lookback_periods=days
            )

            if indicator_data is not None:
                historical['indicators'] = indicator_data
                logger.info(f"Collected historical indicators: {len(indicator_data)} records")

            # For each symbol, try to get some tick data
            all_tick_data = []
            for symbol in symbols:
                tick_data = self.tick_collector.collect(
                    [symbol],
                    num_days=min(days, 5),  # Limit tick data collection
                    max_ticks=10000
                )
                if tick_data is not None:
                    all_tick_data.append(tick_data)

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
            sentiment = self.dtn_collector.get_market_sentiment_snapshot()

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
        Store market snapshot to persistent storage.

        Args:
            snapshot: Market snapshot data

        Returns:
            True if successful
        """
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            success = True

            # Store tick data
            if 'ticks' in snapshot and not snapshot['ticks'].empty:
                for symbol in snapshot['ticks']['symbol'].unique():
                    symbol_ticks = snapshot['ticks'][
                        snapshot['ticks']['symbol'] == symbol
                    ].copy()

                    stored = self.tick_store.store_ticks(
                        symbol=symbol,
                        date=today,
                        tick_df=symbol_ticks,
                        metadata={'source': 'data_engine_snapshot'},
                        overwrite=True
                    )
                    if not stored:
                        success = False

            # TODO: Store indicator data and sentiment data when we have those stores

            logger.info(f"Snapshot storage {'successful' if success else 'failed'}")
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