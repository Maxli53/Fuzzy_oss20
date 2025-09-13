"""
DTN Indicators Collector
Collects thousands of calculated market indicators from DTN/IQFeed
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import logging

from stage_01_data_engine.core.base_collector import BaseCollector, StorageNamespace
from stage_01_data_engine.core.config_loader import get_config
from stage_01_data_engine.connector import IQFeedConnector
import pyiqfeed as iq

logger = logging.getLogger(__name__)

class DTNIndicatorCollector(BaseCollector):
    """
    Professional DTN calculated indicators collector.

    Collects market sentiment indicators including:
    - Market breadth (TICK, TRIN, ADD, VOLD)
    - Options sentiment (Put/Call ratios, volume flow)
    - New highs/lows
    - Moving average breadth
    - Market premium indicators
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("DTNIndicatorCollector", config)

        # Initialize IQFeed connector
        self.connector = IQFeedConnector()

        # Load indicator mappings from config
        self.indicator_config = get_config('indicator', default={})
        self.collection_config = get_config('stream', 'indicator_collection', default={})

        # Create indicator symbol mappings
        self._build_indicator_mappings()

        logger.info("DTNIndicatorCollector initialized")

    def _build_indicator_mappings(self):
        """Build complete indicator mappings from config"""
        self.indicators = {}

        # Load all indicator groups from config
        indicator_groups = [
            'breadth_indicators',
            'trin_indicators',
            'highs_lows',
            'moving_average_breadth',
            'options_sentiment',
            'options_oi',
            'options_flow',
            'market_premium',
            'volatility'
        ]

        for group in indicator_groups:
            group_indicators = self.indicator_config.get(group, {})
            for name, symbol in group_indicators.items():
                self.indicators[name] = {
                    'symbol': symbol,
                    'group': group,
                    'type': self._get_indicator_type(symbol)
                }

        logger.info(f"Loaded {len(self.indicators)} DTN indicators")

    def _get_indicator_type(self, symbol: str) -> str:
        """Determine indicator type from symbol pattern"""
        if symbol.startswith('JT') or symbol.startswith('LI'):
            return 'tick'
        elif symbol.startswith('RI'):
            return 'trin'
        elif symbol.startswith('H'):
            return 'highs_lows'
        elif symbol.startswith('M'):
            return 'moving_average'
        elif symbol.startswith('V') and 'O' in symbol:
            return 'options_volume'
        elif symbol.startswith('O') and 'O' in symbol:
            return 'options_oi'
        elif symbol.startswith('T') and 'O' in symbol:
            return 'options_tick'
        elif symbol.startswith('PR'):
            return 'premium'
        else:
            return 'other'

    def collect(self, indicators: Union[str, List[str]], **kwargs) -> Optional[pd.DataFrame]:
        """
        Collect DTN indicators.

        Args:
            indicators: Indicator name(s) or 'all' for all indicators
            data_type: 'snapshot' for current values, 'historical' for time series
            lookback_periods: Number of periods for historical data

        Returns:
            DataFrame with indicator data
        """
        if isinstance(indicators, str):
            if indicators.lower() == 'all':
                indicators = list(self.indicators.keys())
            else:
                indicators = [indicators]

        data_type = kwargs.get('data_type', 'snapshot')
        lookback_periods = kwargs.get('lookback_periods', 1)

        # Validate indicators exist
        valid_indicators = []
        for indicator in indicators:
            if indicator in self.indicators:
                valid_indicators.append(indicator)
            else:
                logger.warning(f"Unknown indicator: {indicator}")

        if not valid_indicators:
            logger.error("No valid indicators specified")
            return None

        if data_type == 'snapshot':
            return self._collect_snapshot(valid_indicators)
        elif data_type == 'historical':
            return self._collect_historical(valid_indicators, lookback_periods)
        else:
            logger.error(f"Invalid data_type: {data_type}")
            return None

    def _collect_snapshot(self, indicators: List[str]) -> Optional[pd.DataFrame]:
        """Collect current snapshot of indicators"""
        if not self.connector.connect():
            logger.error("Failed to connect to IQFeed")
            return None

        try:
            # For snapshot, we'll use lookup connection to get current quotes
            lookup_conn = self.connector.get_lookup_connection()
            if not lookup_conn:
                logger.error("Failed to get lookup connection")
                return None

            results = []
            timestamp = pd.Timestamp.now()

            with iq.ConnConnector([lookup_conn]) as connector:
                for indicator_name in indicators:
                    try:
                        indicator_info = self.indicators[indicator_name]
                        symbol = indicator_info['symbol']

                        logger.info(f"Collecting {indicator_name} ({symbol})")

                        # Get current quote/value
                        # Note: DTN indicators might need special handling
                        quote_data = self._get_indicator_quote(lookup_conn, symbol)

                        if quote_data is not None:
                            result = {
                                'timestamp': timestamp,
                                'indicator': indicator_name,
                                'symbol': symbol,
                                'group': indicator_info['group'],
                                'type': indicator_info['type'],
                                'value': quote_data.get('last', quote_data.get('price', 0)),
                                'bid': quote_data.get('bid', None),
                                'ask': quote_data.get('ask', None),
                                'change': quote_data.get('change', None),
                                'volume': quote_data.get('volume', None)
                            }
                            results.append(result)

                            self.update_stats(1, success=True)
                        else:
                            logger.warning(f"No data for {indicator_name}")
                            self.update_stats(0, success=False)

                    except Exception as e:
                        logger.error(f"Error collecting {indicator_name}: {e}")
                        self.update_stats(0, success=False)
                        continue

            if not results:
                return None

            df = pd.DataFrame(results)
            return self._post_process_indicators(df)

        except Exception as e:
            logger.error(f"Error in snapshot collection: {e}")
            return None
        finally:
            self.connector.disconnect()

    def _collect_historical(self, indicators: List[str], lookback_periods: int) -> Optional[pd.DataFrame]:
        """Collect historical time series for indicators"""
        if not self.connector.connect():
            logger.error("Failed to connect to IQFeed")
            return None

        try:
            hist_conn = self.connector.get_history_connection()
            if not hist_conn:
                logger.error("Failed to get history connection")
                return None

            all_results = []

            with iq.ConnConnector([hist_conn]) as connector:
                for indicator_name in indicators:
                    try:
                        indicator_info = self.indicators[indicator_name]
                        symbol = indicator_info['symbol']

                        logger.info(f"Collecting historical {indicator_name} ({symbol})")

                        # Get historical data (daily intervals for most indicators)
                        hist_data = hist_conn.request_daily_data(
                            ticker=symbol,
                            num_days=lookback_periods
                        )

                        if hist_data and len(hist_data) > 0:
                            for bar in hist_data:
                                try:
                                    result = {
                                        'timestamp': pd.to_datetime(bar['date']),
                                        'indicator': indicator_name,
                                        'symbol': symbol,
                                        'group': indicator_info['group'],
                                        'type': indicator_info['type'],
                                        'open': float(bar.get('open_p', 0)),
                                        'high': float(bar.get('high_p', 0)),
                                        'low': float(bar.get('low_p', 0)),
                                        'close': float(bar.get('close_p', 0)),
                                        'volume': int(bar.get('prd_vlm', 0))
                                    }
                                    # Use close as primary value
                                    result['value'] = result['close']
                                    all_results.append(result)

                                except Exception as e:
                                    logger.warning(f"Error processing bar for {indicator_name}: {e}")
                                    continue

                            self.update_stats(len(hist_data), success=True)
                        else:
                            logger.warning(f"No historical data for {indicator_name}")
                            self.update_stats(0, success=False)

                    except Exception as e:
                        logger.error(f"Error collecting historical {indicator_name}: {e}")
                        self.update_stats(0, success=False)
                        continue

            if not all_results:
                return None

            df = pd.DataFrame(all_results)
            df = df.sort_values(['indicator', 'timestamp']).reset_index(drop=True)

            return self._post_process_indicators(df)

        except Exception as e:
            logger.error(f"Error in historical collection: {e}")
            return None
        finally:
            self.connector.disconnect()

    def _get_indicator_quote(self, lookup_conn, symbol: str) -> Optional[Dict]:
        """Get current quote for DTN indicator symbol"""
        try:
            # Try to get fundamental data (some DTN indicators available here)
            try:
                fundamental_data = lookup_conn.request_fundamental_fieldnames(symbol)
                if fundamental_data:
                    # Extract relevant data
                    return {'price': fundamental_data.get('Price', 0)}
            except:
                pass

            # Try regular quote lookup
            try:
                quote_data = lookup_conn.request_current_update_fieldnames(symbol)
                if quote_data:
                    return {
                        'last': quote_data.get('Last', 0),
                        'bid': quote_data.get('Bid', None),
                        'ask': quote_data.get('Ask', None),
                        'volume': quote_data.get('Volume', None),
                        'change': quote_data.get('Change', None)
                    }
            except:
                pass

            # If direct quote fails, try to get it as a regular symbol
            logger.warning(f"Could not get quote for {symbol}, using fallback")
            return {'price': 0, 'note': 'fallback'}

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None

    def _post_process_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process indicator data with calculations and validations"""
        if df.empty:
            return df

        # Add derived metrics
        df = self._add_derived_metrics(df)

        # Add market regime flags
        df = self._add_regime_flags(df)

        # Add alert flags based on thresholds
        df = self._add_alert_flags(df)

        # Validate data
        if not self.validate(df):
            logger.warning("Indicator data validation failed")

        return df

    def _add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived metrics and calculations"""
        try:
            # Add z-scores for indicators (if we have historical context)
            if 'value' in df.columns:
                for indicator in df['indicator'].unique():
                    mask = df['indicator'] == indicator
                    values = df.loc[mask, 'value']

                    if len(values) > 1:
                        mean_val = values.mean()
                        std_val = values.std()
                        if std_val > 0:
                            df.loc[mask, 'z_score'] = (values - mean_val) / std_val
                        else:
                            df.loc[mask, 'z_score'] = 0

            # Add percentile rankings
            if 'value' in df.columns:
                for indicator in df['indicator'].unique():
                    mask = df['indicator'] == indicator
                    values = df.loc[mask, 'value']

                    if len(values) > 1:
                        df.loc[mask, 'percentile'] = values.rank(pct=True) * 100

            return df

        except Exception as e:
            logger.error(f"Error adding derived metrics: {e}")
            return df

    def _add_regime_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime classification flags"""
        try:
            # Market regime flags based on indicator values
            regime_flags = []

            for _, row in df.iterrows():
                indicator = row['indicator']
                value = row.get('value', 0)
                flags = []

                # TICK-based regime detection
                if 'TICK' in indicator:
                    if value > 1000:
                        flags.append('BULLISH_MOMENTUM')
                    elif value < -1000:
                        flags.append('BEARISH_MOMENTUM')
                    elif abs(value) < 200:
                        flags.append('NEUTRAL_MOMENTUM')

                # TRIN-based regime detection
                elif 'TRIN' in indicator:
                    if value > 2.0:
                        flags.append('OVERSOLD')
                    elif value < 0.5:
                        flags.append('OVERBOUGHT')
                    elif 0.8 <= value <= 1.2:
                        flags.append('BALANCED')

                # Put/Call ratio regime
                elif 'PC_RATIO' in indicator:
                    if value > 1.2:
                        flags.append('HIGH_FEAR')
                    elif value < 0.7:
                        flags.append('HIGH_GREED')

                # New highs/lows regime
                elif 'HIGH' in indicator or 'LOW' in indicator:
                    if 'HIGH' in indicator and value > 100:
                        flags.append('STRONG_BREADTH')
                    elif 'LOW' in indicator and value > 100:
                        flags.append('WEAK_BREADTH')

                regime_flags.append(', '.join(flags) if flags else 'NORMAL')

            df['regime_flags'] = regime_flags
            return df

        except Exception as e:
            logger.error(f"Error adding regime flags: {e}")
            return df

    def _add_alert_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add alert flags based on configured thresholds"""
        try:
            alert_thresholds = self.indicator_config.get('alert_thresholds', {})

            alerts = []

            for _, row in df.iterrows():
                indicator = row['indicator']
                value = row.get('value', 0)
                alert_list = []

                # Check if this indicator has configured thresholds
                for threshold_key, thresholds in alert_thresholds.items():
                    if threshold_key in indicator:
                        for threshold_name, threshold_value in thresholds.items():
                            if isinstance(threshold_value, (int, float)):
                                if 'positive' in threshold_name and value >= threshold_value:
                                    alert_list.append(f"ALERT_{threshold_name.upper()}")
                                elif 'negative' in threshold_name and value <= threshold_value:
                                    alert_list.append(f"ALERT_{threshold_name.upper()}")
                                elif 'oversold' in threshold_name and value >= threshold_value:
                                    alert_list.append("ALERT_OVERSOLD")
                                elif 'overbought' in threshold_name and value <= threshold_value:
                                    alert_list.append("ALERT_OVERBOUGHT")

                alerts.append(', '.join(alert_list) if alert_list else 'NORMAL')

            df['alerts'] = alerts
            return df

        except Exception as e:
            logger.error(f"Error adding alert flags: {e}")
            return df

    def get_market_sentiment_snapshot(self) -> Dict[str, float]:
        """Get quick market sentiment snapshot from key indicators"""
        key_indicators = [
            'NYSE_TICK',
            'NYSE_TRIN',
            'TOTAL_PC_RATIO',
            'NYSE_ADD'
        ]

        try:
            df = self.collect(key_indicators, data_type='snapshot')
            if df is None or df.empty:
                return {}

            sentiment = {}
            for _, row in df.iterrows():
                sentiment[row['indicator']] = row.get('value', 0)

            # Calculate composite sentiment score
            tick_score = self._normalize_tick(sentiment.get('NYSE_TICK', 0))
            trin_score = self._normalize_trin(sentiment.get('NYSE_TRIN', 1.0))
            pc_score = self._normalize_pc_ratio(sentiment.get('TOTAL_PC_RATIO', 1.0))

            sentiment['COMPOSITE_SENTIMENT'] = np.mean([tick_score, trin_score, pc_score])

            return sentiment

        except Exception as e:
            logger.error(f"Error getting sentiment snapshot: {e}")
            return {}

    def _normalize_tick(self, tick_value: float) -> float:
        """Normalize TICK to -1 to 1 scale"""
        return np.clip(tick_value / 2000, -1, 1)

    def _normalize_trin(self, trin_value: float) -> float:
        """Normalize TRIN to -1 to 1 scale (inverted)"""
        if trin_value <= 0:
            return 0
        # TRIN > 1 = bearish, TRIN < 1 = bullish
        normalized = 2 - trin_value  # Invert scale
        return np.clip((normalized - 1) / 1, -1, 1)

    def _normalize_pc_ratio(self, pc_ratio: float) -> float:
        """Normalize Put/Call ratio to -1 to 1 scale"""
        # P/C > 1 = bearish, P/C < 1 = bullish
        normalized = 2 - pc_ratio  # Invert scale
        return np.clip((normalized - 1) / 0.5, -1, 1)

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate indicator data"""
        try:
            if data.empty:
                return False

            # Check required columns
            required_cols = ['timestamp', 'indicator', 'value']
            for col in required_cols:
                if col not in data.columns:
                    logger.error(f"Missing required column: {col}")
                    return False

            # Check for reasonable values
            if data['value'].isna().all():
                logger.error("All indicator values are NaN")
                return False

            # Check timestamp validity
            if data['timestamp'].isna().any():
                logger.error("Some timestamps are NaN")
                return False

            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def get_storage_key(self, symbol: str, date: str, **kwargs) -> str:
        """Generate storage key for indicator data"""
        indicator_group = kwargs.get('group', 'general')
        return StorageNamespace.indicator_key(f"{indicator_group}_{symbol}", date)

    def list_available_indicators(self) -> Dict[str, Dict]:
        """List all available indicators with metadata"""
        return self.indicators

    def get_indicator_groups(self) -> List[str]:
        """Get list of indicator groups"""
        groups = set()
        for indicator_info in self.indicators.values():
            groups.add(indicator_info['group'])
        return sorted(list(groups))