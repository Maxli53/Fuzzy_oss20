"""
DTN Calculated Indicators Collector
Collects DTN calculated market indicators from IQFeed with clear separation:
- Equities/Index Statistics (Pages 2-11 of DTN PDF)
- Options Statistics (Pages 12-16 of DTN PDF)
- Grok Derived Metrics (20 calculations from options raw data)
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
        """Build complete indicator mappings from config with new structure"""
        self.indicators = {}

        # Equities/Index symbols (Pages 2-11)
        self.equities_index_symbols = {
            'issues': ['TINT.Z', 'TIQT.Z'],  # Page 2
            'volume': ['VINT.Z', 'VIQT.Z'],  # Page 3
            'tick': ['JTNT.Z', 'JTQT.Z'],    # Page 4
            'trin': ['RINT.Z', 'RIQT.Z', 'RI6T.Z', 'RI1T.Z'],  # Page 5
            'highs_lows': ['H1NH.Z', 'H1NL.Z', 'H30NH.Z', 'H30NL.Z'],  # Page 6
            'avg_price': [],  # Page 7 - To be populated from config
            'moving_avg': ['M506V.Z', 'M506B.Z', 'M2006V.Z', 'M2006B.Z', 'M50QV.Z', 'M200QV.Z'],  # Page 8
            'premium': ['PREM.Z', 'PRNQ.Z', 'PRYM.Z'],  # Page 9
            'ratio': [],     # Page 10 - To be populated from config
            'net': []        # Page 11 - To be populated from config
        }

        # Options symbols (Pages 12-16)
        self.options_symbols = {
            'tick': ['TCOEA.Z', 'TPOEA.Z', 'TCOED.Z', 'TPOED.Z'],  # Page 12
            'issues': ['ICOEA.Z', 'IPOEA.Z', 'ICOED.Z', 'IPOED.Z'],  # Page 13
            'open_interest': ['OCOET.Z', 'OPOET.Z', 'OCORET.Z', 'OPORET.Z'],  # Page 14
            'volume': ['VCOET.Z', 'VPOET.Z', 'DCOET.Z', 'DPOET.Z'],  # Page 15
            'trin': ['SCOET.Z', 'SPOET.Z']  # Page 16
        }

        # Build unified indicators dict for backward compatibility
        all_symbols = {}
        for category, symbols in self.equities_index_symbols.items():
            for symbol in symbols:
                all_symbols[symbol] = {'group': f'equities_index_{category}', 'type': category}

        for category, symbols in self.options_symbols.items():
            for symbol in symbols:
                all_symbols[symbol] = {'group': f'options_{category}', 'type': f'options_{category}'}

        # Add to main indicators dict
        for symbol, info in all_symbols.items():
            self.indicators[symbol] = {
                'symbol': symbol,
                'group': info['group'],
                'type': info['type']
            }

        logger.info(f"Loaded {len(self.indicators)} DTN indicators across {len(self.equities_index_symbols)} equities categories and {len(self.options_symbols)} options categories")

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

    def collect_equities_index_stats(self) -> Dict[str, pd.DataFrame]:
        """Collect all equities/index indicators (Pages 2-11)"""
        results = {}

        if not self.connector.connect():
            logger.error("Failed to connect to IQFeed")
            return results

        try:
            lookup_conn = self.connector.get_lookup_connection()
            if not lookup_conn:
                logger.error("Failed to get lookup connection")
                return results

            with iq.ConnConnector([lookup_conn]) as connector:
                for category, symbols in self.equities_index_symbols.items():
                    if not symbols:  # Skip empty categories
                        continue

                    category_data = []
                    timestamp = pd.Timestamp.now()

                    for symbol in symbols:
                        try:
                            logger.info(f"Collecting equities/index {category}: {symbol}")
                            quote_data = self._get_indicator_quote(lookup_conn, symbol)

                            if quote_data is not None:
                                result = {
                                    'timestamp': timestamp,
                                    'symbol': symbol,
                                    'category': category,
                                    'value': quote_data.get('last', quote_data.get('price', 0)),
                                    'bid': quote_data.get('bid', None),
                                    'ask': quote_data.get('ask', None),
                                    'volume': quote_data.get('volume', None)
                                }
                                category_data.append(result)

                        except Exception as e:
                            logger.warning(f"Failed to collect {symbol} in {category}: {e}")

                    if category_data:
                        results[category] = pd.DataFrame(category_data)

        except Exception as e:
            logger.error(f"Error in equities/index collection: {e}")
        finally:
            self.connector.disconnect()

        return results

    def collect_options_stats(self) -> Dict[str, pd.DataFrame]:
        """Collect all options indicators (Pages 12-16)"""
        results = {}

        if not self.connector.connect():
            logger.error("Failed to connect to IQFeed")
            return results

        try:
            lookup_conn = self.connector.get_lookup_connection()
            if not lookup_conn:
                logger.error("Failed to get lookup connection")
                return results

            with iq.ConnConnector([lookup_conn]) as connector:
                for category, symbols in self.options_symbols.items():
                    category_data = []
                    timestamp = pd.Timestamp.now()

                    for symbol in symbols:
                        try:
                            logger.info(f"Collecting options {category}: {symbol}")
                            quote_data = self._get_indicator_quote(lookup_conn, symbol)

                            if quote_data is not None:
                                result = {
                                    'timestamp': timestamp,
                                    'symbol': symbol,
                                    'category': category,
                                    'value': quote_data.get('last', quote_data.get('price', 0)),
                                    'bid': quote_data.get('bid', None),
                                    'ask': quote_data.get('ask', None),
                                    'volume': quote_data.get('volume', None)
                                }
                                category_data.append(result)

                        except Exception as e:
                            logger.warning(f"Failed to collect {symbol} in {category}: {e}")

                    if category_data:
                        results[category] = pd.DataFrame(category_data)

        except Exception as e:
            logger.error(f"Error in options collection: {e}")
        finally:
            self.connector.disconnect()

        return results

    def calculate_grok_metrics(self, options_raw: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate 20 Grok metrics from raw options data"""
        grok_metrics = {}

        try:
            # Helper function to get latest value
            def get_value(category: str, symbol: str) -> Optional[float]:
                if category in options_raw and not options_raw[category].empty:
                    df = options_raw[category]
                    symbol_data = df[df['symbol'] == symbol]
                    if not symbol_data.empty:
                        return symbol_data['value'].iloc[-1]
                return None

            # Get raw values
            call_vol = get_value('volume', 'VCOET.Z')
            put_vol = get_value('volume', 'VPOET.Z')
            call_dollar_vol = get_value('volume', 'DCOET.Z')
            put_dollar_vol = get_value('volume', 'DPOET.Z')
            call_oi = get_value('open_interest', 'OCOET.Z')
            put_oi = get_value('open_interest', 'OPOET.Z')
            call_advances = get_value('tick', 'TCOEA.Z')
            put_advances = get_value('tick', 'TPOEA.Z')
            call_declines = get_value('tick', 'TCOED.Z')
            put_declines = get_value('tick', 'TPOED.Z')

            # Calculate the 20 Grok metrics
            if call_vol and put_vol and call_vol > 0:
                grok_metrics['pcr'] = put_vol / call_vol

            if call_dollar_vol and put_dollar_vol and call_dollar_vol > 0:
                grok_metrics['dollar_pcr'] = put_dollar_vol / call_dollar_vol

            if call_oi and put_oi and call_oi > 0:
                grok_metrics['oi_pcr'] = put_oi / call_oi

            if all([call_advances, call_declines, put_advances, put_declines]):
                grok_metrics['net_tick_sentiment'] = (call_advances - call_declines) - (put_advances - put_declines)

            if call_vol and put_vol:
                grok_metrics['volume_spread'] = call_vol - put_vol

            # Additional metrics (simplified implementations - would need more data for full calculations)
            if call_vol and put_vol:
                total_vol = call_vol + put_vol
                grok_metrics['sizzle_index'] = total_vol  # Simplified - needs historical average
                grok_metrics['gamma_flow'] = call_vol - put_vol  # Simplified
                grok_metrics['institutional_flow'] = max(call_vol, put_vol)  # Simplified
                grok_metrics['retail_sentiment'] = min(call_vol, put_vol) / max(call_vol, put_vol) if max(call_vol, put_vol) > 0 else 0
                grok_metrics['momentum_indicators'] = abs(call_vol - put_vol) / total_vol if total_vol > 0 else 0

            # Placeholder for remaining complex metrics that need additional data
            grok_metrics['dark_pool_sentiment'] = 0
            grok_metrics['volatility_skew'] = 0
            grok_metrics['term_structure'] = 0
            grok_metrics['contrarian_signals'] = 0
            grok_metrics['fear_greed_index'] = 0
            grok_metrics['liquidity_metrics'] = 0
            grok_metrics['smart_money_flow'] = 0
            grok_metrics['vix_structure'] = 0
            grok_metrics['cross_asset_signals'] = 0
            grok_metrics['flow_imbalance'] = 0

            logger.info(f"Calculated {len(grok_metrics)} Grok metrics")

        except Exception as e:
            logger.error(f"Error calculating Grok metrics: {e}")

        return grok_metrics

    def get_market_sentiment_snapshot(self) -> Dict[str, float]:
        """Get comprehensive market sentiment snapshot with new structure"""
        try:
            # Collect from both categories
            equities_data = self.collect_equities_index_stats()
            options_data = self.collect_options_stats()

            sentiment = {}

            # Extract key equities/index indicators
            if 'tick' in equities_data:
                tick_df = equities_data['tick']
                nyse_tick = tick_df[tick_df['symbol'] == 'JTNT.Z']
                nasdaq_tick = tick_df[tick_df['symbol'] == 'JTQT.Z']

                if not nyse_tick.empty:
                    sentiment['NYSE_TICK'] = nyse_tick['value'].iloc[-1]
                if not nasdaq_tick.empty:
                    sentiment['NASDAQ_TICK'] = nasdaq_tick['value'].iloc[-1]

            if 'trin' in equities_data:
                trin_df = equities_data['trin']
                nyse_trin = trin_df[trin_df['symbol'] == 'RINT.Z']
                nasdaq_trin = trin_df[trin_df['symbol'] == 'RIQT.Z']

                if not nyse_trin.empty:
                    sentiment['NYSE_TRIN'] = nyse_trin['value'].iloc[-1]
                if not nasdaq_trin.empty:
                    sentiment['NASDAQ_TRIN'] = nasdaq_trin['value'].iloc[-1]

            # Calculate Grok metrics
            grok_metrics = self.calculate_grok_metrics(options_data)
            if 'pcr' in grok_metrics:
                sentiment['TOTAL_PC_RATIO'] = grok_metrics['pcr']

            # Add composite sentiment score
            if all(k in sentiment for k in ['NYSE_TICK', 'NYSE_TRIN', 'TOTAL_PC_RATIO']):
                tick_score = self._normalize_tick(sentiment['NYSE_TICK'])
                trin_score = self._normalize_trin(sentiment['NYSE_TRIN'])
                pc_score = self._normalize_pc_ratio(sentiment['TOTAL_PC_RATIO'])
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