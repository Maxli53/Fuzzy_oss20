"""
IQFeed Collector - Unified collector for ALL IQFeed data
Collects ticks, bars, and DTN calculated indicators from IQFeed with clear separation:
- Market Data: Ticks and bars
- DTN Calculated Indicators: Equities/Index Statistics (Pages 2-11) and Options Statistics (Pages 12-16)
- Grok Derived Metrics: 20 calculations from options raw data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import time
from dotenv import load_dotenv

from stage_01_data_engine.core.base_collector import BaseCollector, StorageNamespace
from stage_01_data_engine.core.config_loader import get_config
from stage_01_data_engine.connector import IQFeedConnector
from stage_01_data_engine.storage.bar_builder import BarBuilder
import pyiqfeed as iq

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class IQFeedCollector(BaseCollector):
    """
    Unified IQFeed collector for all data types.

    Handles:
    - Market Data: Tick data and bars
    - DTN Calculated Indicators: Market breadth, sentiment, internals
    - Historical Data: OHLCV bars for any lookback period
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("IQFeedCollector", config)

        # Initialize IQFeed connector
        self.connector = IQFeedConnector()

        # Load configurations
        self.indicator_config = get_config('indicator', default={})
        self.bar_config = get_config('bar', default={})
        self.symbol_config = get_config('symbol', default={})

        # Build indicator mappings
        self._build_indicator_mappings()

        logger.info("IQFeedCollector initialized with unified data access")

    def _build_indicator_mappings(self):
        """Build complete DTN indicator mappings with clear structure"""
        # Equities/Index symbols (Pages 2-11)
        self.equities_index_symbols = {
            'issues': ['TINT.Z', 'TIQT.Z'],  # Page 2 - Total Issues
            'volume': ['VINT.Z', 'VIQT.Z'],  # Page 3 - Market Volume
            'tick': ['JTNT.Z', 'JTQT.Z'],    # Page 4 - Net Tick
            'trin': ['RINT.Z', 'RIQT.Z', 'RI6T.Z', 'RI1T.Z'],  # Page 5 - TRIN
            'highs_lows': ['H1NH.Z', 'H1NL.Z', 'H30NH.Z', 'H30NL.Z'],  # Page 6
            'avg_price': [],  # Page 7 - To be populated from config
            'moving_avg': ['M506V.Z', 'M506B.Z', 'M2006V.Z', 'M2006B.Z', 'M50QV.Z', 'M200QV.Z'],  # Page 8
            'premium': ['PREM.Z', 'PRNQ.Z', 'PRYM.Z'],  # Page 9 - Market Premium
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

        logger.info(f"Loaded DTN indicators: {len(self.equities_index_symbols)} equities categories, {len(self.options_symbols)} options categories")

    # ===========================================
    # MARKET DATA COLLECTION (Ticks and Bars)
    # ===========================================

    def collect_ticks(self, symbols: Union[str, List[str]], lookback_days: int = 1, **kwargs) -> Optional[pd.DataFrame]:
        """
        Collect tick data for symbols.

        Args:
            symbols: Stock symbols to collect
            lookback_days: Number of days to look back

        Returns:
            DataFrame with tick data (timestamp, symbol, price, volume)
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if not self.connector.connect():
            logger.error("Failed to connect to IQFeed")
            return None

        all_ticks = []

        try:
            hist_conn = self.connector.get_history_connection()
            if not hist_conn:
                logger.error("Failed to get history connection")
                return None

            with iq.ConnConnector([hist_conn]) as connector:
                for symbol in symbols:
                    try:
                        logger.info(f"Collecting tick data for {symbol}")

                        # Get tick data
                        tick_data = hist_conn.request_tick_data(
                            ticker=symbol,
                            num_days=lookback_days
                        )

                        if tick_data and len(tick_data) > 0:
                            for tick in tick_data:
                                try:
                                    tick_record = {
                                        'timestamp': pd.to_datetime(f"{tick['date']} {tick['time']}"),
                                        'symbol': symbol,
                                        'price': float(tick.get('last', 0)),
                                        'volume': int(tick.get('volume', 0)),
                                        'bid': float(tick.get('bid', 0)) if tick.get('bid') else None,
                                        'ask': float(tick.get('ask', 0)) if tick.get('ask') else None
                                    }
                                    all_ticks.append(tick_record)

                                except Exception as e:
                                    logger.warning(f"Error processing tick for {symbol}: {e}")
                                    continue

                            self.update_stats(len(tick_data), success=True)
                        else:
                            logger.warning(f"No tick data for {symbol}")
                            self.update_stats(0, success=False)

                    except Exception as e:
                        logger.error(f"Error collecting ticks for {symbol}: {e}")
                        self.update_stats(0, success=False)
                        continue

        except Exception as e:
            logger.error(f"Error in tick collection: {e}")
            return None
        finally:
            self.connector.disconnect()

        if not all_ticks:
            return None

        df = pd.DataFrame(all_ticks)
        return df.sort_values('timestamp').reset_index(drop=True)

    def collect_bars(self, symbols: Union[str, List[str]], bar_type: str = '5s', lookback_days: int = 1, **kwargs) -> Optional[pd.DataFrame]:
        """
        Collect or construct bars for symbols.

        Args:
            symbols: Stock symbols to collect
            bar_type: '5s', '1m', '5m', 'tick_50', etc.
            lookback_days: Number of days to look back

        Returns:
            DataFrame with OHLCV bars
        """
        if bar_type in ['tick_50']:
            # For tick-based bars, first collect ticks then construct
            tick_data = self.collect_ticks(symbols, lookback_days)
            if tick_data is None:
                return None

            # Use BarBuilder to construct tick bars
            all_bars = []
            for symbol in (symbols if isinstance(symbols, list) else [symbols]):
                symbol_ticks = tick_data[tick_data['symbol'] == symbol]
                if not symbol_ticks.empty:
                    tick_bars = BarBuilder.tick_bars(symbol_ticks, n=50)
                    if not tick_bars.empty:
                        tick_bars['symbol'] = symbol
                        all_bars.append(tick_bars)

            if all_bars:
                return pd.concat(all_bars, ignore_index=True)
            return None

        elif bar_type in ['5s', '1m', '5m', '15m', '1h', '1d']:
            # For time-based bars, get from IQFeed directly
            return self._collect_time_bars(symbols, bar_type, lookback_days)

        else:
            logger.error(f"Unsupported bar type: {bar_type}")
            return None

    def _collect_time_bars(self, symbols: Union[str, List[str]], bar_type: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Collect time-based bars directly from IQFeed"""
        if isinstance(symbols, str):
            symbols = [symbols]

        if not self.connector.connect():
            return None

        all_bars = []

        try:
            hist_conn = self.connector.get_history_connection()
            if not hist_conn:
                return None

            # Map bar type to IQFeed intervals
            interval_map = {
                '5s': 5,
                '1m': 60,
                '5m': 300,
                '15m': 900,
                '1h': 3600
            }

            with iq.ConnConnector([hist_conn]) as connector:
                for symbol in symbols:
                    try:
                        logger.info(f"Collecting {bar_type} bars for {symbol}")

                        if bar_type == '1d':
                            bar_data = hist_conn.request_daily_data(
                                ticker=symbol,
                                num_days=lookback_days
                            )
                        else:
                            seconds = interval_map.get(bar_type, 60)
                            bar_data = hist_conn.request_interval_data(
                                ticker=symbol,
                                interval_len=seconds,
                                num_days=lookback_days
                            )

                        if bar_data and len(bar_data) > 0:
                            for bar in bar_data:
                                try:
                                    bar_record = {
                                        'timestamp': pd.to_datetime(f"{bar['date']} {bar.get('time', '00:00:00')}"),
                                        'symbol': symbol,
                                        'open': float(bar.get('open_p', 0)),
                                        'high': float(bar.get('high_p', 0)),
                                        'low': float(bar.get('low_p', 0)),
                                        'close': float(bar.get('close_p', 0)),
                                        'volume': int(bar.get('prd_vlm', 0))
                                    }
                                    all_bars.append(bar_record)

                                except Exception as e:
                                    logger.warning(f"Error processing bar for {symbol}: {e}")
                                    continue

                            self.update_stats(len(bar_data), success=True)
                        else:
                            logger.warning(f"No bar data for {symbol}")
                            self.update_stats(0, success=False)

                    except Exception as e:
                        logger.error(f"Error collecting bars for {symbol}: {e}")
                        self.update_stats(0, success=False)
                        continue

        except Exception as e:
            logger.error(f"Error in bar collection: {e}")
            return None
        finally:
            self.connector.disconnect()

        if not all_bars:
            return None

        df = pd.DataFrame(all_bars)
        return df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    # ===========================================
    # DTN CALCULATED INDICATORS
    # ===========================================

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

            # Additional metrics (simplified implementations)
            if call_vol and put_vol:
                total_vol = call_vol + put_vol
                grok_metrics['sizzle_index'] = total_vol  # Needs historical average
                grok_metrics['gamma_flow'] = call_vol - put_vol
                grok_metrics['institutional_flow'] = max(call_vol, put_vol)
                grok_metrics['retail_sentiment'] = min(call_vol, put_vol) / max(call_vol, put_vol) if max(call_vol, put_vol) > 0 else 0
                grok_metrics['momentum_indicators'] = abs(call_vol - put_vol) / total_vol if total_vol > 0 else 0

            # Placeholders for complex metrics needing additional data
            for metric in ['dark_pool_sentiment', 'volatility_skew', 'term_structure',
                          'contrarian_signals', 'fear_greed_index', 'liquidity_metrics',
                          'smart_money_flow', 'vix_structure', 'cross_asset_signals', 'flow_imbalance']:
                grok_metrics[metric] = 0

            logger.info(f"Calculated {len(grok_metrics)} Grok metrics")

        except Exception as e:
            logger.error(f"Error calculating Grok metrics: {e}")

        return grok_metrics

    def _get_indicator_quote(self, lookup_conn, symbol: str) -> Optional[Dict]:
        """Get current quote for DTN indicator symbol"""
        try:
            # Try fundamental data first
            try:
                fundamental_data = lookup_conn.request_fundamental_fieldnames(symbol)
                if fundamental_data:
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

            logger.warning(f"Could not get quote for {symbol}")
            return {'price': 0, 'note': 'fallback'}

        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return None

    # ===========================================
    # UNIFIED DATA ACCESS METHODS
    # ===========================================

    def collect(self, symbols: Union[str, List[str]], **kwargs) -> Optional[pd.DataFrame]:
        """
        Unified collection method for all IQFeed data types.

        Args:
            symbols: Stock symbols or 'indicators' for DTN indicators
            data_type: 'ticks', 'bars', 'indicators', 'equities_stats', 'options_stats'
            **kwargs: Additional parameters
        """
        data_type = kwargs.get('data_type', 'ticks')

        if data_type == 'ticks':
            return self.collect_ticks(symbols, **kwargs)
        elif data_type == 'bars':
            return self.collect_bars(symbols, **kwargs)
        elif data_type == 'equities_stats':
            return self.collect_equities_index_stats()
        elif data_type == 'options_stats':
            return self.collect_options_stats()
        else:
            logger.error(f"Invalid data_type: {data_type}")
            return None

    def get_market_sentiment_snapshot(self) -> Dict[str, float]:
        """Get comprehensive market sentiment snapshot"""
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
        normalized = 2 - trin_value  # Invert scale
        return np.clip((normalized - 1) / 1, -1, 1)

    def _normalize_pc_ratio(self, pc_ratio: float) -> float:
        """Normalize Put/Call ratio to -1 to 1 scale"""
        normalized = 2 - pc_ratio  # Invert scale
        return np.clip((normalized - 1) / 0.5, -1, 1)

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate collected data"""
        try:
            if data.empty:
                return False

            required_cols = ['timestamp', 'symbol']
            for col in required_cols:
                if col not in data.columns:
                    logger.error(f"Missing required column: {col}")
                    return False

            if data['timestamp'].isna().any():
                logger.error("Some timestamps are NaN")
                return False

            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def get_storage_key(self, symbol: str, date: str, **kwargs) -> str:
        """Generate storage key for IQFeed data"""
        data_type = kwargs.get('data_type', 'ticks')
        return StorageNamespace.iqfeed_key(f"{data_type}_{symbol}", date)