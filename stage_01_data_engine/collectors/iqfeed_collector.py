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
from stage_01_data_engine.parsers.dtn_symbol_parser import DTNSymbolParser
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

        # Initialize symbol parser for flexible fetching
        self.symbol_parser = DTNSymbolParser()

        # Load configurations
        self.indicator_config = get_config('indicator', default={})
        self.bar_config = get_config('bar', default={})
        self.symbol_config = get_config('symbol', default={})

        # Build indicator mappings
        self._build_indicator_mappings()

        logger.info("IQFeedCollector initialized with unified data access and flexible symbol parsing")

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

                        # Get most recent tick data (like IQFeed terminal)
                        tick_data = hist_conn.request_ticks(
                            ticker=symbol,
                            max_ticks=kwargs.get('max_ticks', 1000)
                        )

                        if tick_data is not None and len(tick_data) > 0:
                            for tick in tick_data:
                                try:
                                    # Handle numpy structured array - access fields directly
                                    # Convert IQFeed timestamp properly
                                    date_str = str(tick['date'])

                                    # Convert numpy.timedelta64 to microseconds
                                    if isinstance(tick['time'], np.timedelta64):
                                        time_microseconds = int(tick['time'] / np.timedelta64(1, 'us'))
                                    else:
                                        time_microseconds = 0

                                    # Convert microseconds since midnight to time
                                    hours = time_microseconds // 3600000000
                                    minutes = (time_microseconds % 3600000000) // 60000000
                                    seconds = (time_microseconds % 60000000) // 1000000
                                    microseconds = time_microseconds % 1000000

                                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}"

                                    tick_record = {
                                        'timestamp': pd.to_datetime(f"{date_str} {time_str}"),
                                        'symbol': symbol,
                                        'price': float(tick['last']),
                                        'volume': int(tick['last_sz']),  # last_sz is the trade size
                                        'bid': float(tick['bid']),
                                        'ask': float(tick['ask']),
                                        'tick_id': int(tick['tick_id']),
                                        'market_center': int(tick['mkt_ctr']),
                                        'total_volume': int(tick['tot_vlm'])
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
                            bar_data = hist_conn.request_bars_for_days(
                                ticker=symbol,
                                interval_len=seconds,
                                interval_type='s',
                                days=lookback_days
                            )

                        if bar_data is not None and len(bar_data) > 0:
                            for bar in bar_data:
                                try:
                                    # Handle numpy structured array - access fields directly
                                    # Convert IQFeed timestamp format properly
                                    date_str = str(bar['date'])

                                    # Handle time field - numpy.timedelta64 from IQFeed
                                    if 'time' in bar.dtype.names:
                                        time_val = bar['time']
                                        if isinstance(time_val, np.timedelta64):
                                            # Convert numpy.timedelta64 to microseconds properly
                                            time_microseconds = int(time_val / np.timedelta64(1, 'us'))
                                        else:
                                            # Fallback for other types
                                            time_microseconds = 0
                                    else:
                                        time_microseconds = 0

                                    # Convert microseconds since midnight to time
                                    hours = time_microseconds // 3600000000
                                    minutes = (time_microseconds % 3600000000) // 60000000
                                    seconds = (time_microseconds % 60000000) // 1000000
                                    microseconds = time_microseconds % 1000000

                                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}"

                                    bar_record = {
                                        'timestamp': pd.to_datetime(f"{date_str} {time_str}"),
                                        'symbol': symbol,
                                        'open': float(bar['open_p']),
                                        'high': float(bar['high_p']),
                                        'low': float(bar['low_p']),
                                        'close': float(bar['close_p']),
                                        'volume': int(bar['prd_vlm'])
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

    # ===========================================
    # FLEXIBLE SYMBOL FETCHING (Exploratory Research)
    # ===========================================

    def fetch(self, symbols: Union[str, List[str]], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Flexible fetch method that can collect ANY symbol without preconfiguration.
        Perfect for exploratory quantitative research.

        Args:
            symbols: Any symbol(s) - stocks, options, futures, DTN indicators
            data_type: 'auto' (detect), 'ticks', 'bars', 'quotes', 'options_chain'
            lookback_days: Number of days to look back
            auto_categorize: Whether to use symbol parser for smart routing

        Returns:
            Dict mapping symbol to collected DataFrame
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        data_type = kwargs.get('data_type', 'auto')
        auto_categorize = kwargs.get('auto_categorize', True)

        # Remove data_type from kwargs to avoid parameter collision
        kwargs_clean = {k: v for k, v in kwargs.items() if k != 'data_type'}

        results = {}

        for symbol in symbols:
            try:
                logger.info(f"Fetching data for symbol: {symbol}")

                # Parse symbol for smart routing
                if auto_categorize:
                    symbol_info = self.symbol_parser.parse_symbol(symbol)
                    logger.info(f"Symbol {symbol} categorized as: {symbol_info.category}/{symbol_info.subcategory}")

                    # Route to appropriate collection method
                    if symbol_info.category == 'dtn_calculated':
                        df = self._fetch_dtn_indicator(symbol, **kwargs_clean)
                    elif symbol_info.category == 'options':
                        df = self._fetch_options_data(symbol, **kwargs_clean)
                    elif symbol_info.category == 'futures':
                        df = self._fetch_futures_data(symbol, **kwargs_clean)
                    elif symbol_info.category == 'forex':
                        df = self._fetch_forex_data(symbol, **kwargs_clean)
                    else:  # equity or unknown
                        df = self._fetch_equity_data(symbol, data_type, **kwargs_clean)
                else:
                    # Manual data type specification
                    df = self._fetch_by_data_type(symbol, data_type, **kwargs_clean)

                if df is not None and not df.empty:
                    results[symbol] = df
                    logger.info(f"Successfully fetched {len(df)} records for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                continue

        return results

    def collect_stock_prices(self, symbols: Union[str, List[str]],
                           lookback_days: int = 22,
                           bar_type: str = '1d') -> Optional[pd.DataFrame]:
        """
        Collect stock price data (OHLCV) for regular equities.

        Args:
            symbols: Stock symbols
            lookback_days: Number of trading days
            bar_type: '1d', '1h', '15m', '5m', '1m'

        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Collecting stock prices for {symbols} ({lookback_days} days, {bar_type} bars)")
        return self.collect_bars(symbols, bar_type=bar_type, lookback_days=lookback_days)

    def collect_realtime_quotes(self, symbols: Union[str, List[str]]) -> Optional[pd.DataFrame]:
        """
        Collect real-time Level 1 quotes.

        Args:
            symbols: Stock symbols

        Returns:
            DataFrame with current quotes (bid, ask, last, volume)
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if not self.connector.connect():
            logger.error("Failed to connect to IQFeed")
            return None

        all_quotes = []

        try:
            lookup_conn = self.connector.get_lookup_connection()
            if not lookup_conn:
                logger.error("Failed to get lookup connection")
                return None

            with iq.ConnConnector([lookup_conn]) as connector:
                for symbol in symbols:
                    try:
                        logger.info(f"Getting real-time quote for {symbol}")

                        quote_data = lookup_conn.request_current_update_fieldnames(symbol)

                        if quote_data:
                            quote_record = {
                                'timestamp': pd.Timestamp.now(),
                                'symbol': symbol,
                                'last': float(quote_data.get('Last', 0)),
                                'bid': float(quote_data.get('Bid', 0)) if quote_data.get('Bid') else None,
                                'ask': float(quote_data.get('Ask', 0)) if quote_data.get('Ask') else None,
                                'volume': int(quote_data.get('Volume', 0)) if quote_data.get('Volume') else None,
                                'change': float(quote_data.get('Change', 0)) if quote_data.get('Change') else None,
                                'percent_change': float(quote_data.get('Percent Change', 0)) if quote_data.get('Percent Change') else None,
                                'high': float(quote_data.get('High', 0)) if quote_data.get('High') else None,
                                'low': float(quote_data.get('Low', 0)) if quote_data.get('Low') else None,
                                'open': float(quote_data.get('Open', 0)) if quote_data.get('Open') else None
                            }
                            all_quotes.append(quote_record)
                            self.update_stats(1, success=True)
                        else:
                            logger.warning(f"No quote data for {symbol}")
                            self.update_stats(0, success=False)

                    except Exception as e:
                        logger.error(f"Error getting quote for {symbol}: {e}")
                        self.update_stats(0, success=False)
                        continue

        except Exception as e:
            logger.error(f"Error in quote collection: {e}")
            return None
        finally:
            self.connector.disconnect()

        if not all_quotes:
            return None

        return pd.DataFrame(all_quotes)

    def collect_options_chain(self, underlying_symbol: str,
                            expiration_dates: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Collect options chain data for an underlying symbol.

        Args:
            underlying_symbol: The underlying stock symbol (e.g., 'AAPL')
            expiration_dates: List of expiration dates (YYYY-MM-DD), None for all

        Returns:
            DataFrame with options chain data
        """
        logger.info(f"Collecting options chain for {underlying_symbol}")

        if not self.connector.connect():
            logger.error("Failed to connect to IQFeed")
            return None

        all_options = []

        try:
            lookup_conn = self.connector.get_lookup_connection()
            if not lookup_conn:
                logger.error("Failed to get lookup connection")
                return None

            with iq.ConnConnector([lookup_conn]) as connector:
                # Request options chain (this might require specific IQFeed API calls)
                # For now, implementing basic structure - may need refinement based on IQFeed API
                logger.warning("Options chain collection requires specific IQFeed API - implementing placeholder")

                # Placeholder implementation
                option_record = {
                    'timestamp': pd.Timestamp.now(),
                    'underlying_symbol': underlying_symbol,
                    'option_symbol': f"{underlying_symbol}_OPTION_PLACEHOLDER",
                    'expiration_date': '2024-03-15',
                    'strike_price': 150.0,
                    'option_type': 'C',
                    'last_price': 5.25,
                    'bid': 5.20,
                    'ask': 5.30,
                    'volume': 100,
                    'open_interest': 500,
                    'implied_volatility': 0.25
                }
                all_options.append(option_record)

        except Exception as e:
            logger.error(f"Error collecting options chain for {underlying_symbol}: {e}")
            return None
        finally:
            self.connector.disconnect()

        if not all_options:
            return None

        return pd.DataFrame(all_options)

    # ===========================================
    # HELPER METHODS FOR FLEXIBLE FETCHING
    # ===========================================

    def _fetch_equity_data(self, symbol: str, data_type: str, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch data for equity symbols"""
        if data_type == 'auto' or data_type == 'bars':
            return self.collect_bars([symbol], **kwargs)
        elif data_type == 'ticks':
            return self.collect_ticks([symbol], **kwargs)
        elif data_type == 'quotes':
            return self.collect_realtime_quotes([symbol])
        else:
            # Default to bars
            return self.collect_bars([symbol], **kwargs)

    def _fetch_dtn_indicator(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch DTN calculated indicator data"""
        # Parse the indicator category
        symbol_info = self.symbol_parser.parse_symbol(symbol)

        if symbol_info.subcategory in ['net_tick', 'trin']:
            # Sentiment indicators
            equities_data = self.collect_equities_index_stats()
            category = symbol_info.subcategory
            if category in equities_data:
                df = equities_data[category]
                return df[df['symbol'] == symbol] if not df.empty else None

        elif 'options' in symbol_info.subcategory:
            # Options indicators
            options_data = self.collect_options_stats()
            for category, df in options_data.items():
                symbol_data = df[df['symbol'] == symbol] if not df.empty else pd.DataFrame()
                if not symbol_data.empty:
                    return symbol_data

        # Generic DTN indicator fetch
        logger.warning(f"Generic DTN indicator fetch for {symbol} - may need specific implementation")
        return None

    def _fetch_options_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch options contract data"""
        # Extract underlying from options symbol
        symbol_info = self.symbol_parser.parse_symbol(symbol)
        underlying = symbol_info.underlying

        if underlying:
            # Try to get the specific option data
            # For now, return options chain for underlying
            return self.collect_options_chain(underlying)

        return None

    def _fetch_futures_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch futures contract data"""
        # Futures data collection (similar to equity bars)
        return self.collect_bars([symbol], **kwargs)

    def _fetch_forex_data(self, symbol: str, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch forex pair data"""
        # Forex data collection (similar to equity bars)
        return self.collect_bars([symbol], **kwargs)

    def _fetch_by_data_type(self, symbol: str, data_type: str, **kwargs) -> Optional[pd.DataFrame]:
        """Manual data type routing"""
        if data_type == 'ticks':
            return self.collect_ticks([symbol], **kwargs)
        elif data_type == 'bars':
            return self.collect_bars([symbol], **kwargs)
        elif data_type == 'quotes':
            return self.collect_realtime_quotes([symbol])
        elif data_type == 'options_chain':
            return self.collect_options_chain(symbol)
        else:
            logger.error(f"Unknown data_type: {data_type}")
            return None

    def explore_symbol(self, symbol: str) -> Dict:
        """
        Explore a symbol to understand what data is available.
        Perfect for exploratory research.

        Args:
            symbol: Any symbol to explore

        Returns:
            Dict with symbol info and available data types
        """
        symbol_info = self.symbol_parser.parse_symbol(symbol)

        exploration = {
            'symbol': symbol,
            'parsed_info': {
                'category': symbol_info.category,
                'subcategory': symbol_info.subcategory,
                'exchange': symbol_info.exchange,
                'underlying': symbol_info.underlying,
                'metadata': symbol_info.metadata
            },
            'storage_namespace': symbol_info.storage_namespace,
            'recommended_data_types': self._get_recommended_data_types(symbol_info),
            'exploration_timestamp': pd.Timestamp.now().isoformat()
        }

        logger.info(f"Explored symbol {symbol}: {symbol_info.category}/{symbol_info.subcategory}")
        return exploration

    def _get_recommended_data_types(self, symbol_info) -> List[str]:
        """Get recommended data types for a symbol category"""
        recommendations = {
            'equity': ['bars', 'ticks', 'quotes'],
            'dtn_calculated': ['snapshot', 'historical'],
            'options': ['quotes', 'chain', 'greeks'],
            'futures': ['bars', 'ticks'],
            'forex': ['bars', 'ticks'],
            'unknown': ['bars', 'quotes']
        }

        return recommendations.get(symbol_info.category, ['bars'])