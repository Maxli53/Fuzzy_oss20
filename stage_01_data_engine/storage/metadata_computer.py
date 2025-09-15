"""
Comprehensive Metadata Computer for Tick Data
Computes institutional-grade metadata from tick DataFrames
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetadataComputer:
    """
    Computes comprehensive metadata from tick data DataFrames.
    Implements all metadata concepts defined in Data_policy.md.
    """

    METADATA_VERSION = "2.0"

    # Trading hours in ET
    MARKET_OPEN = pd.Timestamp("09:30:00").time()
    MARKET_CLOSE = pd.Timestamp("16:00:00").time()

    # Thresholds
    LARGE_TRADE_SIZE = 10000  # shares
    BLOCK_TRADE_SIZE = 50000  # shares
    SPREAD_JUMP_MULTIPLIER = 3  # 3x normal spread
    PRICE_GAP_THRESHOLD = 0.001  # 0.1% price move

    def __init__(self):
        """Initialize the metadata computer"""
        self.peer_groups = {
            'tech_large': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'etf_index': ['SPY', 'QQQ', 'IWM', 'DIA'],
            'financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS']
        }

    def compute_all_metadata(self, df: pd.DataFrame, symbol: str, date: str) -> Dict:
        """
        Compute all metadata for a symbol-date.

        Args:
            df: DataFrame with tick data and essential metrics
            symbol: Stock symbol
            date: Date string (YYYY-MM-DD)

        Returns:
            Complete metadata dictionary
        """
        logger.info(f"Computing metadata for {symbol} on {date}")

        metadata = {
            'symbol': symbol,
            'date': date,
            'version': self.METADATA_VERSION,
            'computed_at': datetime.now().isoformat(),
        }

        # Phase 1: Essential Metadata (always computed)
        metadata.update(self._compute_phase1_metadata(df))

        # Advanced Metadata Concepts
        metadata['intraday_evolution'] = self._compute_intraday_evolution(df)
        metadata['event_impacts'] = self._compute_event_impacts(df)
        metadata['trade_networks'] = self._compute_trade_networks(df)
        metadata['data_quality'] = self._compute_data_quality(df)
        metadata['venue_analysis'] = self._compute_venue_analysis(df)

        # Note: Phase 2 (relative metrics) computed separately after all symbols processed

        return metadata

    def _compute_phase1_metadata(self, df: pd.DataFrame) -> Dict:
        """Compute essential metadata that doesn't require other symbols"""

        phase1 = {}

        # Basic Statistics
        phase1['basic_stats'] = self._compute_basic_stats(df)

        # Spread Statistics
        if 'spread_bps' in df.columns:
            phase1['spread_stats'] = self._compute_spread_stats(df)

        # Trade Classification
        if 'trade_sign' in df.columns:
            phase1['trade_classification'] = self._compute_trade_classification(df)

        # Liquidity Profile
        phase1['liquidity_profile'] = self._compute_liquidity_profile(df)

        # Execution Metrics
        phase1['execution_metrics'] = self._compute_execution_metrics(df)

        return phase1

    def _compute_basic_stats(self, df: pd.DataFrame) -> Dict:
        """Compute basic statistics"""

        prices = df['price'].values
        volumes = df['volume'].values

        return {
            'total_ticks': len(df),
            'first_tick_time': df['timestamp'].iloc[0].isoformat(),
            'last_tick_time': df['timestamp'].iloc[-1].isoformat(),
            'price_open': float(prices[0]),
            'price_high': float(np.max(prices)),
            'price_low': float(np.min(prices)),
            'price_close': float(prices[-1]),
            'price_mean': float(np.mean(prices)),
            'price_std': float(np.std(prices)),
            'volume_total': int(np.sum(volumes)),
            'volume_mean': float(np.mean(volumes)),
            'volume_std': float(np.std(volumes)),
            'vwap': float(np.sum(prices * volumes) / np.sum(volumes)),
            'dollar_volume': float(np.sum(prices * volumes))
        }

    def _compute_spread_stats(self, df: pd.DataFrame) -> Dict:
        """Compute spread statistics"""

        spreads = df['spread_bps'].values

        # Remove NaN values
        valid_spreads = spreads[~np.isnan(spreads)]

        if len(valid_spreads) == 0:
            return {}

        return {
            'mean_bps': float(np.mean(valid_spreads)),
            'median_bps': float(np.median(valid_spreads)),
            'std_bps': float(np.std(valid_spreads)),
            'min_bps': float(np.min(valid_spreads)),
            'max_bps': float(np.max(valid_spreads)),
            'p25_bps': float(np.percentile(valid_spreads, 25)),
            'p75_bps': float(np.percentile(valid_spreads, 75)),
            'p95_bps': float(np.percentile(valid_spreads, 95)),
            'p99_bps': float(np.percentile(valid_spreads, 99)),
            'zero_spread_count': int(np.sum(valid_spreads == 0)),
            'inverted_count': int(np.sum(valid_spreads < 0))
        }

    def _compute_trade_classification(self, df: pd.DataFrame) -> Dict:
        """Compute trade classification summary"""

        if 'trade_sign' not in df.columns:
            return {}

        signs = df['trade_sign'].values
        volumes = df['volume'].values
        prices = df['price'].values

        buy_mask = signs == 1
        sell_mask = signs == -1
        neutral_mask = signs == 0

        buy_volume = np.sum(volumes[buy_mask])
        sell_volume = np.sum(volumes[sell_mask])

        # Large trades
        large_mask = volumes > self.LARGE_TRADE_SIZE
        large_buy = np.sum((signs == 1) & large_mask)
        large_sell = np.sum((signs == -1) & large_mask)

        return {
            'buy_count': int(np.sum(buy_mask)),
            'sell_count': int(np.sum(sell_mask)),
            'neutral_count': int(np.sum(neutral_mask)),
            'buy_volume': int(buy_volume),
            'sell_volume': int(sell_volume),
            'buy_dollar_volume': float(np.sum(prices[buy_mask] * volumes[buy_mask])),
            'sell_dollar_volume': float(np.sum(prices[sell_mask] * volumes[sell_mask])),
            'buy_sell_ratio': float(buy_volume / sell_volume) if sell_volume > 0 else np.inf,
            'volume_weighted_sign': float(np.sum(signs * volumes) / np.sum(volumes)),
            'large_buy_count': int(large_buy),
            'large_sell_count': int(large_sell)
        }

    def _compute_liquidity_profile(self, df: pd.DataFrame) -> Dict:
        """Compute liquidity profile metrics"""

        # Calculate time differences
        timestamps = pd.to_datetime(df['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds() * 1000  # milliseconds

        # Trading intensity
        duration_hours = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600
        if duration_hours > 0:
            quote_intensity = len(df) / duration_hours / 3600  # updates per second
            trade_frequency = len(df) / (duration_hours * 60)  # trades per minute
        else:
            quote_intensity = 0
            trade_frequency = 0

        # Price levels
        unique_prices = df['price'].nunique()
        price_range = df['price'].max() - df['price'].min()
        if price_range > 0:
            effective_tick_size = df['price'].diff().abs().min()
        else:
            effective_tick_size = 0.01

        # Liquidity score (0-100)
        liquidity_score = self._calculate_liquidity_score(df)

        return {
            'quote_intensity': float(quote_intensity),
            'avg_trade_size': float(df['volume'].mean()),
            'median_trade_size': float(df['volume'].median()),
            'trade_frequency': float(trade_frequency),
            'effective_tick_size': float(effective_tick_size),
            'price_levels_count': int(unique_prices),
            'time_between_trades_ms': float(time_diffs.median()) if len(time_diffs) > 1 else 0,
            'liquidity_score': float(liquidity_score)
        }

    def _compute_execution_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute execution quality metrics"""

        if 'effective_spread' not in df.columns or 'midpoint' not in df.columns:
            return {}

        # Execution locations
        at_bid = df['price'] == df['bid']
        at_ask = df['price'] == df['ask']
        at_mid = df['price'] == df['midpoint']
        inside_spread = (df['price'] > df['bid']) & (df['price'] < df['ask'])
        outside_quote = (df['price'] < df['bid']) | (df['price'] > df['ask'])

        # Trade size categories
        odd_lot = df['volume'] < 100
        block = df['volume'] >= self.BLOCK_TRADE_SIZE

        total_trades = len(df)

        return {
            'effective_spread_mean': float(df['effective_spread'].mean()),
            'effective_spread_median': float(df['effective_spread'].median()),
            'price_improvement_rate': float(inside_spread.sum() / total_trades),
            'at_midpoint_rate': float(at_mid.sum() / total_trades),
            'at_bid_rate': float(at_bid.sum() / total_trades),
            'at_ask_rate': float(at_ask.sum() / total_trades),
            'outside_quote_rate': float(outside_quote.sum() / total_trades),
            'odd_lot_rate': float(odd_lot.sum() / total_trades),
            'block_rate': float(block.sum() / total_trades)
        }

    def _compute_intraday_evolution(self, df: pd.DataFrame) -> Dict:
        """Compute hourly evolution of metrics"""

        df['hour'] = pd.to_datetime(df['timestamp']).dt.floor('H')
        hourly_groups = df.groupby('hour')

        evolution = {}
        total_volume = df['volume'].sum()

        for hour, group in hourly_groups:
            hour_str = f"{hour.strftime('%H:%M')}-{(hour + pd.Timedelta(hours=1)).strftime('%H:%M')}"

            if 'spread_bps' in group.columns:
                spread_mean = group['spread_bps'].mean()
            else:
                spread_mean = 0

            if 'trade_sign' in group.columns:
                buy_volume = group[group['trade_sign'] == 1]['volume'].sum()
                sell_volume = group[group['trade_sign'] == -1]['volume'].sum()
                buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else 1
            else:
                buy_sell_ratio = 1

            evolution[hour_str] = {
                'spread_mean_bps': float(spread_mean),
                'volume_pct': float(group['volume'].sum() / total_volume * 100),
                'volatility': float(group['price'].std()),
                'trade_count': int(len(group)),
                'buy_sell_ratio': float(buy_sell_ratio)
            }

        return evolution

    def _compute_event_impacts(self, df: pd.DataFrame) -> Dict:
        """Compute impact of significant events"""

        events = {
            'large_trades': [],
            'spread_jumps': [],
            'price_gaps': []
        }

        # Large trades
        large_trades = df[df['volume'] >= self.BLOCK_TRADE_SIZE]
        for idx, trade in large_trades.iterrows():
            # Get spreads before and after
            idx_pos = df.index.get_loc(idx)

            if idx_pos > 0 and idx_pos < len(df) - 1:
                spread_before = df.iloc[idx_pos - 1]['spread_bps'] if 'spread_bps' in df.columns else 0
                spread_after = df.iloc[idx_pos + 1]['spread_bps'] if 'spread_bps' in df.columns else 0

                # Price impact
                price_before = df.iloc[idx_pos - 1]['price']
                price_impact_bps = abs((trade['price'] - price_before) / price_before) * 10000

                # Following activity
                following_5s = df[(df['timestamp'] > trade['timestamp']) &
                                 (df['timestamp'] <= trade['timestamp'] + pd.Timedelta(seconds=5))]

                events['large_trades'].append({
                    'timestamp': trade['timestamp'].strftime('%H:%M:%S'),
                    'size': int(trade['volume']),
                    'price': float(trade['price']),
                    'spread_before_bps': float(spread_before),
                    'spread_after_bps': float(spread_after),
                    'price_impact_bps': float(price_impact_bps),
                    'following_trades': int(len(following_5s))
                })

        # Spread jumps
        if 'spread_bps' in df.columns:
            spread_median = df['spread_bps'].median()
            spread_jumps = df[df['spread_bps'] > spread_median * self.SPREAD_JUMP_MULTIPLIER]

            for idx, jump in spread_jumps.iterrows():
                idx_pos = df.index.get_loc(idx)
                if idx_pos > 0:
                    spread_before = df.iloc[idx_pos - 1]['spread_bps']

                    # Find duration of wide spread
                    wide_spreads = df[(df.index >= idx) &
                                      (df['spread_bps'] > spread_median * 2)]
                    if len(wide_spreads) > 1:
                        duration_ms = (wide_spreads.iloc[-1]['timestamp'] -
                                      jump['timestamp']).total_seconds() * 1000
                    else:
                        duration_ms = 0

                    events['spread_jumps'].append({
                        'timestamp': jump['timestamp'].strftime('%H:%M:%S'),
                        'from_bps': float(spread_before),
                        'to_bps': float(jump['spread_bps']),
                        'duration_ms': float(duration_ms),
                        'volume_during': int(wide_spreads['volume'].sum()) if len(wide_spreads) > 0 else 0
                    })

        # Price gaps
        price_returns = df['price'].pct_change().abs()
        gaps = df[price_returns > self.PRICE_GAP_THRESHOLD]

        for idx, gap in gaps.iterrows():
            idx_pos = df.index.get_loc(idx)
            if idx_pos > 0:
                events['price_gaps'].append({
                    'timestamp': gap['timestamp'].strftime('%H:%M:%S'),
                    'from_price': float(df.iloc[idx_pos - 1]['price']),
                    'to_price': float(gap['price']),
                    'gap_bps': float(price_returns.iloc[idx_pos] * 10000)
                })

        return events

    def _compute_trade_networks(self, df: pd.DataFrame) -> Dict:
        """Compute trade clustering and network effects"""

        networks = {
            'trade_clusters': [],
            'momentum_chains': [],
            'liquidity_clusters': []
        }

        # Trade clusters (rapid sequences)
        df['time_diff'] = pd.to_datetime(df['timestamp']).diff().dt.total_seconds()

        # Find rapid trade sequences (< 1 second apart)
        rapid_mask = df['time_diff'] < 1
        cluster_starts = (~rapid_mask).shift(1).fillna(True) & rapid_mask

        cluster_id = 0
        for start_idx in df[cluster_starts].index:
            # Find cluster end
            idx_pos = df.index.get_loc(start_idx)
            cluster_df = df.iloc[idx_pos:]
            cluster_end = cluster_df[cluster_df['time_diff'] >= 1].index

            if len(cluster_end) > 0:
                cluster = df.loc[start_idx:cluster_end[0]]
            else:
                cluster = df.loc[start_idx:]

            if len(cluster) >= 5:  # Minimum cluster size
                networks['trade_clusters'].append({
                    'cluster_id': cluster_id,
                    'start_time': cluster.iloc[0]['timestamp'].strftime('%H:%M:%S'),
                    'end_time': cluster.iloc[-1]['timestamp'].strftime('%H:%M:%S'),
                    'trade_count': int(len(cluster)),
                    'total_volume': int(cluster['volume'].sum()),
                    'avg_trade_size': float(cluster['volume'].mean()),
                    'price_range': [float(cluster['price'].min()), float(cluster['price'].max())],
                    'fragmentation': float(cluster['exchange'].nunique() / len(cluster)) if 'exchange' in cluster.columns else 0
                })
                cluster_id += 1

        # Momentum chains (directional moves)
        if 'trade_sign' in df.columns:
            # Find sequences of same-direction trades
            df['sign_change'] = df['trade_sign'].diff().fillna(0) != 0
            df['momentum_group'] = df['sign_change'].cumsum()

            for group_id, group in df.groupby('momentum_group'):
                if len(group) >= 5 and group['trade_sign'].iloc[0] != 0:
                    price_move = (group['price'].iloc[-1] - group['price'].iloc[0]) / group['price'].iloc[0] * 10000

                    if abs(price_move) > 10:  # Significant move (10+ bps)
                        networks['momentum_chains'].append({
                            'chain_id': int(group_id),
                            'start_time': group['timestamp'].iloc[0].strftime('%H:%M:%S'),
                            'initial_trade': int(group['volume'].iloc[0]),
                            'initial_price': float(group['price'].iloc[0]),
                            'follower_trades': int(len(group) - 1),
                            'total_momentum_volume': int(group['volume'].sum()),
                            'price_move_bps': float(price_move),
                            'duration_seconds': float((group['timestamp'].iloc[-1] -
                                                      group['timestamp'].iloc[0]).total_seconds())
                        })

        return networks

    def _compute_data_quality(self, df: pd.DataFrame) -> Dict:
        """Compute data quality metrics"""

        quality = {
            'completeness_score': 0.0,
            'confidence_level': 'high',
            'tick_gaps': [],
            'anomalies': {'count': 0, 'details': []},
            'crossed_markets': {'count': 0, 'total_duration_ms': 0, 'max_cross_bps': 0}
        }

        # Completeness score
        timestamps = pd.to_datetime(df['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds()

        # Expect at least 1 tick every 5 seconds during active trading
        gaps = time_diffs[time_diffs > 5]
        if len(gaps) > 0:
            quality['completeness_score'] = float(1 - (len(gaps) / len(df)))
        else:
            quality['completeness_score'] = 1.0

        # Find tick gaps
        for idx, gap in enumerate(gaps):
            if gap > 15:  # Gap > 15 seconds
                gap_idx = gaps.index[idx]
                quality['tick_gaps'].append({
                    'from': timestamps.iloc[gap_idx - 1].strftime('%H:%M:%S'),
                    'to': timestamps.iloc[gap_idx].strftime('%H:%M:%S'),
                    'missing_ticks_estimate': int(gap / 0.5),  # Assume 2 ticks/second normally
                    'last_price_before': float(df.iloc[gap_idx - 1]['price']),
                    'first_price_after': float(df.iloc[gap_idx]['price'])
                })

        # Anomaly detection
        price_zscore = np.abs(stats.zscore(df['price'].values))
        anomaly_mask = price_zscore > 4

        for idx in df[anomaly_mask].index:
            quality['anomalies']['details'].append({
                'time': df.loc[idx, 'timestamp'].strftime('%H:%M:%S'),
                'type': 'price_spike',
                'deviation_sigma': float(price_zscore[df.index.get_loc(idx)])
            })

        quality['anomalies']['count'] = len(quality['anomalies']['details'])

        # Crossed markets
        if 'bid' in df.columns and 'ask' in df.columns:
            crossed = df['bid'] > df['ask']
            quality['crossed_markets']['count'] = int(crossed.sum())

            if crossed.sum() > 0:
                cross_spreads = df.loc[crossed, 'bid'] - df.loc[crossed, 'ask']
                quality['crossed_markets']['max_cross_bps'] = float(
                    cross_spreads.max() / df.loc[crossed, 'midpoint'].mean() * 10000
                )

        # Confidence level
        if quality['completeness_score'] < 0.9 or quality['anomalies']['count'] > 10:
            quality['confidence_level'] = 'low'
        elif quality['completeness_score'] < 0.95 or quality['anomalies']['count'] > 5:
            quality['confidence_level'] = 'medium'

        return quality

    def _compute_venue_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze execution venues"""

        if 'exchange' not in df.columns:
            return {}

        venue = {
            'exchange_distribution': {},
            'venue_quality': {},
            'fragmentation_score': 0.0,
            'dark_pool_percentage': 0.0,
            'best_execution_venue': '',
            'venue_migration': {}
        }

        # Exchange distribution
        total_volume = df['volume'].sum()
        for exchange, group in df.groupby('exchange'):
            venue['exchange_distribution'][exchange] = float(group['volume'].sum() / total_volume)

            # Venue quality metrics
            if 'spread_bps' in group.columns:
                avg_spread = group['spread_bps'].mean()
            else:
                avg_spread = 0

            venue['venue_quality'][exchange] = {
                'avg_spread_bps': float(avg_spread),
                'fill_rate': 1.0,  # Placeholder - would need order data
                'avg_size': float(group['volume'].mean())
            }

        # Fragmentation score (Herfindahl index)
        shares = list(venue['exchange_distribution'].values())
        venue['fragmentation_score'] = float(1 - sum(s**2 for s in shares))

        # Dark pool percentage (ADF trades)
        if 'D' in venue['exchange_distribution']:
            venue['dark_pool_percentage'] = float(venue['exchange_distribution']['D'] * 100)

        # Best execution venue (lowest spread)
        if venue['venue_quality']:
            best = min(venue['venue_quality'].items(),
                      key=lambda x: x[1]['avg_spread_bps'])
            venue['best_execution_venue'] = best[0]

        # Venue migration through the day
        df['period'] = pd.cut(pd.to_datetime(df['timestamp']).dt.hour,
                              bins=[0, 12, 14, 24],
                              labels=['morning', 'midday', 'close'])

        for period, group in df.groupby('period'):
            period_dist = {}
            period_volume = group['volume'].sum()
            for exchange, ex_group in group.groupby('exchange'):
                period_dist[exchange] = float(ex_group['volume'].sum() / period_volume)
            venue['venue_migration'][period] = period_dist

        return venue

    def _calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """
        Calculate composite liquidity score (0-100).

        Higher score = better liquidity
        """
        score = 50.0  # Base score

        # Factor 1: Spread tightness (up to +20 points)
        if 'spread_bps' in df.columns:
            median_spread = df['spread_bps'].median()
            if median_spread < 5:
                score += 20
            elif median_spread < 10:
                score += 15
            elif median_spread < 20:
                score += 10
            elif median_spread < 50:
                score += 5

        # Factor 2: Trade frequency (up to +20 points)
        trades_per_minute = len(df) / ((df['timestamp'].iloc[-1] -
                                        df['timestamp'].iloc[0]).total_seconds() / 60)
        if trades_per_minute > 100:
            score += 20
        elif trades_per_minute > 50:
            score += 15
        elif trades_per_minute > 20:
            score += 10
        elif trades_per_minute > 10:
            score += 5

        # Factor 3: Volume consistency (up to +10 points)
        volume_cv = df['volume'].std() / df['volume'].mean()  # Coefficient of variation
        if volume_cv < 1:
            score += 10
        elif volume_cv < 2:
            score += 5

        return min(100, max(0, score))


class RelativeMetadataComputer:
    """
    Computes Phase 2 metadata that requires multiple symbols.
    Run after all Phase 1 metadata is complete.
    """

    def __init__(self, metadata_store):
        """
        Initialize with access to metadata store.

        Args:
            metadata_store: Object with get_metadata(symbol, date) method
        """
        self.metadata_store = metadata_store
        self.sector_etfs = {
            'tech': 'XLK',
            'financial': 'XLF',
            'energy': 'XLE',
            'healthcare': 'XLV'
        }

    def compute_relative_metadata(self, symbol: str, date: str,
                                 peer_symbols: List[str]) -> Dict:
        """
        Compute relative metrics vs peers and market.

        Args:
            symbol: Target symbol
            date: Date string
            peer_symbols: List of peer symbols for comparison

        Returns:
            Dictionary of relative metrics
        """
        relative = {}

        # Get target metadata
        target_meta = self.metadata_store.get_metadata(symbol, date)
        if not target_meta:
            return {}

        # Get peer metadata
        peer_metadata = {}
        for peer in peer_symbols:
            meta = self.metadata_store.get_metadata(peer, date)
            if meta:
                peer_metadata[peer] = meta

        if not peer_metadata:
            return {}

        # Compute relative metrics
        relative['vs_peers'] = self._compare_to_peers(target_meta, peer_metadata)
        relative['vs_market'] = self._compare_to_market(target_meta, date)
        relative['peer_rank'] = self._calculate_peer_rankings(target_meta, peer_metadata)

        return relative

    def _compare_to_peers(self, target_meta: Dict, peer_metadata: Dict) -> Dict:
        """Compare target to peer average"""

        comparison = {}

        # Extract peer spreads
        peer_spreads = []
        for peer_meta in peer_metadata.values():
            if 'spread_stats' in peer_meta and 'mean_bps' in peer_meta['spread_stats']:
                peer_spreads.append(peer_meta['spread_stats']['mean_bps'])

        if peer_spreads and 'spread_stats' in target_meta:
            target_spread = target_meta['spread_stats'].get('mean_bps', 0)
            comparison['spread_ratio'] = float(target_spread / np.mean(peer_spreads))

        # Volume comparison
        peer_volumes = []
        for peer_meta in peer_metadata.values():
            if 'basic_stats' in peer_meta:
                peer_volumes.append(peer_meta['basic_stats'].get('volume_total', 0))

        if peer_volumes and 'basic_stats' in target_meta:
            target_volume = target_meta['basic_stats'].get('volume_total', 0)
            comparison['volume_ratio'] = float(target_volume / np.mean(peer_volumes))

        return comparison

    def _compare_to_market(self, target_meta: Dict, date: str) -> Dict:
        """Compare to market (SPY)"""

        market_meta = self.metadata_store.get_metadata('SPY', date)
        if not market_meta:
            return {}

        comparison = {}

        # Spread comparison
        if 'spread_stats' in target_meta and 'spread_stats' in market_meta:
            target_spread = target_meta['spread_stats'].get('mean_bps', 0)
            market_spread = market_meta['spread_stats'].get('mean_bps', 0)
            comparison['spread_vs_spy'] = float(target_spread / market_spread) if market_spread > 0 else 0

        return comparison

    def _calculate_peer_rankings(self, target_meta: Dict, peer_metadata: Dict) -> Dict:
        """Calculate rankings among peers"""

        rankings = {}
        all_metadata = {'target': target_meta, **peer_metadata}

        # Rank by spread
        spreads = []
        for symbol, meta in all_metadata.items():
            if 'spread_stats' in meta:
                spreads.append((symbol, meta['spread_stats'].get('mean_bps', float('inf'))))

        spreads.sort(key=lambda x: x[1])
        for rank, (sym, _) in enumerate(spreads, 1):
            if sym == 'target':
                rankings['spread_rank'] = rank
                break

        rankings['universe_size'] = len(all_metadata)

        return rankings