"""
Metadata Computation Module

Implements institutional-grade metadata computation following Data_policy.md specifications:
- Tier 2 metadata: Aggregated statistics computed from tick DataFrames
- Phase 1: Essential metadata (basic stats, spread stats, trade classification, liquidity, execution)
- Designed for both day-level and bar-level metadata computation

The module processes tick DataFrames to compute comprehensive metadata
that enables instant insights without loading full tick data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class MetadataComputer:
    """
    Compute institutional-grade Tier 2 metadata from tick data.

    Tier 2 metadata is aggregated statistics computed AFTER DataFrame storage,
    providing instant insights without loading the full tick data.
    """

    @staticmethod
    def compute_phase1_metadata(df: pd.DataFrame, symbol: str, date: str) -> Dict[str, Any]:
        """
        Compute Phase 1 Tier 2 metadata from tick DataFrame.

        This computes aggregated statistics across all ticks in the DataFrame,
        not per-tick metadata (which is Tier 1, already in the DataFrame).

        Args:
            df: DataFrame with tick data (containing Tier 1 per-tick metadata)
            symbol: Symbol identifier
            date: Date string (YYYY-MM-DD)

        Returns:
            Dictionary with Phase 1 Tier 2 metadata categories
        """
        metadata = {}

        # 1.1 Basic Statistics
        metadata['basic_stats'] = MetadataComputer._compute_basic_stats(df, symbol, date)

        # 1.2 Spread Statistics
        metadata['spread_stats'] = MetadataComputer._compute_spread_stats(df)

        # 1.3 Trade Classification
        metadata['trade_classification'] = MetadataComputer._compute_trade_classification(df)

        # 1.4 Liquidity Profile
        metadata['liquidity_profile'] = MetadataComputer._compute_liquidity_profile(df)

        # 1.5 Execution Metrics
        metadata['execution_metrics'] = MetadataComputer._compute_execution_metrics(df)

        # Meta information
        metadata['meta'] = {
            'computed_at': datetime.now().isoformat(),
            'phase': 1,
            'tick_count': len(df),
            'metadata_version': '1.0.0',
            'tier': 2  # Explicitly mark as Tier 2 metadata
        }

        return metadata

    @staticmethod
    def _compute_basic_stats(df: pd.DataFrame, symbol: str, date: str) -> Dict[str, Any]:
        """Compute basic OHLCV statistics (aggregated across all ticks)"""
        try:
            stats = {
                'symbol': symbol,
                'date': date,
                'total_ticks': len(df),
                'first_tick_time': str(df['timestamp'].iloc[0]) if len(df) > 0 else None,
                'last_tick_time': str(df['timestamp'].iloc[-1]) if len(df) > 0 else None,
                'price_open': float(df['price'].iloc[0]) if len(df) > 0 else None,
                'price_high': float(df['price'].max()) if len(df) > 0 else None,
                'price_low': float(df['price'].min()) if len(df) > 0 else None,
                'price_close': float(df['price'].iloc[-1]) if len(df) > 0 else None,
                'volume_total': int(df['volume'].sum()) if 'volume' in df.columns else 0,
                'vwap': None,
                'dollar_volume': float(df['dollar_volume'].sum()) if 'dollar_volume' in df.columns else 0.0
            }

            # Calculate VWAP (Volume Weighted Average Price)
            if 'volume' in df.columns and stats['volume_total'] > 0:
                stats['vwap'] = float((df['price'] * df['volume']).sum() / df['volume'].sum())

            return stats

        except Exception as e:
            logger.error(f"Error computing basic stats: {e}")
            return {}

    @staticmethod
    def _compute_spread_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute spread statistics in basis points.
        Aggregates the spread_bps column (Tier 1) into statistics (Tier 2).
        """
        try:
            if 'spread_bps' not in df.columns:
                return {}

            spread_data = df['spread_bps'].dropna()
            if len(spread_data) == 0:
                return {}

            stats = {
                'mean_bps': float(spread_data.mean()),
                'median_bps': float(spread_data.median()),
                'std_bps': float(spread_data.std()) if len(spread_data) > 1 else 0.0,
                'min_bps': float(spread_data.min()),
                'max_bps': float(spread_data.max()),
                'p25_bps': float(spread_data.quantile(0.25)),
                'p75_bps': float(spread_data.quantile(0.75)),
                'p95_bps': float(spread_data.quantile(0.95)),
                'p99_bps': float(spread_data.quantile(0.99)),
                'zero_spread_count': int((spread_data == 0).sum()),
                'inverted_count': int((spread_data < 0).sum())
            }

            return stats

        except Exception as e:
            logger.error(f"Error computing spread stats: {e}")
            return {}

    @staticmethod
    def _compute_trade_classification(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute trade classification statistics.
        Aggregates the trade_sign column (Tier 1) into flow metrics (Tier 2).
        """
        try:
            if 'trade_sign' not in df.columns:
                return {}

            # Count trades by classification
            buy_mask = df['trade_sign'] == 1
            sell_mask = df['trade_sign'] == -1
            neutral_mask = df['trade_sign'] == 0

            stats = {
                'buy_count': int(buy_mask.sum()),
                'sell_count': int(sell_mask.sum()),
                'neutral_count': int(neutral_mask.sum()),
                'buy_volume': int(df.loc[buy_mask, 'volume'].sum()) if 'volume' in df.columns else 0,
                'sell_volume': int(df.loc[sell_mask, 'volume'].sum()) if 'volume' in df.columns else 0,
                'buy_dollar_volume': float(df.loc[buy_mask, 'dollar_volume'].sum()) if 'dollar_volume' in df.columns else 0.0,
                'sell_dollar_volume': float(df.loc[sell_mask, 'dollar_volume'].sum()) if 'dollar_volume' in df.columns else 0.0,
                'buy_sell_ratio': None,
                'volume_weighted_sign': None,
                'large_buy_count': 0,
                'large_sell_count': 0
            }

            # Calculate ratios
            if stats['sell_count'] > 0:
                stats['buy_sell_ratio'] = float(stats['buy_count'] / stats['sell_count'])

            total_volume = stats['buy_volume'] + stats['sell_volume']
            if total_volume > 0:
                stats['volume_weighted_sign'] = float((stats['buy_volume'] - stats['sell_volume']) / total_volume)

            # Count large trades (>10k shares)
            if 'volume' in df.columns:
                large_trades = df['volume'] > 10000
                stats['large_buy_count'] = int((large_trades & buy_mask).sum())
                stats['large_sell_count'] = int((large_trades & sell_mask).sum())

            return stats

        except Exception as e:
            logger.error(f"Error computing trade classification: {e}")
            return {}

    @staticmethod
    def _compute_liquidity_profile(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute liquidity profile metrics"""
        try:
            stats = {
                'quote_intensity': 0.0,
                'avg_trade_size': 0.0,
                'median_trade_size': 0.0,
                'trade_frequency': 0.0,
                'effective_tick_size': 0.0,
                'price_levels_count': 0,
                'time_between_trades_ms': 0.0,
                'liquidity_score': 0.0
            }

            if len(df) < 2:
                return stats

            # Quote intensity (updates per second)
            if 'timestamp' in df.columns:
                time_range = (pd.to_datetime(df['timestamp'].iloc[-1]) -
                            pd.to_datetime(df['timestamp'].iloc[0])).total_seconds()
                if time_range > 0:
                    stats['quote_intensity'] = float(len(df) / time_range)
                    stats['trade_frequency'] = float(len(df) / (time_range / 60))  # per minute

                # Time between trades
                time_diffs = pd.to_datetime(df['timestamp']).diff().dt.total_seconds() * 1000
                stats['time_between_trades_ms'] = float(time_diffs.mean())

            # Trade sizes
            if 'volume' in df.columns:
                stats['avg_trade_size'] = float(df['volume'].mean())
                stats['median_trade_size'] = float(df['volume'].median())

            # Price levels and tick size
            if 'price' in df.columns:
                unique_prices = df['price'].unique()
                stats['price_levels_count'] = int(len(unique_prices))

                # Effective tick size (minimum non-zero price movement)
                if len(unique_prices) > 1:
                    price_diffs = np.diff(np.sort(unique_prices))
                    positive_diffs = price_diffs[price_diffs > 0]
                    if len(positive_diffs) > 0:
                        stats['effective_tick_size'] = float(positive_diffs.min())

            # Composite liquidity score (0-100)
            stats['liquidity_score'] = MetadataComputer._calculate_liquidity_score(df, stats)

            return stats

        except Exception as e:
            logger.error(f"Error computing liquidity profile: {e}")
            return {}

    @staticmethod
    def _compute_execution_metrics(df: pd.DataFrame) -> Dict[str, Any]:
        """Compute execution quality metrics"""
        try:
            stats = {
                'effective_spread_mean': 0.0,
                'effective_spread_median': 0.0,
                'price_improvement_rate': 0.0,
                'at_midpoint_rate': 0.0,
                'at_bid_rate': 0.0,
                'at_ask_rate': 0.0,
                'outside_quote_rate': 0.0,
                'odd_lot_rate': 0.0,
                'block_rate': 0.0
            }

            if len(df) == 0:
                return stats

            # Effective spread (2 * |price - midpoint|)
            if 'price' in df.columns and 'midpoint' in df.columns:
                effective_spreads = 2 * np.abs(df['price'] - df['midpoint'])
                stats['effective_spread_mean'] = float(effective_spreads.mean())
                stats['effective_spread_median'] = float(effective_spreads.median())

                # Price improvement (trades inside spread)
                if 'bid' in df.columns and 'ask' in df.columns:
                    inside_spread = (df['price'] > df['bid']) & (df['price'] < df['ask'])
                    stats['price_improvement_rate'] = float(inside_spread.sum() / len(df))

                    # Execution location rates
                    at_midpoint = np.abs(df['price'] - df['midpoint']) < 0.0001
                    stats['at_midpoint_rate'] = float(at_midpoint.sum() / len(df))

                    at_bid = np.abs(df['price'] - df['bid']) < 0.0001
                    stats['at_bid_rate'] = float(at_bid.sum() / len(df))

                    at_ask = np.abs(df['price'] - df['ask']) < 0.0001
                    stats['at_ask_rate'] = float(at_ask.sum() / len(df))

                    outside = (df['price'] < df['bid']) | (df['price'] > df['ask'])
                    stats['outside_quote_rate'] = float(outside.sum() / len(df))

            # Trade size classifications
            if 'volume' in df.columns:
                odd_lots = df['volume'] < 100
                stats['odd_lot_rate'] = float(odd_lots.sum() / len(df))

                blocks = df['volume'] > 10000
                stats['block_rate'] = float(blocks.sum() / len(df))

            return stats

        except Exception as e:
            logger.error(f"Error computing execution metrics: {e}")
            return {}

    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame, liquidity_stats: Dict[str, Any]) -> float:
        """
        Calculate composite liquidity score (0-100).

        Components:
        - Trade frequency (25 points)
        - Average trade size (25 points)
        - Spread tightness (25 points)
        - Price levels diversity (25 points)
        """
        score = 0.0

        # Trade frequency component (0-25)
        if liquidity_stats['trade_frequency'] > 0:
            # Scale: 10 trades/min = 25 points
            freq_score = min(25.0, liquidity_stats['trade_frequency'] / 10 * 25)
            score += freq_score

        # Trade size component (0-25)
        if liquidity_stats['avg_trade_size'] > 0:
            # Scale: 1000 shares avg = 25 points
            size_score = min(25.0, liquidity_stats['avg_trade_size'] / 1000 * 25)
            score += size_score

        # Spread tightness component (0-25)
        if 'spread_bps' in df.columns:
            mean_spread = df['spread_bps'].mean()
            if mean_spread >= 0:
                # Scale: 0 bps = 25 points, 10 bps = 0 points
                spread_score = max(0.0, 25.0 - mean_spread * 2.5)
                score += spread_score

        # Price levels diversity (0-25)
        if liquidity_stats['price_levels_count'] > 0:
            # Scale: 100 levels = 25 points
            levels_score = min(25.0, liquidity_stats['price_levels_count'] / 100 * 25)
            score += levels_score

        return min(100.0, max(0.0, score))

    @staticmethod
    def compute_bar_metadata(bar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute Tier 2 metadata for a single bar.
        This is a lighter version focused on bar-specific metrics.

        Args:
            bar_data: Dictionary containing bar information (OHLCV + enriched fields)

        Returns:
            Dictionary with bar-level Tier 2 metadata
        """
        metadata = {
            'spread_stats': {
                'mean_bps': bar_data.get('avg_spread_bps', 0.0),
                'volatility': bar_data.get('spread_volatility', 0.0)
            },
            'liquidity_metrics': {
                'score': bar_data.get('liquidity_score', 0.0),
                'trade_count': bar_data.get('trade_count', 0)
            },
            'execution_quality': {
                'effective_spread': bar_data.get('effective_spread_mean', 0.0),
                'vwap': float(bar_data.get('vwap', 0.0)) if bar_data.get('vwap') else 0.0
            },
            'flow_analysis': {
                'buy_volume': int(bar_data.get('buy_volume', 0)),
                'sell_volume': int(bar_data.get('sell_volume', 0)),
                'imbalance': int(bar_data.get('buy_volume', 0)) - int(bar_data.get('sell_volume', 0))
            },
            'meta': {
                'computed_at': datetime.now().isoformat(),
                'bar_type': bar_data.get('bar_type', 'unknown'),
                'tier': 2
            }
        }

        return metadata

    @staticmethod
    def compute_summary_report(metadata: Dict[str, Any]) -> str:
        """Generate human-readable summary report from Tier 2 metadata"""
        report = []
        report.append("=" * 60)
        report.append("TIER 2 METADATA SUMMARY REPORT")
        report.append("=" * 60)

        # Basic info
        if 'basic_stats' in metadata:
            stats = metadata['basic_stats']
            report.append(f"\n=== BASIC STATISTICS ===")
            report.append(f"Symbol: {stats.get('symbol', 'N/A')}")
            report.append(f"Date: {stats.get('date', 'N/A')}")
            report.append(f"Total Ticks: {stats.get('total_ticks', 0):,}")
            report.append(f"Trading Hours: {stats.get('first_tick_time', 'N/A')} to {stats.get('last_tick_time', 'N/A')}")
            report.append(f"Price Range: ${stats.get('price_low', 0):.2f} - ${stats.get('price_high', 0):.2f}")
            report.append(f"VWAP: ${stats.get('vwap', 0):.2f}")
            report.append(f"Total Volume: {stats.get('volume_total', 0):,}")
            report.append(f"Dollar Volume: ${stats.get('dollar_volume', 0):,.2f}")

        # Spread statistics
        if 'spread_stats' in metadata:
            stats = metadata['spread_stats']
            report.append(f"\n=== SPREAD STATISTICS (Tier 2) ===")
            report.append(f"Mean Spread: {stats.get('mean_bps', 0):.2f} bps")
            report.append(f"Median Spread: {stats.get('median_bps', 0):.2f} bps")
            report.append(f"Spread Range: {stats.get('min_bps', 0):.2f} - {stats.get('max_bps', 0):.2f} bps")
            report.append(f"95th Percentile: {stats.get('p95_bps', 0):.2f} bps")
            report.append(f"Zero Spreads: {stats.get('zero_spread_count', 0):,}")
            report.append(f"Inverted Spreads: {stats.get('inverted_count', 0):,}")

        # Trade classification
        if 'trade_classification' in metadata:
            stats = metadata['trade_classification']
            report.append(f"\n=== TRADE CLASSIFICATION ===")
            report.append(f"Buy Orders: {stats.get('buy_count', 0):,} ({stats.get('buy_volume', 0):,} shares)")
            report.append(f"Sell Orders: {stats.get('sell_count', 0):,} ({stats.get('sell_volume', 0):,} shares)")
            report.append(f"Buy/Sell Ratio: {stats.get('buy_sell_ratio', 0):.2f}")
            report.append(f"Net Order Flow: {stats.get('volume_weighted_sign', 0):.3f}")
            report.append(f"Large Buy Blocks: {stats.get('large_buy_count', 0):,}")
            report.append(f"Large Sell Blocks: {stats.get('large_sell_count', 0):,}")

        # Liquidity profile
        if 'liquidity_profile' in metadata:
            stats = metadata['liquidity_profile']
            report.append(f"\n=== LIQUIDITY PROFILE ===")
            report.append(f"Quote Intensity: {stats.get('quote_intensity', 0):.2f} updates/sec")
            report.append(f"Trade Frequency: {stats.get('trade_frequency', 0):.2f} trades/min")
            report.append(f"Avg Trade Size: {stats.get('avg_trade_size', 0):.0f} shares")
            report.append(f"Median Trade Size: {stats.get('median_trade_size', 0):.0f} shares")
            report.append(f"Price Levels: {stats.get('price_levels_count', 0)} unique prices")
            report.append(f"Effective Tick Size: ${stats.get('effective_tick_size', 0):.4f}")
            report.append(f"Liquidity Score: {stats.get('liquidity_score', 0):.1f}/100")

        # Execution metrics
        if 'execution_metrics' in metadata:
            stats = metadata['execution_metrics']
            report.append(f"\n=== EXECUTION QUALITY ===")
            report.append(f"Effective Spread Mean: ${stats.get('effective_spread_mean', 0):.4f}")
            report.append(f"Effective Spread Median: ${stats.get('effective_spread_median', 0):.4f}")
            report.append(f"Price Improvement: {stats.get('price_improvement_rate', 0)*100:.1f}%")
            report.append(f"At Midpoint: {stats.get('at_midpoint_rate', 0)*100:.1f}%")
            report.append(f"At Bid: {stats.get('at_bid_rate', 0)*100:.1f}%")
            report.append(f"At Ask: {stats.get('at_ask_rate', 0)*100:.1f}%")
            report.append(f"Outside Quotes: {stats.get('outside_quote_rate', 0)*100:.1f}%")
            report.append(f"Odd Lots: {stats.get('odd_lot_rate', 0)*100:.1f}%")
            report.append(f"Block Trades: {stats.get('block_rate', 0)*100:.1f}%")

        # Metadata info
        if 'meta' in metadata:
            meta = metadata['meta']
            report.append(f"\n=== METADATA INFO ===")
            report.append(f"Computed At: {meta.get('computed_at', 'N/A')}")
            report.append(f"Phase: {meta.get('phase', 'N/A')}")
            report.append(f"Tier: {meta.get('tier', 'N/A')}")
            report.append(f"Version: {meta.get('metadata_version', 'N/A')}")
            report.append(f"Tick Count: {meta.get('tick_count', 0):,}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)