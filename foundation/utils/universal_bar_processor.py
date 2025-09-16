"""
Universal Bar Processor

Implements the GUI-driven architecture documented in Data_policy.md.
Processes ticks through multiple bar builders simultaneously to provide
instant access to any bar type a user might request.

Key Features:
- Processes ticks through ALL configured bar types in parallel
- Each completed bar gets Tier 2 metadata computed
- Bars stored in ArcticDB for instant retrieval
- Supports time, tick, volume, dollar, range, and renko bars
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import logging

from foundation.models.market import TickData, TimeBar
from foundation.models.enums import BarType
from foundation.utils.bar_builder import (
    TimeBarBuilder, VolumeBarBuilder, DollarBarBuilder,
    TickBarBuilder, RangeBarBuilder, RenkoBarBuilder,
    ImbalanceBarBuilder
)
from foundation.utils.metadata_computer import MetadataComputer

logger = logging.getLogger(__name__)


class UniversalBarProcessor:
    """
    Processes ticks through multiple bar builders simultaneously.
    Each completed bar gets its own Tier 2 metadata computed and stored.
    """

    def __init__(self, symbol: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Universal Bar Processor.

        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            config: Configuration for bar types and parameters
        """
        self.symbol = symbol
        self.config = config or self._get_default_config()
        self.builders = {}
        self.completed_bars = []
        self.tick_count = 0
        self.metadata_computer = MetadataComputer()

        # Initialize all configured bar builders
        self._initialize_builders()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for all bar types"""
        return {
            'time_bars': {
                'enabled': True,
                'intervals': [60, 300, 900]  # 1-min, 5-min, 15-min in seconds
            },
            'tick_bars': {
                'enabled': True,
                'sizes': [100, 500, 1000]  # Number of ticks
            },
            'volume_bars': {
                'enabled': True,
                'thresholds': [10000, 50000, 100000]  # Share volume
            },
            'dollar_bars': {
                'enabled': True,
                'thresholds': [100000, 500000, 1000000]  # Dollar volume
            },
            'range_bars': {
                'enabled': True,
                'ranges': [0.50, 1.00, 2.00]  # Price range
            },
            'renko_bars': {
                'enabled': True,
                'brick_sizes': [0.25, 0.50, 1.00]  # Brick size
            },
            'imbalance_bars': {
                'enabled': True,
                'thresholds': [5000, 10000, 20000]  # Volume imbalance
            }
        }

    def _initialize_builders(self) -> None:
        """Initialize all configured bar builders"""

        # Time bars
        if self.config['time_bars']['enabled']:
            for interval in self.config['time_bars']['intervals']:
                key = f'time_{interval}'
                self.builders[key] = TimeBarBuilder(self.symbol, interval_seconds=interval)
                logger.info(f"Initialized TimeBarBuilder: {interval}s")

        # Tick bars
        if self.config['tick_bars']['enabled']:
            for size in self.config['tick_bars']['sizes']:
                key = f'tick_{size}'
                self.builders[key] = TickBarBuilder(self.symbol, tick_threshold=size)
                logger.info(f"Initialized TickBarBuilder: {size} ticks")

        # Volume bars
        if self.config['volume_bars']['enabled']:
            for threshold in self.config['volume_bars']['thresholds']:
                key = f'volume_{threshold}'
                self.builders[key] = VolumeBarBuilder(
                    self.symbol,
                    volume_threshold=Decimal(str(threshold))
                )
                logger.info(f"Initialized VolumeBarBuilder: {threshold} shares")

        # Dollar bars
        if self.config['dollar_bars']['enabled']:
            for threshold in self.config['dollar_bars']['thresholds']:
                key = f'dollar_{threshold}'
                self.builders[key] = DollarBarBuilder(
                    self.symbol,
                    dollar_threshold=Decimal(str(threshold))
                )
                logger.info(f"Initialized DollarBarBuilder: ${threshold}")

        # Range bars
        if self.config['range_bars']['enabled']:
            for price_range in self.config['range_bars']['ranges']:
                key = f'range_{price_range}'
                self.builders[key] = RangeBarBuilder(
                    self.symbol,
                    price_range=Decimal(str(price_range))
                )
                logger.info(f"Initialized RangeBarBuilder: ${price_range}")

        # Renko bars
        if self.config['renko_bars']['enabled']:
            for brick_size in self.config['renko_bars']['brick_sizes']:
                key = f'renko_{brick_size}'
                self.builders[key] = RenkoBarBuilder(
                    self.symbol,
                    brick_size=Decimal(str(brick_size))
                )
                logger.info(f"Initialized RenkoBarBuilder: ${brick_size}")

        # Imbalance bars
        if self.config['imbalance_bars']['enabled']:
            for threshold in self.config['imbalance_bars']['thresholds']:
                key = f'imbalance_{threshold}'
                self.builders[key] = ImbalanceBarBuilder(
                    self.symbol,
                    imbalance_threshold=Decimal(str(threshold))
                )
                logger.info(f"Initialized ImbalanceBarBuilder: {threshold} shares")

        logger.info(f"Initialized {len(self.builders)} bar builders for {self.symbol}")

    def process_tick(self, tick: TickData) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """
        Process single tick through all builders.
        Returns completed bars with their metadata.

        Args:
            tick: TickData object to process

        Returns:
            Dictionary of completed bars with their Tier 2 metadata
            Format: {builder_key: (bar_object, tier2_metadata)}
        """
        self.tick_count += 1
        completed = {}

        # Process tick through each builder
        for key, builder in self.builders.items():
            bar = builder.add_tick(tick)

            if bar is not None:
                # Compute Tier 2 metadata for the completed bar
                metadata = self._compute_bar_metadata(bar, key)

                # Store for return
                completed[key] = (bar, metadata)

                # Track internally
                self.completed_bars.append({
                    'builder': key,
                    'bar': bar,
                    'metadata': metadata,
                    'timestamp': datetime.now()
                })

                logger.debug(f"Completed {key} bar at {bar.timestamp}")

        return completed

    def process_ticks(self, ticks: List[TickData]) -> Dict[str, List[Tuple[Any, Dict[str, Any]]]]:
        """
        Process multiple ticks in batch.

        Args:
            ticks: List of TickData objects

        Returns:
            Dictionary with all completed bars grouped by builder type
        """
        all_completed = {}

        for tick in ticks:
            completed = self.process_tick(tick)

            for key, (bar, metadata) in completed.items():
                if key not in all_completed:
                    all_completed[key] = []
                all_completed[key].append((bar, metadata))

        # Log summary
        total_bars = sum(len(bars) for bars in all_completed.values())
        logger.info(f"Processed {len(ticks)} ticks, generated {total_bars} bars")

        return all_completed

    def force_close_all(self) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """
        Force close all bars (e.g., at market close).
        Useful for time bars that haven't naturally completed.

        Returns:
            Dictionary of forced closed bars with metadata
        """
        completed = {}

        for key, builder in self.builders.items():
            # Only time bars support force close
            if hasattr(builder, 'force_close'):
                bar = builder.force_close()
                if bar is not None:
                    metadata = self._compute_bar_metadata(bar, key)
                    completed[key] = (bar, metadata)
                    logger.info(f"Force closed {key} bar")

        return completed

    def _compute_bar_metadata(self, bar: Any, builder_key: str) -> Dict[str, Any]:
        """
        Compute Tier 2 metadata for a completed bar.

        Args:
            bar: Completed bar object
            builder_key: Key identifying the builder type

        Returns:
            Dictionary containing Tier 2 metadata
        """
        metadata = {
            'symbol': self.symbol,
            'bar_type': builder_key,
            'timestamp': bar.timestamp.isoformat(),
            'computed_at': datetime.now().isoformat()
        }

        # Extract bar type and parameters
        parts = builder_key.split('_')
        bar_type_str = parts[0]

        metadata['bar_type_category'] = bar_type_str
        if len(parts) > 1:
            metadata['bar_parameter'] = parts[1]

        # Add bar-specific metrics
        metadata['ohlcv'] = {
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': int(bar.volume)
        }

        # Add microstructure metrics if available
        if hasattr(bar, 'avg_spread_bps') and bar.avg_spread_bps is not None:
            metadata['avg_spread_bps'] = bar.avg_spread_bps

        if hasattr(bar, 'liquidity_score'):
            metadata['liquidity_score'] = bar.liquidity_score

        if hasattr(bar, 'buy_volume') and hasattr(bar, 'sell_volume'):
            metadata['flow_imbalance'] = {
                'buy_volume': int(bar.buy_volume),
                'sell_volume': int(bar.sell_volume),
                'net_flow': int(bar.buy_volume - bar.sell_volume)
            }

        if hasattr(bar, 'vwap') and bar.vwap is not None:
            metadata['vwap'] = float(bar.vwap)

        if hasattr(bar, 'trade_count'):
            metadata['trade_count'] = bar.trade_count

        # Add time-specific metrics
        if bar_type_str == 'time' and hasattr(bar, 'interval_seconds'):
            metadata['interval_seconds'] = bar.interval_seconds
            if hasattr(bar, 'gaps'):
                metadata['gaps'] = bar.gaps
            if hasattr(bar, 'is_complete'):
                metadata['is_complete'] = bar.is_complete

        return metadata

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with processing stats
        """
        stats = {
            'symbol': self.symbol,
            'total_ticks_processed': self.tick_count,
            'total_bars_generated': len(self.completed_bars),
            'builders_active': len(self.builders),
            'bars_by_type': {}
        }

        # Count bars by builder type
        for bar_info in self.completed_bars:
            builder_key = bar_info['builder']
            if builder_key not in stats['bars_by_type']:
                stats['bars_by_type'][builder_key] = 0
            stats['bars_by_type'][builder_key] += 1

        return stats

    def reset(self) -> None:
        """Reset all builders and statistics"""
        for builder in self.builders.values():
            if hasattr(builder, 'reset'):
                builder.accumulator.reset()

        self.completed_bars.clear()
        self.tick_count = 0
        logger.info(f"Reset Universal Bar Processor for {self.symbol}")


def create_universal_processor(symbol: str, **kwargs) -> UniversalBarProcessor:
    """
    Factory function to create a Universal Bar Processor.

    Args:
        symbol: Trading symbol
        **kwargs: Additional configuration options

    Returns:
        Configured UniversalBarProcessor instance
    """
    config = kwargs.get('config', None)
    return UniversalBarProcessor(symbol, config)