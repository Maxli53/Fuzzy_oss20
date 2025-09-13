"""
Professional BarBuilder - Hedge Fund Grade Bar Construction
Supports all classic and modern adaptive bar types
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import logging
from numba import jit
import warnings

logger = logging.getLogger(__name__)

class BarBuilder:
    """
    Professional-grade bar builder supporting all classic and modern bar types.
    Designed for tick-level IQFeed data stored in ArcticDB.
    """

    # =======================
    # Classic Bars
    # =======================

    @staticmethod
    def tick_bars(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Aggregate every n ticks into a bar.

        Args:
            df: DataFrame with tick data (timestamp, price, volume)
            n: Number of ticks per bar

        Returns:
            DataFrame with OHLCV bars and metadata
        """
        if len(df) == 0:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_ticks', 'vwap'])

        bars = []
        metadata = []

        for group_idx in range(0, len(df), n):
            chunk = df.iloc[group_idx:group_idx + n]
            if len(chunk) == 0:
                continue

            prices = chunk['price'].values
            volumes = chunk['volume'].values

            bar_data = {
                'timestamp': chunk.index[-1] if hasattr(chunk.index[-1], 'to_pydatetime') else chunk.iloc[-1]['timestamp'],
                'open': prices[0],
                'high': np.max(prices),
                'low': np.min(prices),
                'close': prices[-1],
                'volume': np.sum(volumes),
                'num_ticks': len(chunk),
                'vwap': np.sum(prices * volumes) / np.sum(volumes) if np.sum(volumes) > 0 else prices[-1]
            }

            # Metadata
            meta = {
                'bar_type': 'tick',
                'cum_dollar': np.sum(prices * volumes),
                'avg_tick_size': np.mean(volumes),
                'price_range': np.max(prices) - np.min(prices)
            }

            bars.append(bar_data)
            metadata.append(meta)

        result = pd.DataFrame(bars)
        if not result.empty:
            result['metadata'] = metadata

        return result

    @staticmethod
    def volume_bars(df: pd.DataFrame, volume_threshold: int) -> pd.DataFrame:
        """
        Aggregate ticks until cumulative volume exceeds threshold.

        Args:
            df: DataFrame with tick data
            volume_threshold: Volume threshold for bar completion

        Returns:
            DataFrame with volume-based bars
        """
        if len(df) == 0:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_ticks', 'vwap'])

        bars = []
        metadata = []

        cum_vol = 0
        prices = []
        volumes = []
        start_idx = 0

        for idx, row in df.iterrows():
            prices.append(row['price'])
            volumes.append(row['volume'])
            cum_vol += row['volume']

            if cum_vol >= volume_threshold:
                price_array = np.array(prices)
                volume_array = np.array(volumes)

                bar_data = {
                    'timestamp': idx if hasattr(idx, 'to_pydatetime') else row['timestamp'],
                    'open': price_array[0],
                    'high': np.max(price_array),
                    'low': np.min(price_array),
                    'close': price_array[-1],
                    'volume': np.sum(volume_array),
                    'num_ticks': len(prices),
                    'vwap': np.sum(price_array * volume_array) / np.sum(volume_array)
                }

                meta = {
                    'bar_type': 'volume',
                    'cum_dollar': np.sum(price_array * volume_array),
                    'avg_tick_size': np.mean(volume_array),
                    'price_range': np.max(price_array) - np.min(price_array),
                    'volume_threshold': volume_threshold
                }

                bars.append(bar_data)
                metadata.append(meta)

                # Reset accumulators
                cum_vol = 0
                prices = []
                volumes = []

        result = pd.DataFrame(bars)
        if not result.empty:
            result['metadata'] = metadata

        return result

    @staticmethod
    def dollar_bars(df: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
        """
        Aggregate ticks until cumulative dollar volume exceeds threshold.

        Args:
            df: DataFrame with tick data
            dollar_threshold: Dollar volume threshold

        Returns:
            DataFrame with dollar-based bars
        """
        if len(df) == 0:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_ticks', 'vwap'])

        bars = []
        metadata = []

        cum_dollar = 0
        prices = []
        volumes = []

        for idx, row in df.iterrows():
            prices.append(row['price'])
            volumes.append(row['volume'])
            cum_dollar += row['price'] * row['volume']

            if cum_dollar >= dollar_threshold:
                price_array = np.array(prices)
                volume_array = np.array(volumes)

                bar_data = {
                    'timestamp': idx if hasattr(idx, 'to_pydatetime') else row['timestamp'],
                    'open': price_array[0],
                    'high': np.max(price_array),
                    'low': np.min(price_array),
                    'close': price_array[-1],
                    'volume': np.sum(volume_array),
                    'num_ticks': len(prices),
                    'vwap': np.sum(price_array * volume_array) / np.sum(volume_array)
                }

                meta = {
                    'bar_type': 'dollar',
                    'cum_dollar': np.sum(price_array * volume_array),
                    'avg_tick_size': np.mean(volume_array),
                    'price_range': np.max(price_array) - np.min(price_array),
                    'dollar_threshold': dollar_threshold
                }

                bars.append(bar_data)
                metadata.append(meta)

                # Reset
                cum_dollar = 0
                prices = []
                volumes = []

        result = pd.DataFrame(bars)
        if not result.empty:
            result['metadata'] = metadata

        return result

    # =======================
    # Modern/Adaptive Bars
    # =======================

    @staticmethod
    def imbalance_bars(df: pd.DataFrame, imbalance_threshold: int) -> pd.DataFrame:
        """
        Form bars when cumulative buy-sell imbalance exceeds threshold.
        Requires 'side' column with 'buy'/'sell' or 1/-1 values.

        Args:
            df: DataFrame with tick data including 'side' column
            imbalance_threshold: Imbalance threshold for bar completion

        Returns:
            DataFrame with imbalance-based bars
        """
        if len(df) == 0:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_ticks', 'vwap'])

        if 'side' not in df.columns:
            logger.warning("No 'side' column found for imbalance bars, using volume bars instead")
            return BarBuilder.volume_bars(df, imbalance_threshold)

        bars = []
        metadata = []

        cum_imbalance = 0
        prices = []
        volumes = []
        sides = []

        for idx, row in df.iterrows():
            side = 1 if row['side'] in ['buy', 'B', 1] else -1
            prices.append(row['price'])
            volumes.append(row['volume'])
            sides.append(side)
            cum_imbalance += side * row['volume']

            if abs(cum_imbalance) >= imbalance_threshold:
                price_array = np.array(prices)
                volume_array = np.array(volumes)
                side_array = np.array(sides)

                buy_volume = np.sum(volume_array[side_array == 1])
                sell_volume = np.sum(volume_array[side_array == -1])

                bar_data = {
                    'timestamp': idx if hasattr(idx, 'to_pydatetime') else row['timestamp'],
                    'open': price_array[0],
                    'high': np.max(price_array),
                    'low': np.min(price_array),
                    'close': price_array[-1],
                    'volume': np.sum(volume_array),
                    'num_ticks': len(prices),
                    'vwap': np.sum(price_array * volume_array) / np.sum(volume_array)
                }

                meta = {
                    'bar_type': 'imbalance',
                    'cum_dollar': np.sum(price_array * volume_array),
                    'avg_tick_size': np.mean(volume_array),
                    'price_range': np.max(price_array) - np.min(price_array),
                    'imbalance_ratio': buy_volume / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0.5,
                    'buy_volume': buy_volume,
                    'sell_volume': sell_volume,
                    'final_imbalance': cum_imbalance,
                    'imbalance_threshold': imbalance_threshold
                }

                bars.append(bar_data)
                metadata.append(meta)

                # Reset
                cum_imbalance = 0
                prices = []
                volumes = []
                sides = []

        result = pd.DataFrame(bars)
        if not result.empty:
            result['metadata'] = metadata

        return result

    @staticmethod
    def volatility_bars(df: pd.DataFrame, volatility_threshold: float) -> pd.DataFrame:
        """
        Create bars when cumulative volatility (high-low) exceeds threshold.

        Args:
            df: DataFrame with tick data
            volatility_threshold: Cumulative volatility threshold

        Returns:
            DataFrame with volatility-based bars
        """
        if len(df) == 0:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_ticks', 'vwap'])

        bars = []
        metadata = []

        cum_volatility = 0
        prices = []
        volumes = []
        bar_high = -np.inf
        bar_low = np.inf

        for idx, row in df.iterrows():
            prices.append(row['price'])
            volumes.append(row['volume'])

            # Update bar high/low
            bar_high = max(bar_high, row['price'])
            bar_low = min(bar_low, row['price'])

            # Current volatility contribution
            cum_volatility = bar_high - bar_low

            if cum_volatility >= volatility_threshold:
                price_array = np.array(prices)
                volume_array = np.array(volumes)

                bar_data = {
                    'timestamp': idx if hasattr(idx, 'to_pydatetime') else row['timestamp'],
                    'open': price_array[0],
                    'high': bar_high,
                    'low': bar_low,
                    'close': price_array[-1],
                    'volume': np.sum(volume_array),
                    'num_ticks': len(prices),
                    'vwap': np.sum(price_array * volume_array) / np.sum(volume_array)
                }

                meta = {
                    'bar_type': 'volatility',
                    'cum_dollar': np.sum(price_array * volume_array),
                    'avg_tick_size': np.mean(volume_array),
                    'price_range': bar_high - bar_low,
                    'volatility_contribution': cum_volatility,
                    'volatility_threshold': volatility_threshold
                }

                bars.append(bar_data)
                metadata.append(meta)

                # Reset
                cum_volatility = 0
                prices = []
                volumes = []
                bar_high = -np.inf
                bar_low = np.inf

        result = pd.DataFrame(bars)
        if not result.empty:
            result['metadata'] = metadata

        return result

    @staticmethod
    def range_bars(df: pd.DataFrame, price_range: float) -> pd.DataFrame:
        """
        Create a bar whenever price moves by price_range from last bar close.

        Args:
            df: DataFrame with tick data
            price_range: Price range for bar completion

        Returns:
            DataFrame with range-based bars
        """
        if len(df) == 0:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_ticks', 'vwap'])

        bars = []
        metadata = []

        prices = []
        volumes = []
        last_close = None

        for idx, row in df.iterrows():
            if last_close is None:
                last_close = row['price']

            prices.append(row['price'])
            volumes.append(row['volume'])

            if abs(row['price'] - last_close) >= price_range:
                price_array = np.array(prices)
                volume_array = np.array(volumes)

                bar_data = {
                    'timestamp': idx if hasattr(idx, 'to_pydatetime') else row['timestamp'],
                    'open': price_array[0],
                    'high': np.max(price_array),
                    'low': np.min(price_array),
                    'close': price_array[-1],
                    'volume': np.sum(volume_array),
                    'num_ticks': len(prices),
                    'vwap': np.sum(price_array * volume_array) / np.sum(volume_array)
                }

                meta = {
                    'bar_type': 'range',
                    'cum_dollar': np.sum(price_array * volume_array),
                    'avg_tick_size': np.mean(volume_array),
                    'price_range': np.max(price_array) - np.min(price_array),
                    'range_threshold': price_range,
                    'move_from_last_close': abs(price_array[-1] - last_close)
                }

                bars.append(bar_data)
                metadata.append(meta)

                # Reset
                last_close = row['price']
                prices = []
                volumes = []

        result = pd.DataFrame(bars)
        if not result.empty:
            result['metadata'] = metadata

        return result

    @staticmethod
    def renko_bars(df: pd.DataFrame, brick_size: float) -> pd.DataFrame:
        """
        Renko bar builder (price move based).

        Args:
            df: DataFrame with tick data
            brick_size: Size of each Renko brick

        Returns:
            DataFrame with Renko bars
        """
        if len(df) == 0:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_ticks'])

        bars = []
        last_close = None

        for idx, row in df.iterrows():
            if last_close is None:
                last_close = row['price']
                continue

            while abs(row['price'] - last_close) >= brick_size:
                direction = np.sign(row['price'] - last_close)
                new_close = last_close + direction * brick_size

                bar_data = {
                    'timestamp': idx if hasattr(idx, 'to_pydatetime') else row['timestamp'],
                    'open': last_close,
                    'high': max(last_close, new_close),
                    'low': min(last_close, new_close),
                    'close': new_close,
                    'volume': 0,  # Renko bars don't track volume
                    'num_ticks': 1,
                    'brick_size': brick_size,
                    'direction': direction
                }

                bars.append(bar_data)
                last_close = new_close

        return pd.DataFrame(bars)

    @staticmethod
    def event_driven_bars(df: pd.DataFrame, event_func: Callable) -> pd.DataFrame:
        """
        Create bars based on a custom event function.

        Args:
            df: DataFrame with tick data
            event_func: Function that takes (row, open, high, low, close, volume) and returns bool

        Returns:
            DataFrame with event-driven bars
        """
        if len(df) == 0:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'num_ticks', 'vwap'])

        bars = []
        metadata = []

        prices = []
        volumes = []
        bar_open = None
        bar_high = -np.inf
        bar_low = np.inf

        for idx, row in df.iterrows():
            if bar_open is None:
                bar_open = row['price']

            prices.append(row['price'])
            volumes.append(row['volume'])
            bar_high = max(bar_high, row['price'])
            bar_low = min(bar_low, row['price'])

            # Check event condition
            try:
                if event_func(row, bar_open, bar_high, bar_low, row['price'], np.sum(volumes)):
                    price_array = np.array(prices)
                    volume_array = np.array(volumes)

                    bar_data = {
                        'timestamp': idx if hasattr(idx, 'to_pydatetime') else row['timestamp'],
                        'open': bar_open,
                        'high': bar_high,
                        'low': bar_low,
                        'close': row['price'],
                        'volume': np.sum(volume_array),
                        'num_ticks': len(prices),
                        'vwap': np.sum(price_array * volume_array) / np.sum(volume_array)
                    }

                    meta = {
                        'bar_type': 'event_driven',
                        'cum_dollar': np.sum(price_array * volume_array),
                        'avg_tick_size': np.mean(volume_array),
                        'price_range': bar_high - bar_low
                    }

                    bars.append(bar_data)
                    metadata.append(meta)

                    # Reset
                    prices = []
                    volumes = []
                    bar_open = None
                    bar_high = -np.inf
                    bar_low = np.inf

            except Exception as e:
                logger.error(f"Error in event function: {e}")
                continue

        result = pd.DataFrame(bars)
        if not result.empty:
            result['metadata'] = metadata

        return result