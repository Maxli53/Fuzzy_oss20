"""
Adaptive Threshold Calibration System
Dynamic, volatility-aware bar thresholds for professional trading
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class AdaptiveThresholds:
    """
    Professional adaptive threshold calibration system.

    Features:
    - Per-symbol calibration based on historical statistics
    - Market regime awareness (high/low volatility, volume)
    - Percentile-based dynamic adjustments
    - Calibration snapshots for reproducibility
    - Real-time threshold updates
    """

    def __init__(self,
                 tick_store=None,
                 config_path: str = "stage_01_data_engine/config/symbol_config.yaml"):
        """
        Initialize adaptive threshold system.

        Args:
            tick_store: TickStore instance for historical data
            config_path: Path to symbol configuration file
        """
        self.tick_store = tick_store
        self.config_path = Path(config_path)

        # Load base configuration
        self.base_config = self._load_base_config()

        # Calibration cache
        self.calibration_cache = {}

        # Market regime detection parameters
        self.regime_params = {
            'vol_lookback_days': 20,
            'vol_high_threshold': 1.5,  # Multiple of historical volatility
            'vol_low_threshold': 0.7,
            'volume_high_threshold': 1.3,  # Multiple of avg volume
            'volume_low_threshold': 0.8
        }

        logger.info("Adaptive threshold system initialized")

    def calibrate(self,
                 symbol: str,
                 lookback_days: int = 20,
                 save_snapshot: bool = True) -> Dict[str, float]:
        """
        Calibrate adaptive thresholds for a symbol.

        Args:
            symbol: Stock symbol
            lookback_days: Days of historical data to analyze
            save_snapshot: Whether to save calibration snapshot

        Returns:
            Dictionary of calibrated thresholds
        """
        try:
            logger.info(f"Calibrating thresholds for {symbol} using {lookback_days} days")

            # Get historical tick data
            historical_data = self._get_historical_data(symbol, lookback_days)

            if historical_data is None or historical_data.empty:
                logger.warning(f"No historical data for {symbol}, using base config")
                return self._get_base_thresholds(symbol)

            # Calculate market statistics
            stats = self._calculate_market_stats(historical_data)

            # Detect current market regime
            regime = self._detect_market_regime(historical_data, stats)

            # Calculate adaptive thresholds
            thresholds = self._calculate_adaptive_thresholds(symbol, stats, regime)

            # Apply regime-based multipliers
            adjusted_thresholds = self._apply_regime_multipliers(thresholds, regime)

            # Cache calibration results
            calibration_result = {
                'symbol': symbol,
                'calibrated_at': datetime.now().isoformat(),
                'lookback_days': lookback_days,
                'market_regime': regime,
                'statistics': stats,
                'thresholds': adjusted_thresholds
            }

            self.calibration_cache[symbol] = calibration_result

            # Save snapshot if requested
            if save_snapshot:
                self._save_calibration_snapshot(symbol, calibration_result)

            logger.info(f"Calibrated thresholds for {symbol}: {adjusted_thresholds}")
            logger.info(f"Market regime: {regime}")

            return adjusted_thresholds

        except Exception as e:
            logger.error(f"Error calibrating thresholds for {symbol}: {e}")
            return self._get_base_thresholds(symbol)

    def get_thresholds(self, symbol: str, use_cached: bool = True) -> Dict[str, float]:
        """
        Get current thresholds for a symbol.

        Args:
            symbol: Stock symbol
            use_cached: Whether to use cached calibration

        Returns:
            Dictionary of current thresholds
        """
        if use_cached and symbol in self.calibration_cache:
            cached = self.calibration_cache[symbol]

            # Check if calibration is still fresh (within 24 hours)
            calibrated_at = pd.Timestamp(cached['calibrated_at'])
            if datetime.now() - calibrated_at.to_pydatetime() < timedelta(hours=24):
                return cached['thresholds']

        # Recalibrate if no fresh cache
        return self.calibrate(symbol)

    def batch_calibrate(self, symbols: List[str], lookback_days: int = 20) -> Dict[str, Dict]:
        """
        Calibrate thresholds for multiple symbols.

        Args:
            symbols: List of symbols to calibrate
            lookback_days: Historical data lookback period

        Returns:
            Dictionary mapping symbol to calibration results
        """
        logger.info(f"Batch calibrating {len(symbols)} symbols")

        results = {}

        for i, symbol in enumerate(symbols):
            logger.info(f"Calibrating {symbol} ({i+1}/{len(symbols)})")

            try:
                thresholds = self.calibrate(symbol, lookback_days)
                results[symbol] = {
                    'success': True,
                    'thresholds': thresholds,
                    'calibration_data': self.calibration_cache.get(symbol, {})
                }
            except Exception as e:
                logger.error(f"Failed to calibrate {symbol}: {e}")
                results[symbol] = {
                    'success': False,
                    'error': str(e),
                    'thresholds': self._get_base_thresholds(symbol)
                }

        logger.info(f"Batch calibration completed: {len(results)} symbols processed")
        return results

    def update_real_time(self, symbol: str, recent_data: pd.DataFrame) -> Dict[str, float]:
        """
        Update thresholds based on recent market activity.

        Args:
            symbol: Stock symbol
            recent_data: Recent tick data (last few hours)

        Returns:
            Updated thresholds
        """
        try:
            # Get current cached thresholds
            current_thresholds = self.get_thresholds(symbol)

            # Calculate recent market statistics
            recent_stats = self._calculate_market_stats(recent_data)

            # Detect current regime from recent data
            recent_regime = self._detect_market_regime(recent_data, recent_stats)

            # Apply dynamic adjustment based on recent activity
            adjustment_factor = self._calculate_adjustment_factor(recent_stats, recent_regime)

            # Update thresholds
            updated_thresholds = {}
            for key, value in current_thresholds.items():
                updated_thresholds[key] = value * adjustment_factor.get(key, 1.0)

            logger.info(f"Real-time threshold update for {symbol}: adjustment factor = {adjustment_factor}")

            return updated_thresholds

        except Exception as e:
            logger.error(f"Error updating real-time thresholds for {symbol}: {e}")
            return self.get_thresholds(symbol)

    def _get_historical_data(self, symbol: str, lookback_days: int) -> Optional[pd.DataFrame]:
        """Get historical tick data for calibration"""
        if self.tick_store is None:
            return None

        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)

            return self.tick_store.load_ticks(
                symbol,
                (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            )
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return None

    def _calculate_market_stats(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive market statistics"""
        try:
            prices = data['price'].values
            volumes = data['volume'].values

            # Price statistics
            price_returns = np.diff(np.log(prices))
            realized_volatility = np.std(price_returns) * np.sqrt(252 * 24 * 60 * 60)  # Annualized

            # Volume statistics
            volume_mean = np.mean(volumes)
            volume_std = np.std(volumes)

            # Dollar volume
            dollar_volumes = prices[:-1] * volumes[:-1]  # Align with returns
            dollar_volume_mean = np.mean(dollar_volumes)

            # Trade size distribution
            volume_percentiles = np.percentile(volumes, [10, 25, 50, 75, 90])

            # Intraday patterns
            data_with_time = data.copy()
            data_with_time['hour'] = pd.to_datetime(data_with_time['timestamp']).dt.hour
            hourly_volume = data_with_time.groupby('hour')['volume'].mean()

            stats = {
                # Price volatility
                'realized_volatility': realized_volatility,
                'price_std': np.std(prices),
                'daily_range': (np.max(prices) - np.min(prices)) / np.mean(prices),

                # Volume characteristics
                'volume_mean': volume_mean,
                'volume_std': volume_std,
                'volume_skew': float(pd.Series(volumes).skew()),
                'volume_p10': volume_percentiles[0],
                'volume_p25': volume_percentiles[1],
                'volume_p50': volume_percentiles[2],
                'volume_p75': volume_percentiles[3],
                'volume_p90': volume_percentiles[4],

                # Dollar volume
                'dollar_volume_mean': dollar_volume_mean,
                'dollar_volume_std': np.std(dollar_volumes),

                # Activity patterns
                'tick_frequency': len(data) / ((data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]).total_seconds() / 3600),
                'avg_trade_size': volume_mean,
                'market_hours_activity': float(hourly_volume.loc[hourly_volume.index.isin(range(9, 16))].mean()) if len(hourly_volume) > 0 else volume_mean
            }

            return stats

        except Exception as e:
            logger.error(f"Error calculating market stats: {e}")
            return {}

    def _detect_market_regime(self, data: pd.DataFrame, stats: Dict[str, float]) -> Dict[str, str]:
        """Detect current market regime"""
        try:
            regime = {
                'volatility': 'normal',
                'volume': 'normal',
                'activity': 'normal'
            }

            # Volatility regime
            vol_threshold_high = self.regime_params['vol_high_threshold']
            vol_threshold_low = self.regime_params['vol_low_threshold']

            if stats.get('realized_volatility', 1.0) > vol_threshold_high:
                regime['volatility'] = 'high'
            elif stats.get('realized_volatility', 1.0) < vol_threshold_low:
                regime['volatility'] = 'low'

            # Volume regime
            vol_high_threshold = self.regime_params['volume_high_threshold']
            vol_low_threshold = self.regime_params['volume_low_threshold']

            if stats.get('volume_mean', 1000) > vol_high_threshold * 10000:  # Arbitrary baseline
                regime['volume'] = 'high'
            elif stats.get('volume_mean', 1000) < vol_low_threshold * 10000:
                regime['volume'] = 'low'

            # Activity regime based on tick frequency
            tick_freq = stats.get('tick_frequency', 100)
            if tick_freq > 500:
                regime['activity'] = 'high'
            elif tick_freq < 100:
                regime['activity'] = 'low'

            return regime

        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {'volatility': 'normal', 'volume': 'normal', 'activity': 'normal'}

    def _calculate_adaptive_thresholds(self,
                                     symbol: str,
                                     stats: Dict[str, float],
                                     regime: Dict[str, str]) -> Dict[str, float]:
        """Calculate adaptive thresholds based on statistics"""
        try:
            base_thresholds = self._get_base_thresholds(symbol)

            # Volume threshold: 25th percentile of historical volume
            volume_threshold = max(
                stats.get('volume_p25', base_thresholds['volume_threshold']),
                base_thresholds['volume_threshold'] * 0.5  # Minimum threshold
            )

            # Dollar threshold: 1% of average dollar volume
            dollar_threshold = max(
                stats.get('dollar_volume_mean', 0) * 0.01,
                base_thresholds['dollar_threshold'] * 0.5
            )

            # Imbalance threshold: 2 * volume std
            imbalance_threshold = max(
                stats.get('volume_std', 0) * 2.0,
                base_thresholds['imbalance_threshold'] * 0.5
            )

            # Volatility threshold: 0.5 * recent price volatility
            volatility_threshold = max(
                stats.get('price_std', 0) * 0.5,
                base_thresholds['volatility_threshold'] * 0.3
            )

            # Range and brick size based on daily range
            daily_range = stats.get('daily_range', 0.01)
            range_size = max(
                daily_range * 0.1,  # 10% of daily range
                base_thresholds['range_size'] * 0.5
            )

            brick_size = max(
                daily_range * 0.05,  # 5% of daily range
                base_thresholds['brick_size'] * 0.5
            )

            adaptive_thresholds = {
                'volume_threshold': int(volume_threshold),
                'dollar_threshold': int(dollar_threshold),
                'imbalance_threshold': int(imbalance_threshold),
                'flow_threshold': int(imbalance_threshold * 1.2),  # Slightly higher than imbalance
                'volatility_threshold': round(volatility_threshold, 3),
                'range_size': round(range_size, 3),
                'brick_size': round(brick_size, 3)
            }

            return adaptive_thresholds

        except Exception as e:
            logger.error(f"Error calculating adaptive thresholds: {e}")
            return self._get_base_thresholds(symbol)

    def _apply_regime_multipliers(self,
                                 thresholds: Dict[str, float],
                                 regime: Dict[str, str]) -> Dict[str, float]:
        """Apply regime-based multipliers to thresholds"""
        try:
            adjusted = thresholds.copy()

            # Volatility regime adjustments
            vol_multiplier = 1.0
            if regime['volatility'] == 'high':
                vol_multiplier = 1.5  # Increase thresholds in high volatility
            elif regime['volatility'] == 'low':
                vol_multiplier = 0.7  # Decrease thresholds in low volatility

            # Volume regime adjustments
            volume_multiplier = 1.0
            if regime['volume'] == 'high':
                volume_multiplier = 1.3
            elif regime['volume'] == 'low':
                volume_multiplier = 0.6

            # Apply multipliers
            vol_sensitive = ['volatility_threshold', 'range_size', 'brick_size']
            volume_sensitive = ['volume_threshold', 'dollar_threshold', 'imbalance_threshold', 'flow_threshold']

            for key, value in adjusted.items():
                if key in vol_sensitive:
                    adjusted[key] = value * vol_multiplier
                elif key in volume_sensitive:
                    adjusted[key] = value * volume_multiplier

            return adjusted

        except Exception as e:
            logger.error(f"Error applying regime multipliers: {e}")
            return thresholds

    def _calculate_adjustment_factor(self,
                                   recent_stats: Dict[str, float],
                                   recent_regime: Dict[str, str]) -> Dict[str, float]:
        """Calculate real-time adjustment factors"""
        adjustment = {}

        # Base adjustment is 1.0 (no change)
        for key in ['volume_threshold', 'dollar_threshold', 'imbalance_threshold',
                   'flow_threshold', 'volatility_threshold', 'range_size', 'brick_size']:
            adjustment[key] = 1.0

        # Adjust based on recent regime
        if recent_regime['volatility'] == 'high':
            adjustment['volatility_threshold'] = 1.2
            adjustment['range_size'] = 1.2
            adjustment['brick_size'] = 1.2
        elif recent_regime['volatility'] == 'low':
            adjustment['volatility_threshold'] = 0.8
            adjustment['range_size'] = 0.8
            adjustment['brick_size'] = 0.8

        if recent_regime['volume'] == 'high':
            adjustment['volume_threshold'] = 1.3
            adjustment['dollar_threshold'] = 1.3
            adjustment['imbalance_threshold'] = 1.3
            adjustment['flow_threshold'] = 1.3
        elif recent_regime['volume'] == 'low':
            adjustment['volume_threshold'] = 0.7
            adjustment['dollar_threshold'] = 0.7
            adjustment['imbalance_threshold'] = 0.7
            adjustment['flow_threshold'] = 0.7

        return adjustment

    def _get_base_thresholds(self, symbol: str) -> Dict[str, float]:
        """Get base thresholds from configuration"""
        if symbol in self.base_config:
            return self.base_config[symbol].copy()
        return self.base_config.get('DEFAULT', {
            'volume_threshold': 50000,
            'dollar_threshold': 5000000,
            'imbalance_threshold': 25000,
            'flow_threshold': 30000,
            'volatility_threshold': 0.3,
            'range_size': 0.15,
            'brick_size': 0.05
        })

    def _load_base_config(self) -> Dict:
        """Load base configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Config file not found: {self.config_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _save_calibration_snapshot(self, symbol: str, calibration_data: Dict):
        """Save calibration snapshot for reproducibility"""
        try:
            snapshot_dir = Path("data/calibration_snapshots")
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            snapshot_file = snapshot_dir / f"{symbol}_{timestamp}.yaml"

            with open(snapshot_file, 'w') as f:
                yaml.dump(calibration_data, f, default_flow_style=False)

            logger.info(f"Saved calibration snapshot: {snapshot_file}")

        except Exception as e:
            logger.error(f"Error saving calibration snapshot: {e}")