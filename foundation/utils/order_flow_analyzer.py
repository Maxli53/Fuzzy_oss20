"""
Order Flow Analysis Utilities

Advanced order flow metrics that can be computed without Level 2 data.
Based on market microstructure theory and López de Prado's work.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)


class OrderFlowAnalyzer:
    """Compute advanced order flow metrics from tick data"""

    @staticmethod
    def calculate_vpin(trades_df: pd.DataFrame,
                       bucket_size: int = 50,
                       window_size: int = 50) -> float:
        """
        Volume-Synchronized Probability of Informed Trading (VPIN)

        Measures order flow toxicity - the likelihood that market makers
        are trading with informed traders.

        Args:
            trades_df: DataFrame with columns ['size', 'trade_sign']
            bucket_size: Volume per bucket (e.g., 50 shares)
            window_size: Number of buckets for rolling average

        Returns:
            VPIN score [0, 1] where higher = more toxic flow
        """
        if trades_df.empty or 'size' not in trades_df.columns:
            return 0.0

        # Create volume buckets
        trades_df['cum_volume'] = trades_df['size'].cumsum()
        trades_df['bucket'] = (trades_df['cum_volume'] // bucket_size).astype(int)

        # Calculate buy/sell volumes per bucket
        bucket_imbalance = trades_df.groupby('bucket').apply(
            lambda x: abs(
                x[x['trade_sign'] > 0]['size'].sum() -
                x[x['trade_sign'] < 0]['size'].sum()
            ) if len(x) > 0 else 0
        )

        # VPIN = mean(|V_buy - V_sell|) / bucket_size
        if len(bucket_imbalance) >= window_size:
            vpin = bucket_imbalance.rolling(window_size).mean() / bucket_size
            return float(vpin.iloc[-1]) if not vpin.empty else 0.0

        return float(bucket_imbalance.mean() / bucket_size) if len(bucket_imbalance) > 0 else 0.0

    @staticmethod
    def calculate_kyle_lambda(prices: np.ndarray,
                             signed_volumes: np.ndarray) -> Optional[float]:
        """
        Kyle's Lambda - Measures permanent price impact per unit of net order flow

        Higher values indicate less liquid markets where trades have
        larger permanent price impact.

        Args:
            prices: Array of trade prices
            signed_volumes: Array of signed volumes (volume * trade_sign)

        Returns:
            Lambda coefficient (price impact per unit volume)
        """
        if len(prices) < 10:  # Need sufficient data
            return None

        try:
            # Convert to float to handle Decimal types
            prices = np.array(prices, dtype=float)
            signed_volumes = np.array(signed_volumes, dtype=float)

            # Calculate log price changes
            log_prices = np.log(prices)
            price_changes = np.diff(log_prices)

            # Cumulative signed volume (net order flow)
            cumulative_flow = np.cumsum(signed_volumes[:-1]).reshape(-1, 1)

            # Regression: Δp = λ * Σ(signed_volume) + ε
            model = LinearRegression()
            model.fit(cumulative_flow, price_changes)

            # Lambda is the regression coefficient
            kyle_lambda = float(model.coef_[0])

            # Sanity check - lambda should be positive and reasonable
            if kyle_lambda < 0 or kyle_lambda > 1:
                logger.warning(f"Unusual Kyle's lambda value: {kyle_lambda}")

            return kyle_lambda

        except Exception as e:
            logger.error(f"Error calculating Kyle's lambda: {e}")
            return None

    @staticmethod
    def calculate_roll_spread(prices: np.ndarray) -> float:
        """
        Roll's Effective Spread Estimator

        Estimates the effective spread from price changes alone,
        without needing quote data.

        Args:
            prices: Array of trade prices

        Returns:
            Estimated effective spread in price units
        """
        if len(prices) < 3:
            return 0.0

        # Convert to float to handle Decimal types
        prices = np.array(prices, dtype=float)
        price_changes = np.diff(prices)

        if len(price_changes) > 1:
            # Roll's measure: spread = 2 * sqrt(-cov(Δp_t, Δp_{t-1}))
            # Negative serial covariance indicates bid-ask bounce

            # Ensure we have arrays, not scalars
            changes_t = price_changes[:-1]
            changes_t_plus_1 = price_changes[1:]

            # Check if we have enough variation
            if len(changes_t) > 0 and np.std(changes_t) > 0 and np.std(changes_t_plus_1) > 0:
                covariance = np.cov(changes_t, changes_t_plus_1)[0, 1]

                if covariance < 0:
                    return 2 * np.sqrt(-covariance)

        return 0.0

    @staticmethod
    def calculate_trade_entropy(timestamps: np.ndarray,
                               window: int = 100) -> float:
        """
        Trade Intensity Entropy

        Measures the information content in trade arrival patterns.
        High entropy = more random/uninformed flow
        Low entropy = more predictable/potentially informed flow

        Args:
            timestamps: Array of trade timestamps (as floats/seconds)
            window: Number of recent trades to analyze

        Returns:
            Normalized entropy [0, 1]
        """
        if len(timestamps) < 2:
            return 0.5  # Neutral value for insufficient data

        # Ensure we're working with floats
        timestamps = np.array(timestamps, dtype=float)

        # Calculate inter-arrival times
        inter_arrivals = np.diff(timestamps[-window:])

        if len(inter_arrivals) < 2:
            return 0.5

        # Create histogram bins
        hist, _ = np.histogram(inter_arrivals, bins=min(10, len(inter_arrivals)))

        # Calculate Shannon entropy
        probs = hist / hist.sum()
        probs = probs[probs > 0]  # Remove zero probabilities

        if len(probs) <= 1:
            return 0.0  # No entropy if all in one bin

        entropy = -np.sum(probs * np.log(probs))

        # Normalize by maximum possible entropy
        max_entropy = np.log(len(probs))

        return entropy / max_entropy if max_entropy > 0 else 0.0

    @staticmethod
    def calculate_weighted_price_contribution(prices: np.ndarray,
                                            volumes: np.ndarray,
                                            signs: np.ndarray) -> Dict[str, float]:
        """
        Weighted Price Contribution (WPC)

        Decomposes price movement by trade direction to understand
        which side (buyers or sellers) is driving price.

        Args:
            prices: Array of trade prices
            volumes: Array of trade volumes
            signs: Array of trade signs (+1 buy, -1 sell)

        Returns:
            Dictionary with buy_wpc, sell_wpc, and dominance metrics
        """
        if len(prices) < 2:
            return {'buy_wpc': 0.0, 'sell_wpc': 0.0, 'dominance': 0.0}

        # Convert to float arrays to handle Decimal types
        prices = np.array(prices, dtype=float)
        volumes = np.array(volumes, dtype=float)
        signs = np.array(signs, dtype=float)

        # Calculate log returns
        log_returns = np.diff(np.log(prices))

        # Weight returns by volume and sign
        buy_mask = signs[:-1] > 0
        sell_mask = signs[:-1] < 0

        # Volume-weighted contributions
        buy_contribution = np.sum(log_returns * volumes[:-1] * buy_mask)
        sell_contribution = np.sum(log_returns * volumes[:-1] * sell_mask)

        # Total absolute price movement
        total_movement = np.sum(np.abs(log_returns * volumes[:-1]))

        if total_movement == 0:
            return {'buy_wpc': 0.0, 'sell_wpc': 0.0, 'dominance': 0.0}

        return {
            'buy_wpc': buy_contribution / total_movement,
            'sell_wpc': sell_contribution / total_movement,
            'dominance': (buy_contribution - abs(sell_contribution)) / total_movement
        }

    @staticmethod
    def calculate_amihud_illiquidity(prices: np.ndarray,
                                    volumes: np.ndarray,
                                    dollar_volumes: np.ndarray) -> float:
        """
        Amihud Illiquidity Ratio

        Measures price impact per dollar traded. Higher values indicate
        less liquid markets.

        Args:
            prices: Array of trade prices
            volumes: Array of trade volumes
            dollar_volumes: Array of dollar volumes (price * volume)

        Returns:
            Amihud illiquidity ratio
        """
        if len(prices) < 2:
            return 0.0

        # Convert to float arrays to handle Decimal types
        prices = np.array(prices, dtype=float)
        volumes = np.array(volumes, dtype=float)
        dollar_volumes = np.array(dollar_volumes, dtype=float)

        if dollar_volumes.sum() == 0:
            return 0.0

        # Calculate absolute returns
        returns = np.abs(np.diff(np.log(prices)))

        # Average |return| / dollar_volume
        # Use volumes[:-1] to match returns length
        dv = dollar_volumes[:-1]
        valid_mask = dv > 0

        if not valid_mask.any():
            return 0.0

        illiquidity = np.mean(returns[valid_mask] / dv[valid_mask])

        return float(illiquidity) * 1e6  # Scale for readability

    @staticmethod
    def calculate_flow_toxicity(trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        Comprehensive flow toxicity metrics

        Combines multiple indicators to assess the likelihood of
        adverse selection in the order flow.

        Args:
            trades_df: DataFrame with trade data

        Returns:
            Dictionary of toxicity metrics
        """
        metrics = {}

        # VPIN (primary toxicity measure)
        metrics['vpin'] = OrderFlowAnalyzer.calculate_vpin(trades_df)

        # Trade sign persistence (toxic flow tends to be one-sided)
        if 'trade_sign' in trades_df.columns and len(trades_df) > 1:
            sign_changes = np.diff(trades_df['trade_sign'].values)
            metrics['sign_persistence'] = 1 - (np.abs(sign_changes).sum() / (2 * len(sign_changes)))
        else:
            metrics['sign_persistence'] = 0.5

        # Volume concentration (toxic flow often comes in bursts)
        if 'size' in trades_df.columns and len(trades_df) > 10:
            volumes = trades_df['size'].values
            top_10_pct = int(len(volumes) * 0.1)
            top_volume = np.sort(volumes)[-top_10_pct:].sum()
            metrics['volume_concentration'] = top_volume / volumes.sum()
        else:
            metrics['volume_concentration'] = 0.1

        # Composite toxicity score
        metrics['toxicity_score'] = (
            metrics['vpin'] * 0.5 +
            metrics['sign_persistence'] * 0.3 +
            metrics['volume_concentration'] * 0.2
        )

        return metrics

    @staticmethod
    def compute_all_metrics(trades_df: pd.DataFrame) -> Dict[str, any]:
        """
        Compute all available order flow metrics

        Args:
            trades_df: DataFrame with columns:
                - price: Trade price
                - size: Trade volume
                - trade_sign: +1 for buy, -1 for sell
                - timestamp: Trade timestamp
                - dollar_volume: Price * volume

        Returns:
            Dictionary of all computed metrics
        """
        metrics = {}

        # Extract arrays for efficiency
        if len(trades_df) > 0:
            prices = trades_df['price'].values if 'price' in trades_df else np.array([])
            volumes = trades_df['size'].values if 'size' in trades_df else np.array([])
            signs = trades_df['trade_sign'].values if 'trade_sign' in trades_df else np.array([])
            timestamps = trades_df['timestamp'].values if 'timestamp' in trades_df else np.array([])
            dollar_volumes = trades_df['dollar_volume'].values if 'dollar_volume' in trades_df else np.array([])

            # Signed volumes for Kyle's lambda
            signed_volumes = volumes * signs if len(volumes) > 0 and len(signs) > 0 else np.array([])

            # Individual metrics
            metrics['vpin'] = OrderFlowAnalyzer.calculate_vpin(trades_df)
            metrics['kyle_lambda'] = OrderFlowAnalyzer.calculate_kyle_lambda(prices, signed_volumes)
            metrics['roll_spread'] = OrderFlowAnalyzer.calculate_roll_spread(prices)

            # Convert timestamps to seconds if datetime
            if len(timestamps) > 0 and hasattr(timestamps[0], 'timestamp'):
                timestamps = np.array([t.timestamp() for t in timestamps])
            metrics['trade_entropy'] = OrderFlowAnalyzer.calculate_trade_entropy(timestamps)

            # Weighted price contribution
            wpc = OrderFlowAnalyzer.calculate_weighted_price_contribution(prices, volumes, signs)
            metrics.update(wpc)

            # Amihud illiquidity
            metrics['amihud_illiquidity'] = OrderFlowAnalyzer.calculate_amihud_illiquidity(
                prices, volumes, dollar_volumes
            )

            # Flow toxicity suite
            toxicity = OrderFlowAnalyzer.calculate_flow_toxicity(trades_df)
            metrics.update({f'toxicity_{k}': v for k, v in toxicity.items()})

        return metrics