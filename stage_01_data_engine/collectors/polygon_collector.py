"""
Polygon API Collector
Collects news, sentiment, analyst ratings, and earnings data from Polygon API
Complementary to IQFeed for comprehensive market intelligence
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
import requests
from dotenv import load_dotenv
import time

from stage_01_data_engine.core.base_collector import BaseCollector, StorageNamespace

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class PolygonCollector(BaseCollector):
    """
    Professional Polygon API collector for market intelligence data.

    Collects:
    - Real-time news with sentiment scores
    - Analyst ratings and price targets
    - Earnings data and guidance
    - Insider trading activity
    - Social sentiment indicators
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__("PolygonCollector", config)

        # Get API key from environment
        self.api_key = os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable not set")

        self.base_url = "https://api.polygon.io"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

        # Rate limiting (Polygon has 5 requests per minute for free tier)
        self.rate_limit_delay = 12  # seconds between requests for free tier
        self.last_request_time = 0

        logger.info("PolygonCollector initialized with API key")

    def collect(self, symbols: Union[str, List[str]], **kwargs) -> Optional[pd.DataFrame]:
        """
        Collect Polygon data for specified symbols.

        Args:
            symbols: Stock symbols to collect data for
            data_type: 'news', 'analyst_ratings', 'earnings', 'insider_trading'
            lookback_days: Number of days to look back for data

        Returns:
            DataFrame with collected data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        data_type = kwargs.get('data_type', 'news')
        lookback_days = kwargs.get('lookback_days', 7)

        if data_type == 'news':
            return self.collect_news_sentiment(symbols, lookback_days)
        elif data_type == 'analyst_ratings':
            return self.collect_analyst_ratings(symbols, lookback_days)
        elif data_type == 'earnings':
            return self.collect_earnings(symbols, lookback_days)
        elif data_type == 'insider_trading':
            return self.collect_insider_trading(symbols, lookback_days)
        else:
            logger.error(f"Invalid data_type: {data_type}")
            return None

    def collect_news_sentiment(self, symbols: List[str], lookback_days: int = 7) -> Optional[pd.DataFrame]:
        """Collect news articles with sentiment analysis"""
        all_news = []

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        for symbol in symbols:
            try:
                logger.info(f"Collecting news for {symbol}")

                # Rate limiting
                self._enforce_rate_limit()

                url = f"{self.base_url}/v2/reference/news"
                params = {
                    "ticker": symbol,
                    "published_utc.gte": start_date.strftime("%Y-%m-%d"),
                    "published_utc.lte": end_date.strftime("%Y-%m-%d"),
                    "order": "desc",
                    "limit": 50,
                    "apikey": self.api_key
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if 'results' in data:
                        for article in data['results']:
                            try:
                                news_item = {
                                    'timestamp': pd.to_datetime(article.get('published_utc')),
                                    'symbol': symbol,
                                    'title': article.get('title', ''),
                                    'description': article.get('description', ''),
                                    'author': article.get('author', ''),
                                    'url': article.get('article_url', ''),
                                    'publisher': article.get('publisher', {}).get('name', ''),
                                    'amp_url': article.get('amp_url', ''),
                                    'image_url': article.get('image_url', ''),
                                    'keywords': ', '.join(article.get('keywords', [])),
                                    'sentiment': self._analyze_sentiment(
                                        article.get('title', '') + ' ' + article.get('description', '')
                                    )
                                }
                                all_news.append(news_item)

                            except Exception as e:
                                logger.warning(f"Error processing news item for {symbol}: {e}")
                                continue

                        self.update_stats(len(data['results']), success=True)
                    else:
                        logger.warning(f"No news results for {symbol}")
                        self.update_stats(0, success=False)

                else:
                    logger.error(f"Error fetching news for {symbol}: {response.status_code}")
                    self.update_stats(0, success=False)

            except Exception as e:
                logger.error(f"Error collecting news for {symbol}: {e}")
                self.update_stats(0, success=False)
                continue

        if not all_news:
            return None

        df = pd.DataFrame(all_news)
        df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)

        return self._post_process_news(df)

    def collect_analyst_ratings(self, symbols: List[str], lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """Collect analyst ratings and price targets"""
        all_ratings = []

        # Note: Polygon analyst data might require premium subscription
        for symbol in symbols:
            try:
                logger.info(f"Collecting analyst ratings for {symbol}")

                self._enforce_rate_limit()

                # This endpoint might not be available in free tier
                url = f"{self.base_url}/v3/reference/tickers/{symbol}/events"
                params = {
                    "apikey": self.api_key,
                    "types": "earnings,dividend"  # Available event types
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if 'results' in data:
                        for event in data['results']:
                            try:
                                rating_item = {
                                    'timestamp': pd.to_datetime(event.get('event_date')),
                                    'symbol': symbol,
                                    'event_type': event.get('type', ''),
                                    'event_description': event.get('description', ''),
                                    'source': 'polygon',
                                    'rating': 'N/A',  # Placeholder - might need premium for actual ratings
                                    'price_target': None,
                                    'previous_rating': 'N/A'
                                }
                                all_ratings.append(rating_item)

                            except Exception as e:
                                logger.warning(f"Error processing rating for {symbol}: {e}")
                                continue

                        self.update_stats(len(data['results']), success=True)
                    else:
                        logger.warning(f"No analyst data for {symbol}")
                        self.update_stats(0, success=False)

                else:
                    logger.warning(f"Analyst ratings not available for {symbol} (might require premium)")
                    self.update_stats(0, success=False)

            except Exception as e:
                logger.error(f"Error collecting ratings for {symbol}: {e}")
                self.update_stats(0, success=False)
                continue

        if not all_ratings:
            return None

        return pd.DataFrame(all_ratings)

    def collect_earnings(self, symbols: List[str], lookback_days: int = 90) -> Optional[pd.DataFrame]:
        """Collect earnings data and guidance"""
        all_earnings = []

        for symbol in symbols:
            try:
                logger.info(f"Collecting earnings for {symbol}")

                self._enforce_rate_limit()

                # Get company financials (available in free tier)
                url = f"{self.base_url}/vX/reference/financials"
                params = {
                    "ticker": symbol,
                    "apikey": self.api_key,
                    "limit": 4  # Last 4 quarters
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if 'results' in data:
                        for financial in data['results']:
                            try:
                                earnings_item = {
                                    'timestamp': pd.to_datetime(financial.get('end_date')),
                                    'symbol': symbol,
                                    'period': financial.get('timeframe', ''),
                                    'fiscal_period': financial.get('fiscal_period', ''),
                                    'fiscal_year': financial.get('fiscal_year', ''),
                                    'start_date': pd.to_datetime(financial.get('start_date')),
                                    'end_date': pd.to_datetime(financial.get('end_date')),
                                    'filing_date': pd.to_datetime(financial.get('filing_date')),
                                    'acceptance_datetime': pd.to_datetime(financial.get('acceptance_datetime')),
                                    'financials': str(financial.get('financials', {})),  # JSON as string
                                    'source': 'polygon'
                                }
                                all_earnings.append(earnings_item)

                            except Exception as e:
                                logger.warning(f"Error processing earnings for {symbol}: {e}")
                                continue

                        self.update_stats(len(data['results']), success=True)
                    else:
                        logger.warning(f"No earnings data for {symbol}")
                        self.update_stats(0, success=False)

                else:
                    logger.error(f"Error fetching earnings for {symbol}: {response.status_code}")
                    self.update_stats(0, success=False)

            except Exception as e:
                logger.error(f"Error collecting earnings for {symbol}: {e}")
                self.update_stats(0, success=False)
                continue

        if not all_earnings:
            return None

        return pd.DataFrame(all_earnings)

    def collect_insider_trading(self, symbols: List[str], lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """Collect insider trading data"""
        all_insider_trades = []

        # Note: This might require premium subscription
        for symbol in symbols:
            try:
                logger.info(f"Collecting insider trades for {symbol}")

                self._enforce_rate_limit()

                # This endpoint might not be available in free tier
                logger.warning(f"Insider trading data for {symbol} might require premium subscription")

                # Placeholder for when premium access is available
                insider_item = {
                    'timestamp': pd.Timestamp.now(),
                    'symbol': symbol,
                    'insider_name': 'N/A',
                    'title': 'N/A',
                    'transaction_type': 'N/A',
                    'shares': 0,
                    'price': 0,
                    'value': 0,
                    'shares_owned_after': 0,
                    'source': 'polygon_placeholder'
                }
                all_insider_trades.append(insider_item)

            except Exception as e:
                logger.error(f"Error collecting insider trades for {symbol}: {e}")
                continue

        if not all_insider_trades:
            return None

        return pd.DataFrame(all_insider_trades)

    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis (placeholder for more sophisticated analysis)"""
        if not text:
            return 0.0

        # Simple keyword-based sentiment (could be replaced with proper NLP)
        positive_words = ['bullish', 'positive', 'growth', 'strong', 'beat', 'exceed', 'outperform', 'upgrade', 'buy', 'gains']
        negative_words = ['bearish', 'negative', 'decline', 'weak', 'miss', 'underperform', 'downgrade', 'sell', 'losses', 'risk']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count + negative_count == 0:
            return 0.0

        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        return sentiment_score

    def _post_process_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process news data with additional analytics"""
        if df.empty:
            return df

        try:
            # Add news volume indicators
            df['news_count_1d'] = df.groupby('symbol')['timestamp'].transform(
                lambda x: x.dt.date.value_counts().reindex(x.dt.date, fill_value=0)
            )

            # Add rolling sentiment scores
            for window in [3, 7, 14]:
                df[f'sentiment_ma_{window}d'] = df.groupby('symbol')['sentiment'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

            # Add sentiment volatility
            df['sentiment_volatility'] = df.groupby('symbol')['sentiment'].transform(
                lambda x: x.rolling(7, min_periods=2).std().fillna(0)
            )

            # Add news urgency score (based on how recent and sentiment strength)
            hours_since = (pd.Timestamp.now() - df['timestamp']).dt.total_seconds() / 3600
            urgency_decay = np.exp(-hours_since / 24)  # Exponential decay over 24 hours
            df['urgency_score'] = abs(df['sentiment']) * urgency_decay

            return df

        except Exception as e:
            logger.error(f"Error in news post-processing: {e}")
            return df

    def _enforce_rate_limit(self):
        """Enforce rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def validate(self, data: pd.DataFrame) -> bool:
        """Validate Polygon data"""
        try:
            if data.empty:
                return False

            required_cols = ['timestamp', 'symbol']
            for col in required_cols:
                if col not in data.columns:
                    logger.error(f"Missing required column: {col}")
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
        """Generate storage key for Polygon data"""
        data_type = kwargs.get('data_type', 'news')
        return StorageNamespace.polygon_key(f"{data_type}_{symbol}", date)

    def get_comprehensive_sentiment(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get comprehensive sentiment analysis combining all data types"""
        comprehensive_sentiment = {}

        for symbol in symbols:
            try:
                symbol_sentiment = {
                    'news_sentiment': None,
                    'news_volume': 0,
                    'analyst_sentiment': None,
                    'earnings_surprise': None,
                    'insider_activity': None,
                    'composite_score': 0.0
                }

                # News sentiment
                news_df = self.collect_news_sentiment([symbol], lookback_days=7)
                if news_df is not None and not news_df.empty:
                    symbol_sentiment['news_sentiment'] = news_df['sentiment'].mean()
                    symbol_sentiment['news_volume'] = len(news_df)

                # Analyst sentiment (if available)
                ratings_df = self.collect_analyst_ratings([symbol], lookback_days=30)
                if ratings_df is not None and not ratings_df.empty:
                    # Placeholder for analyst sentiment calculation
                    symbol_sentiment['analyst_sentiment'] = 0.0

                # Calculate composite score
                scores = []
                if symbol_sentiment['news_sentiment'] is not None:
                    scores.append(symbol_sentiment['news_sentiment'])

                if scores:
                    symbol_sentiment['composite_score'] = np.mean(scores)

                comprehensive_sentiment[symbol] = symbol_sentiment

            except Exception as e:
                logger.error(f"Error getting comprehensive sentiment for {symbol}: {e}")
                continue

        return comprehensive_sentiment