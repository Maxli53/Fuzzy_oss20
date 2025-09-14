"""
IQFeed Connector - Connects to existing IQFeed instance
"""
import pyiqfeed as iq
import os
from datetime import datetime
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IQFeedConnector:
    """Manages connection to existing IQFeed instance"""

    def __init__(self):
        self.service: Optional[iq.FeedService] = None
        self.history_conn: Optional[iq.HistoryConn] = None
        self.is_connected = False

        # Get credentials from environment
        self.username = os.getenv('IQFEED_USERNAME', '487854')
        self.password = os.getenv('IQFEED_PASSWORD', 't1wnjnuz')

    def connect(self) -> bool:
        """Connect to existing IQFeed instance"""
        try:
            logger.info("Attempting to connect to existing IQFeed instance...")

            # Create FeedService - this will detect existing instance
            self.service = iq.FeedService(
                product="PLACEHOLDER_ID",  # Can use placeholder for existing instance
                version="1.0",             # Can use placeholder
                login=self.username,
                password=self.password
            )

            # This will connect to existing instance, not launch new one
            logger.info("Launching/connecting to IQFeed service...")
            self.service.launch(headless=True)

            logger.info("Successfully connected to IQFeed!")
            self.is_connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to IQFeed: {e}")
            self.is_connected = False
            return False

    def get_history_connection(self) -> Optional[iq.HistoryConn]:
        """Get history connection for data requests"""
        if not self.is_connected:
            logger.error("Not connected to IQFeed. Call connect() first.")
            return None

        try:
            if self.history_conn is None:
                logger.info("Creating history connection...")
                self.history_conn = iq.HistoryConn(name="fuzzy-oss20-history")
                logger.info("History connection created successfully")

            return self.history_conn

        except Exception as e:
            logger.error(f"Failed to create history connection: {e}")
            return None

    def disconnect(self):
        """Clean disconnect"""
        if self.history_conn:
            self.history_conn = None
        if self.service:
            self.service = None
        self.is_connected = False
        logger.info("Disconnected from IQFeed")

    def get_lookup_connection(self) -> Optional[iq.LookupConn]:
        """Get lookup connection for DTN symbol lookups"""
        if not self.is_connected:
            logger.error("Not connected to IQFeed. Call connect() first.")
            return None

        try:
            logger.info("Creating lookup connection...")
            lookup_conn = iq.LookupConn(name="fuzzy-oss20-lookup")
            logger.info("Lookup connection created successfully")
            return lookup_conn

        except Exception as e:
            logger.error(f"Failed to create lookup connection: {e}")
            return None

    def test_connection(self) -> bool:
        """Test if connection is working"""
        if not self.is_connected:
            return False

        try:
            # Try to get a history connection as a test
            hist_conn = self.get_history_connection()
            return hist_conn is not None
        except:
            return False