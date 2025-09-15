"""
Test Script - Retrieve and display 22 AAPL daily closing prices
"""
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage_01_data_engine.collectors.iqfeed_collector import IQFeedCollector
import logging

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Test the IQFeed connection and retrieve AAPL closing prices"""
    print("="*60)
    print("Stage 1 Data Engine Test - AAPL Closing Prices")
    print("="*60)

    try:
        # Create data collector
        collector = IQFeedCollector()

        # Test with AAPL for 22 trading days
        symbol = "AAPL"
        num_days = 22

        print(f"\nTesting connection to IQFeed...")
        print(f"Attempting to retrieve {num_days} daily closing prices for {symbol}")
        print("\nThis will connect to your existing IQFeed instance...")
        print("Make sure IQFeed is running and logged in with your credentials.")

        # Attempt to get and display the data
        success = collector.display_closing_prices(symbol, num_days)

        if success:
            print(f"\n[SUCCESS] Successfully retrieved real {symbol} data from IQFeed")
            print("Stage 1 Data Engine is working correctly!")
        else:
            print(f"\n[FAILED] to retrieve {symbol} data")
            print("Check that IQFeed is running and credentials are correct")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"\n[ERROR] Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure IQFeed client is running")
        print("2. Verify you're logged in with credentials: 487854/t1wnjnuz")
        print("3. Check that IQFeed is connected to data servers")

    print("="*60)


if __name__ == "__main__":
    main()