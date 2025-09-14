"""
Test round-trip data flow after fixes
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from gui.data_interface import GUIDataInterface
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_round_trip():
    """Test the complete round-trip flow"""
    print("="*60)
    print("Testing Round-Trip Data Flow")
    print("="*60)

    # Initialize data interface
    print("\n1. Initializing GUIDataInterface...")
    try:
        interface = GUIDataInterface()
        print("   SUCCESS: GUIDataInterface initialized")
    except Exception as e:
        print(f"   ERROR: Failed to initialize: {e}")
        return False

    # Check connection status
    print("\n2. Checking connection status...")
    status = interface.get_connection_status()
    for component, connected in status.items():
        print(f"   {component}: {'Connected' if connected else 'Disconnected'}")

    # Test symbol parsing
    print("\n3. Testing symbol parsing for AAPL...")
    try:
        parse_result = interface.parse_symbol("AAPL")
        if parse_result['success']:
            print("   SUCCESS: Symbol parsed successfully")
            print(f"   Category: {parse_result['parsed_info']['category']}")
            print(f"   Subcategory: {parse_result['parsed_info']['subcategory']}")
        else:
            print(f"   ERROR: {parse_result['error']}")
    except Exception as e:
        print(f"   ERROR: Exception during parsing: {e}")

    # Test data fetching
    print("\n4. Testing data fetching for AAPL...")
    try:
        fetch_result = interface.fetch_real_data("AAPL", data_type="ticks", max_records=50)
        if fetch_result['success']:
            print("   SUCCESS: Data fetched successfully!")
            print(f"   Source: {fetch_result['source']}")
            print(f"   Records: {fetch_result['metadata']['records_count']}")
            print(f"   Date range: {fetch_result['metadata']['date_range']}")

            # Show sample data
            data = fetch_result['data']
            print("\n   Sample data (first 3 rows):")
            print(data.head(3))

            return data, interface
        else:
            print(f"   ERROR: {fetch_result['error']}")
            return None, interface
    except Exception as e:
        print(f"   ERROR: Exception during fetch: {e}")
        import traceback
        traceback.print_exc()
        return None, interface

def test_storage(data, interface):
    """Test storage functionality"""
    if data is None:
        print("\n5. SKIPPED: Storage test (no data to store)")
        return False

    print("\n5. Testing data storage...")
    try:
        store_result = interface.store_data("AAPL", data, "bars")
        if store_result['success']:
            print("   SUCCESS: Data stored successfully!")
            print(f"   Backend: {store_result['backend_used']}")
            print(f"   Records stored: {store_result['metadata']['records_stored']}")
            return True
        else:
            print(f"   ERROR: {store_result['error']}")
            return False
    except Exception as e:
        print(f"   ERROR: Exception during storage: {e}")
        return False

def test_retrieval(interface):
    """Test data retrieval"""
    print("\n6. Testing data retrieval...")
    try:
        retrieve_result = interface.retrieve_stored_data("AAPL")
        if retrieve_result['success']:
            print("   SUCCESS: Data retrieved successfully!")
            print(f"   Records retrieved: {retrieve_result['metadata']['records_retrieved']}")
            print(f"   Date range: {retrieve_result['metadata']['date_range']}")
            return True
        else:
            print(f"   ERROR: {retrieve_result['error']}")
            return False
    except Exception as e:
        print(f"   ERROR: Exception during retrieval: {e}")
        return False

if __name__ == "__main__":
    # Test fetching
    data, interface = test_round_trip()

    # Test storage
    storage_success = test_storage(data, interface)

    # Test retrieval
    if storage_success:
        retrieval_success = test_retrieval(interface)

        if retrieval_success:
            print("\n" + "="*60)
            print("ROUND-TRIP TEST: SUCCESS!")
            print("All components working correctly with real data")
            print("="*60)
        else:
            print("\nROUND-TRIP TEST: PARTIAL - Fetch and Storage work, Retrieval failed")
    else:
        if data is not None:
            print("\nROUND-TRIP TEST: PARTIAL - Fetch works, Storage failed")
        else:
            print("\nROUND-TRIP TEST: FAILED - Data fetch failed")