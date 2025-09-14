"""
Simple debug test - no Unicode
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

print("=== GUI DEBUG TEST ===")

try:
    print("1. Testing GUIDataInterface...")
    from gui.data_interface import GUIDataInterface

    data_interface = GUIDataInterface()
    print("   SUCCESS: Interface initialized")

    status = data_interface.get_connection_status()
    print(f"   Connection status: {status}")

    # Test symbol parsing
    print("2. Testing symbol parsing...")
    test_symbols = ['AAPL', 'MSFT']

    for symbol in test_symbols:
        try:
            result = data_interface.parse_symbol(symbol)
            if result['success']:
                print(f"   SUCCESS: {symbol} -> {result['parsed_info']['category']}")
            else:
                print(f"   ERROR: {symbol} -> {result['error']}")
        except Exception as e:
            print(f"   ERROR: {symbol} -> {e}")

    print("3. Testing DataEngine directly...")
    from stage_01_data_engine.core.data_engine import DataEngine

    engine = DataEngine()
    print("   SUCCESS: DataEngine initialized")
    print(f"   Stats: {engine.stats}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("=== DEBUG TEST COMPLETE ===")