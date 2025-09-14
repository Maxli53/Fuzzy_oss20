"""
Debug test for GUI functionality
Test individual components outside of Streamlit
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_data_interface():
    """Test the data interface directly"""
    print("Testing GUIDataInterface...")

    try:
        from gui.data_interface import GUIDataInterface

        # Initialize interface
        data_interface = GUIDataInterface()
        print(f"SUCCESS: GUIDataInterface initialized")

        # Test connection status
        status = data_interface.get_connection_status()
        print(f"Connection status: {status}")

        # Test symbol parsing
        print("\nTesting symbol parsing...")
        symbols_to_test = ['AAPL', '@ES#', 'EUR/USD', '$TICK']

        for symbol in symbols_to_test:
            try:
                result = data_interface.parse_symbol(symbol)
                if result['success']:
                    print(f"✅ {symbol}: {result['parsed_info']['category']}")
                else:
                    print(f"❌ {symbol}: {result['error']}")
            except Exception as e:
                print(f"❌ {symbol}: Error - {e}")

        return True

    except Exception as e:
        print(f"❌ GUIDataInterface failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_symbol_parser_widget():
    """Test symbol parser widget components"""
    print("\nTesting symbol parser widget...")

    try:
        from gui.components.symbol_parser_widget import display_symbol_parse_results

        # Create a mock parse result
        mock_result = {
            'success': True,
            'symbol': 'AAPL',
            'parsed_info': {
                'category': 'equity',
                'subcategory': 'stock',
                'base_symbol': 'AAPL',
                'exchange': 'NASDAQ',
                'instrument_type': 'stock',
                'is_valid': True
            },
            'routing_recommendation': 'flexible',
            'error': None
        }

        print(f"✅ Symbol parser widget components loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Symbol parser widget failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test all critical imports"""
    print("Testing critical imports...")

    imports_to_test = [
        ('pandas', 'pd'),
        ('streamlit', 'st'),
        ('plotly.express', 'px'),
        ('plotly.graph_objects', 'go'),
        ('gui.data_interface', 'GUIDataInterface'),
        ('stage_01_data_engine.core.data_engine', 'DataEngine'),
        ('stage_01_data_engine.parsers.dtn_symbol_parser', 'DTNSymbolParser')
    ]

    for module_name, import_as in imports_to_test:
        try:
            if import_as == 'GUIDataInterface':
                from gui.data_interface import GUIDataInterface
                print(f"✅ {module_name} -> {import_as}")
            elif import_as == 'DataEngine':
                from stage_01_data_engine.core.data_engine import DataEngine
                print(f"✅ {module_name} -> {import_as}")
            elif import_as == 'DTNSymbolParser':
                from stage_01_data_engine.parsers.dtn_symbol_parser import DTNSymbolParser
                print(f"✅ {module_name} -> {import_as}")
            elif import_as == 'pd':
                import pandas as pd
                print(f"✅ {module_name} -> {import_as}")
            elif import_as == 'st':
                import streamlit as st
                print(f"✅ {module_name} -> {import_as}")
            elif import_as == 'px':
                import plotly.express as px
                print(f"✅ {module_name} -> {import_as}")
            elif import_as == 'go':
                import plotly.graph_objects as go
                print(f"✅ {module_name} -> {import_as}")

        except Exception as e:
            print(f"❌ {module_name}: {e}")

def test_data_engine_direct():
    """Test DataEngine initialization directly"""
    print("\nTesting DataEngine directly...")

    try:
        from stage_01_data_engine.core.data_engine import DataEngine

        engine = DataEngine()
        print(f"✅ DataEngine initialized")
        print(f"Stats: {engine.stats}")

        # Test symbol parser
        from stage_01_data_engine.parsers.dtn_symbol_parser import DTNSymbolParser
        parser = DTNSymbolParser()

        test_symbols = ['AAPL', '@ES#', 'EUR/USD']
        for symbol in test_symbols:
            try:
                parsed = parser.parse_symbol(symbol)
                print(f"✅ {symbol}: {parsed.category}")
            except Exception as e:
                print(f"❌ {symbol}: {e}")

        return True

    except Exception as e:
        print(f"❌ DataEngine direct test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("DEBUG: GUI Debug Test Starting...\n")

    # Test 1: Critical imports
    test_imports()

    # Test 2: DataEngine direct
    test_data_engine_direct()

    # Test 3: Data interface
    test_data_interface()

    # Test 4: Widget components
    test_symbol_parser_widget()

    print("\nDEBUG: Debug test completed!")