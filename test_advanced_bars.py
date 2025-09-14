"""
Test all advanced bar types and user workflow
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from gui.data_interface import GUIDataInterface
import logging

# Setup minimal logging
logging.basicConfig(level=logging.WARNING)

def test_user_workflow():
    """Test complete user workflow: Days → Bar Type → Frequency → Market Hours"""
    print("Testing Complete User Workflow")
    print("="*50)

    # Initialize interface
    interface = GUIDataInterface()
    symbol = "AAPL"

    # Test scenarios based on user workflow
    test_scenarios = [
        # Scenario 1: Time-based bars with market hours
        {
            'name': 'Market Hours 1-minute bars',
            'bar_type': '1m',
            'lookback_days': 1,
            'market_hours_only': True,
            'include_premarket': False,
            'include_afterhours': False
        },
        # Scenario 2: Tick-based bars
        {
            'name': '50-tick bars',
            'bar_type': 'tick_50',
            'lookback_days': 1,
            'market_hours_only': True,
            'include_premarket': False,
            'include_afterhours': False
        },
        # Scenario 3: Volume bars
        {
            'name': '1000-share volume bars',
            'bar_type': 'volume_1000',
            'lookback_days': 1,
            'market_hours_only': True,
            'include_premarket': False,
            'include_afterhours': False
        },
        # Scenario 4: Raw ticks (most recent)
        {
            'name': 'Recent tick data',
            'bar_type': 'ticks',
            'lookback_days': 0,  # Use most recent
            'market_hours_only': False,  # Get latest available
            'include_premarket': False,
            'include_afterhours': False
        }
    ]

    results = {}

    for scenario in test_scenarios:
        print(f"\n--- Testing: {scenario['name']} ---")

        try:
            # Test direct collector access for advanced types
            if scenario['bar_type'] in ['ticks']:
                # Use interface for basic ticks
                result = interface.fetch_real_data(
                    symbol=symbol,
                    data_type=scenario['bar_type'],
                    max_records=100
                )
            else:
                # Use collector directly for advanced bar types
                collector_data = interface.data_engine.iqfeed_collector.collect_bars(
                    symbols=[symbol],
                    bar_type=scenario['bar_type'],
                    lookback_days=scenario['lookback_days'],
                    market_hours_only=scenario['market_hours_only'],
                    include_premarket=scenario['include_premarket'],
                    include_afterhours=scenario['include_afterhours'],
                    max_ticks=100
                )

                if collector_data is not None and not collector_data.empty:
                    result = {
                        'success': True,
                        'data': collector_data,
                        'records_count': len(collector_data)
                    }
                else:
                    result = {'success': False, 'error': 'No data returned'}

            if result['success']:
                data = result['data']
                print(f"   SUCCESS: Got {len(data)} {scenario['bar_type']} records")
                print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                print(f"   Columns: {list(data.columns)}")

                # Show sample
                if len(data) > 0:
                    print(f"   Sample: {data.iloc[0][['timestamp', 'price' if 'price' in data.columns else 'close', 'volume']].to_dict()}")

                results[scenario['name']] = 'SUCCESS'
            else:
                print(f"   FAILED: {result['error']}")
                results[scenario['name']] = 'FAILED'

        except Exception as e:
            print(f"   ERROR: {str(e)}")
            results[scenario['name']] = 'ERROR'

    # Summary
    print(f"\n" + "="*50)
    print("USER WORKFLOW TEST SUMMARY:")
    print("="*50)
    for scenario, status in results.items():
        status_icon = "✓" if status == 'SUCCESS' else "✗"
        print(f"{status_icon} {scenario}: {status}")

    success_count = sum(1 for status in results.values() if status == 'SUCCESS')
    total_count = len(results)
    print(f"\nOverall: {success_count}/{total_count} scenarios working")

    if success_count == total_count:
        print("🎉 ALL USER WORKFLOW SCENARIOS WORKING!")
    elif success_count > 0:
        print("⚠️ PARTIAL SUCCESS - Some scenarios working")
    else:
        print("❌ NO SCENARIOS WORKING - Need debugging")

if __name__ == "__main__":
    test_user_workflow()