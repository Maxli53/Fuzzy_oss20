#!/usr/bin/env python3
"""
Test Foundation Models Integration with Real IQFeed Data
"""
import sys
import os
sys.path.append('.')
sys.path.append('pyiqfeed_orig')

import pyiqfeed as iq
import numpy as np
from foundation.utils.iqfeed_converter import convert_iqfeed_tick_to_pydantic

def test_foundation_integration():
    """Test foundation model conversion with real IQFeed data"""
    print('=' * 60)
    print('TESTING FOUNDATION MODELS WITH REAL IQFEED DATA')
    print('=' * 60)

    try:
        print('\n1. Testing PyIQFeed connection...')
        hist_conn = iq.HistoryConn(name='test-foundation')

        with iq.ConnConnector([hist_conn]) as connector:
            tick_data = hist_conn.request_ticks('AAPL', max_ticks=3)
            print(f'   Got {len(tick_data)} ticks from IQFeed')

            print('\n2. Testing tick data structure...')
            first_tick = tick_data[0]
            print(f'   Raw tick fields: {tick_data.dtype.names}')
            print(f'   Sample tick: {first_tick}')

            print('\n3. Testing Pydantic conversion...')
            # Convert to Pydantic model
            tick_model = convert_iqfeed_tick_to_pydantic(first_tick, 'AAPL')

            print(f'   Symbol: {tick_model.symbol}')
            print(f'   Price: ${tick_model.price}')
            print(f'   Size: {tick_model.size} shares')
            print(f'   Timestamp: {tick_model.timestamp}')
            print(f'   Exchange: {tick_model.exchange}')
            print(f'   Trade sign: {tick_model.trade_sign}')
            print(f'   Model type: {type(tick_model).__name__}')

            print('\n4. Testing all ticks conversion...')
            for i, tick in enumerate(tick_data):
                tick_model = convert_iqfeed_tick_to_pydantic(tick, 'AAPL')
                print(f'   Tick {i+1}: ${tick_model.price} ({tick_model.size} shares) at {tick_model.timestamp}')

            print('\n' + '=' * 60)
            print('SUCCESS: Foundation models working with real IQFeed data!')
            print('=' * 60)
            return True

    except Exception as e:
        print(f'\nERROR: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_foundation_integration()
    sys.exit(0 if success else 1)