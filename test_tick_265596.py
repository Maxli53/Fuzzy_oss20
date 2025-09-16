#!/usr/bin/env python3
"""
Test TICK 265596 Complete Transformation
NumPy → Pydantic with all fields
"""

import sys
import os
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('pyiqfeed_orig')
sys.path.append('.')

from iqfeed_collector import IQFeedCollector
import numpy as np
from foundation.utils.iqfeed_converter import convert_iqfeed_tick_to_pydantic
import json
from decimal import Decimal

# Custom JSON encoder for Decimal
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

collector = IQFeedCollector()
if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

# Get AAPL ticks
ticks = collector.get_tick_data('AAPL', num_days=1, max_ticks=10000)

# Find TICK 265596
found_tick = None
for t in ticks:
    if t['tick_id'] == 265596:
        found_tick = t
        break

if found_tick is None:
    print("TICK 265596 not found in fetched data")
    sys.exit(1)

print("="*100)
print("TICK 265596 COMPLETE TRANSFORMATION TEST")
print("="*100)

print("\n1. RAW NUMPY DATA:")
print("-"*80)
print(f"Raw tuple: {found_tick}")
print("\nField breakdown:")
for i, name in enumerate(found_tick.dtype.names):
    value = found_tick[i]
    if isinstance(value, np.timedelta64):
        time_us = int(value.astype('int64'))
        h = time_us // 3600000000
        m = (time_us % 3600000000) // 60000000
        s = (time_us % 60000000) // 1000000
        us = time_us % 1000000
        print(f"  [{i:2}] {name:10}: {value} = {h:02d}:{m:02d}:{s:02d}.{us:06d} ET")
    else:
        print(f"  [{i:2}] {name:10}: {value}")

print("\n2. PYDANTIC TRANSFORMATION:")
print("-"*80)

# Convert to Pydantic
try:
    pydantic_tick = convert_iqfeed_tick_to_pydantic(found_tick, "AAPL")

    # Get all fields
    tick_dict = pydantic_tick.model_dump()

    print(f"Total fields in Pydantic model: {len(tick_dict)}")
    print("\nAll fields:")

    # Group fields by category
    core_fields = ['symbol', 'timestamp', 'price', 'size', 'exchange', 'market_center',
                   'total_volume', 'bid', 'ask', 'conditions']
    spread_fields = ['spread', 'midpoint', 'spread_bps', 'spread_pct', 'effective_spread']
    trade_fields = ['trade_sign', 'dollar_volume', 'price_improvement']
    condition_flags = ['is_regular', 'is_extended_hours', 'is_odd_lot', 'is_intermarket_sweep',
                      'is_derivatively_priced', 'is_qualified', 'is_block_trade']
    other_fields = [k for k in tick_dict.keys() if k not in core_fields + spread_fields + trade_fields + condition_flags]

    print("\nCORE FIELDS (from NumPy):")
    for field in core_fields:
        if field in tick_dict:
            value = tick_dict[field]
            if field == 'timestamp':
                print(f"  {field:20}: {value} (TZ: {value.tzinfo if hasattr(value, 'tzinfo') else 'N/A'})")
            else:
                print(f"  {field:20}: {value}")

    print("\nSPREAD METRICS:")
    for field in spread_fields:
        if field in tick_dict and tick_dict[field] is not None:
            print(f"  {field:20}: {tick_dict[field]}")

    print("\nTRADE ANALYSIS:")
    for field in trade_fields:
        if field in tick_dict and tick_dict[field] is not None:
            print(f"  {field:20}: {tick_dict[field]}")

    print("\nCONDITION FLAGS:")
    for field in condition_flags:
        if field in tick_dict:
            print(f"  {field:20}: {tick_dict[field]}")

    if other_fields:
        print("\nOTHER FIELDS:")
        for field in other_fields:
            if tick_dict[field] is not None:
                print(f"  {field:20}: {tick_dict[field]}")

    # Validation checks
    print("\n3. VALIDATION CHECKS:")
    print("-"*80)

    # Check transformation correctness
    checks = []

    # Price check
    if float(pydantic_tick.price) == found_tick['last']:
        checks.append("[OK] Price matches")
    else:
        checks.append(f"[FAIL] Price mismatch: {pydantic_tick.price} vs {found_tick['last']}")

    # Size check
    if pydantic_tick.size == found_tick['last_sz']:
        checks.append("[OK] Size matches")
    else:
        checks.append(f"[FAIL] Size mismatch: {pydantic_tick.size} vs {found_tick['last_sz']}")

    # Spread calculation check
    if pydantic_tick.bid and pydantic_tick.ask:
        expected_spread = float(pydantic_tick.ask) - float(pydantic_tick.bid)
        if abs(float(pydantic_tick.spread) - expected_spread) < 0.0001:
            checks.append("[OK] Spread calculation correct")
        else:
            checks.append(f"[FAIL] Spread mismatch: {pydantic_tick.spread} vs {expected_spread}")

    # Condition flags check
    cond1 = found_tick['cond1']
    cond2 = found_tick['cond2']
    cond3 = found_tick['cond3']

    # Check is_derivatively_priced (should be False since cond2=0)
    if not pydantic_tick.is_derivatively_priced:
        checks.append("[OK] is_derivatively_priced correct (cond2=0)")
    else:
        checks.append(f"[FAIL] is_derivatively_priced wrong (cond2={cond2})")

    # Check is_odd_lot (should be False since cond3=0)
    if not pydantic_tick.is_odd_lot:
        checks.append("[OK] is_odd_lot correct (cond3=0)")
    else:
        checks.append(f"[FAIL] is_odd_lot wrong (cond3={cond3})")

    # Check price_improvement for buy trade
    if pydantic_tick.trade_sign == 1 and pydantic_tick.price_improvement is not None:
        checks.append(f"[OK] price_improvement calculated: {pydantic_tick.price_improvement}")

    for check in checks:
        print(f"  {check}")

    print("\n4. SUMMARY:")
    print("-"*80)
    print(f"NumPy fields: 14")
    print(f"Pydantic fields: {len([v for v in tick_dict.values() if v is not None])} non-null / {len(tick_dict)} total")
    print(f"Enhancement ratio: {len(tick_dict)/14:.1f}x")

except Exception as e:
    print(f"ERROR during conversion: {e}")
    import traceback
    traceback.print_exc()