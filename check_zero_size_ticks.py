#!/usr/bin/env python3
"""
Check the zero-size ticks in IQFeed data
"""

import sys
import numpy as np

# Setup paths
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('.')

from iqfeed_collector import IQFeedCollector

# Initialize collector
collector = IQFeedCollector()

if not collector.ensure_connection():
    print("Failed to connect to IQFeed")
    sys.exit(1)

print("Fetching AAPL ticks...")
tick_array = collector.get_tick_data('AAPL', num_days=1, max_ticks=100000)

print(f"Total ticks fetched: {len(tick_array)}")

# Check for zero-size ticks
# Field 4 is 'last_sz' (size)
zero_size_indices = np.where(tick_array['last_sz'] == 0)[0]

print(f"\nFound {len(zero_size_indices)} ticks with size=0")
print(f"Indices: {zero_size_indices[:10]}...")  # Show first 10

# Show details of first few zero-size ticks
for i, idx in enumerate(zero_size_indices[:5]):
    tick = tick_array[idx]
    print(f"\n--- Tick at index {idx} ---")
    print(f"  tick_id:   {tick['tick_id']}")
    print(f"  date:      {tick['date']}")
    print(f"  time:      {tick['time']}")
    print(f"  last:      ${tick['last']:.4f}")
    print(f"  last_sz:   {tick['last_sz']} <- ZERO SIZE")
    print(f"  last_type: {tick['last_type']}")
    print(f"  mkt_ctr:   {tick['mkt_ctr']}")
    print(f"  tot_vlm:   {tick['tot_vlm']:,}")
    print(f"  bid:       ${tick['bid']:.4f}")
    print(f"  ask:       ${tick['ask']:.4f}")
    print(f"  conditions: [{tick['cond1']}, {tick['cond2']}, {tick['cond3']}, {tick['cond4']}]")

# Check what exchange/type these are
print("\nAnalyzing zero-size ticks by exchange type:")
for idx in zero_size_indices[:20]:
    tick = tick_array[idx]
    print(f"  Index {idx}: last_type='{tick['last_type']}', mkt_ctr={tick['mkt_ctr']}, cond1={tick['cond1']}")