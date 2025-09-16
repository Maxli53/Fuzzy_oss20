#!/usr/bin/env python3
"""
SHOW RAW NUMPY DATA FROM IQFEED
"""

import sys
import os
sys.path.insert(0, 'stage_01_data_engine/collectors')
sys.path.append('pyiqfeed_orig')

from iqfeed_collector import IQFeedCollector
import numpy as np

collector = IQFeedCollector()
if not collector.ensure_connection():
    print("FAILED TO CONNECT TO IQFEED")
    sys.exit(1)

# Get AAPL ticks
ticks = collector.get_tick_data('AAPL', num_days=1, max_ticks=10000)

print('='*100)
print('    RAW NUMPY STRUCTURED ARRAY - DIRECTLY FROM IQFEED')
print('='*100)
print(f'\nTotal ticks fetched: {len(ticks)}')
print(f'Array dtype: {ticks.dtype}')
print(f'Field names: {ticks.dtype.names}')
print()

# Show raw array slice
print('RAW NUMPY ARRAY[:5]:')
print(repr(ticks[:5]))
print()

# Filter for regular hours
regular = []
for t in ticks:
    time_us = int(t['time'].astype('int64')) if isinstance(t['time'], np.timedelta64) else int(t['time'])
    hours = time_us // 3600000000
    minutes = (time_us % 3600000000) // 60000000
    total_minutes = hours * 60 + minutes

    # 9:30 AM = 570 minutes, 4:00 PM = 960 minutes
    if 570 <= total_minutes <= 960 and t['cond1'] != 135:
        regular.append(t)

if regular:
    regular_ticks = np.array(regular, dtype=ticks.dtype)
    print(f'FILTERED: {len(regular_ticks)} regular hours ticks (9:30 AM - 4:00 PM, excluding cond1=135)')
else:
    print('NO REGULAR HOURS TICKS FOUND - SHOWING ALL TICKS')
    regular_ticks = ticks

print('\n' + '='*100)
print('    FIRST 15 TICKS - ALL 14 FIELDS')
print('='*100)

for i in range(min(15, len(regular_ticks))):
    t = regular_ticks[i]
    time_us = int(t['time'].astype('int64')) if isinstance(t['time'], np.timedelta64) else int(t['time'])
    h = time_us // 3600000000
    m = (time_us % 3600000000) // 60000000
    s = (time_us % 60000000) // 1000000
    us = time_us % 1000000

    print(f'\n--- TICK {i+1} ---')
    print(f'Complete Raw Tuple: {t}')
    print('Field Breakdown:')
    print(f'  [0]  tick_id:    {t["tick_id"]} (IQFeed internal number - NOT unique)')
    print(f'  [1]  date:       {t["date"]} ')
    print(f'  [2]  time:       {t["time"]} = {h:02d}:{m:02d}:{s:02d}.{us:06d} ET')
    print(f'  [3]  last:       ${t["last"]:.4f} (trade price)')
    print(f'  [4]  last_sz:    {t["last_sz"]} (trade size/volume)')
    print(f'  [5]  last_type:  {t["last_type"]} (exchange code)')
    print(f'  [6]  mkt_ctr:    {t["mkt_ctr"]} (market center)')
    print(f'  [7]  tot_vlm:    {t["tot_vlm"]:,} (cumulative daily volume)')
    print(f'  [8]  bid:        ${t["bid"]:.4f} (best bid)')
    print(f'  [9]  ask:        ${t["ask"]:.4f} (best ask)')
    print(f'  [10] cond1:      {t["cond1"]} (condition code 1)')
    print(f'  [11] cond2:      {t["cond2"]} (condition code 2)')
    print(f'  [12] cond3:      {t["cond3"]} (condition code 3)')
    print(f'  [13] cond4:      {t["cond4"]} (condition code 4)')

print('\n' + '='*100)
print(f'SUMMARY: Showing {min(15, len(regular_ticks))} of {len(regular_ticks)} regular hours ticks')
print('='*100)