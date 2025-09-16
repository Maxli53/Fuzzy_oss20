#!/usr/bin/env python3
"""
Debug script for metadata storage in ArcticDB
"""

from arcticdb import Arctic
import pandas as pd
from datetime import datetime

# Connect to ArcticDB
arctic = Arctic("lmdb://C:/ArcticDB/tick_data")

# Create or get library
try:
    lib = arctic['tick_data']
except:
    lib = arctic.create_library('tick_data')
    print("Created tick_data library")

# Create test data
df = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-15 09:30:00', periods=100, freq='1s'),
    'price': [100.0 + i * 0.01 for i in range(100)],
    'volume': [100 + i for i in range(100)]
})

# Create test metadata
test_metadata = {
    'basic_info': {
        'symbol': 'TEST',
        'date': '2025-01-15',
        'count': 100
    },
    'tier2_metadata': {
        'basic_stats': {
            'total_ticks': 100,
            'vwap': 100.50
        },
        'spread_stats': {
            'mean_bps': 1.5
        }
    }
}

# Store with metadata
storage_key = 'TEST/2025-01-15'
print(f"Storing data with metadata to {storage_key}...")
lib.write(storage_key, df, metadata=test_metadata)
print("Stored successfully")

# Read back just metadata
print("\nReading metadata...")
metadata = lib.read_metadata(storage_key)
print(f"Type of metadata: {type(metadata)}")
print(f"Metadata content: {metadata}")

# Extract tier2_metadata
if isinstance(metadata, dict):
    tier2 = metadata.get('tier2_metadata')
    print(f"\nTier 2 metadata: {tier2}")
else:
    print(f"\nMetadata is not a dict, it's: {type(metadata)}")

# Clean up
lib.delete(storage_key)
print("\nTest data deleted")