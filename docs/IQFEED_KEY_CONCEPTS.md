# IQFeed Integration - Key Concepts Reference

## Overview
This document summarizes the key concepts discovered and documented during our IQFeed integration work, particularly focusing on the NumPy array to DataFrame conversion pipeline and the weekend tick data advantage.

## 1. Data Flow Architecture

```
IQFeed API → PyIQFeed → NumPy Structured Array → TickStore → DataFrame → ArcticDB
```

### Key Components:
- **IQFeedCollector**: Returns raw NumPy structured arrays (no DataFrame conversion)
- **TickStore**: Handles NumPy → DataFrame conversion with field preservation
- **ArcticDB**: Final storage requiring DataFrame format

## 2. NumPy Structured Array Format

IQFeed returns tick data as NumPy structured arrays with 14 fields:

```python
Fields: ('tick_id', 'date', 'time', 'last', 'last_sz', 'last_type',
         'mkt_ctr', 'tot_vlm', 'bid', 'ask', 'cond1', 'cond2', 'cond3', 'cond4')
```

### Field Meanings:
- **tick_id**: Unique identifier for the tick
- **date**: Date as datetime64[D] (e.g., '2025-09-15')
- **time**: Microseconds since midnight ET as timedelta64[us]
- **last**: Trade price
- **last_sz**: Trade volume (shares)
- **last_type**: Exchange code as bytes (b'O'=NYSE Arca, b'Q'=NASDAQ)
- **mkt_ctr**: Market center numeric ID
- **tot_vlm**: Cumulative daily volume
- **bid/ask**: Best bid/ask at time of trade
- **cond1-4**: Trade condition codes

## 3. Time Handling

### Critical Discovery:
IQFeed splits timestamp into two fields that must be combined:
```python
# Correct conversion:
timestamp = pd.to_datetime(date) + pd.to_timedelta(time_microseconds)
# Example: '2025-09-15' + 26540953311μs = '2025-09-15 07:22:20.953311'
```

### Common Error:
Initially displayed times incorrectly (e.g., "05:42 Sunday" instead of "07:22 Monday") due to improper time conversion.

## 4. Weekend Tick Data Advantage

### Discovery:
IQFeed has different data limits based on request timing:

| Request Time | Tick Data Limit | Reason |
|-------------|----------------|---------|
| Market Hours (9:30-16:00 ET weekdays) | 8 days | Server load management |
| After Hours/Weekends | 180 days | Lower server demand |

### Implementation:
```python
if is_weekend or is_after_hours:
    # Use request_ticks_in_period() for up to 180 days
    tick_data = hist_conn.request_ticks_in_period(ticker, start, end, max_ticks)
else:
    # Use request_ticks() limited to 8 days
    tick_data = hist_conn.request_ticks(ticker, max_ticks)
```

## 5. Trade Condition Codes

Common condition codes found in cond1-4 fields:

| Code | Meaning |
|------|---------|
| 135 | Extended hours trade |
| 23 | Odd lot (< 100 shares) |
| 61 | Trade qualifier |
| 0 | Regular trade |

## 6. Exchange Codes

Exchange codes come as bytes and need decoding:

| Code | Exchange |
|------|----------|
| b'O' | NYSE Arca |
| b'Q' | NASDAQ |
| b'N' | NYSE |
| b'A' | NYSE American |
| b'C' | NSX |
| b'D' | FINRA ADF |

## 7. Storage Layer Solution

### Problem:
Mismatch between IQFeedCollector (returns NumPy) and TickStore (expects DataFrame)

### Solution Chosen:
Update storage layer to accept NumPy arrays directly:

```python
# New method in TickStore:
def store_numpy_ticks(self, symbol: str, date: str, tick_array: np.ndarray, ...):
    tick_df = self._numpy_ticks_to_dataframe(tick_array)
    return self.store_ticks(symbol, date, tick_df, metadata, overwrite)
```

### Field Preservation:
All 14 original fields are preserved plus 2 derived fields:
- **spread**: ask - bid (liquidity indicator)
- **midpoint**: (bid + ask) / 2 (fair value estimate)

## 8. File Structure

### Key Files Modified:
1. **stage_01_data_engine/storage/tick_store.py**
   - Added store_numpy_ticks() method
   - Added _numpy_ticks_to_dataframe() converter
   - Preserves all 14 fields + calculates spread/midpoint

2. **stage_01_data_engine/collectors/iqfeed_collector.py**
   - Returns raw NumPy arrays
   - Implements weekend advantage logic
   - No DataFrame conversion (keeps it efficient)

## 9. Testing Approach

### Verification Steps:
1. Collect tick data as NumPy array
2. Convert to DataFrame via storage layer
3. Verify all 14 fields preserved
4. Check timestamp accuracy
5. Confirm trade conditions decoded

### Test Files Created:
- **test_simple_comments.py**: Basic IQFeed functionality test
- **show_aapl_ticks.py**: Display tick structure and fields
- **verify_all_fields.py**: Confirm field preservation

## 10. Best Practices Learned

1. **Always preserve original data**: Keep all 14 fields even if not immediately used
2. **Document field meanings**: Trade conditions and exchange codes need explanation
3. **Test time conversions**: Easy to get wrong, especially with timezones
4. **Use weekend advantage**: Schedule heavy data downloads for weekends
5. **Keep NumPy arrays**: Don't convert to DataFrame until necessary (performance)

## 11. PyIQFeed Coverage Improvement

Started at 73% utilization, targeting 85% by adding:
- Weekly/monthly data methods
- Industry classification lookups
- News analytics integration
- Administrative monitoring

Current focus has been on fixing the core tick data pipeline first before expanding to additional features.

## Conclusion

The main achievement was discovering and fixing the systematic mismatch between IQFeedCollector (NumPy arrays) and TickStore (DataFrames), while preserving all data fields and discovering the 180-day weekend advantage for tick data collection.