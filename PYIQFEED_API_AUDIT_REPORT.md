# PyIQFeed API Comprehensive Audit Report

## Executive Summary

This report documents a comprehensive audit of all available API endpoints and methods in the PyIQFeed library. The audit identified **26 total request methods** across **8 connection classes**, providing complete coverage of IQFeed's data retrieval capabilities.

## Connection Classes Overview

PyIQFeed provides 8 main connection classes for different types of market data and operations:

1. **HistoryConn** - Historical data (bars, ticks, daily/weekly/monthly data)
2. **LookupConn** - Symbol lookups and option chains
3. **QuoteConn** - Real-time quotes and market data streaming
4. **BarConn** - Real-time bar data streaming
5. **NewsConn** - News headlines and stories
6. **AdminConn** - Administrative functions and client management
7. **TableConn** - Market reference data tables
8. **FeedConn** - Base feed connection functionality

## Detailed API Method Analysis

### 1. HistoryConn (10 request methods)
**Primary use:** Historical market data retrieval

#### Bar Data Methods (4 methods):
- `request_bars()` - Request historical bars with specific interval
- `request_bars_for_days()` - Request bars for specified number of days
- `request_bars_in_period()` - Request bars within a date range
- `request_daily_data()` - Request daily OHLCV data

#### Tick Data Methods (3 methods):
- `request_ticks()` - Request most recent tick data
- `request_ticks_for_days()` - Request ticks for specified number of days
- `request_ticks_in_period()` - Request ticks within a date range

#### Time-Series Data Methods (3 methods):
- `request_daily_data_for_dates()` - Daily data for specific date range
- `request_weekly_data()` - Weekly aggregated data
- `request_monthly_data()` - Monthly aggregated data

**Key Parameters:**
- `ticker`: Symbol to retrieve data for
- `interval_len` & `interval_type`: Bar interval (e.g., 60 seconds)
- `max_ticks`/`max_bars`: Limit number of data points
- `bgn_flt`/`end_flt`: Time filters for market hours
- `ascend`: Sort order (default descending)

### 2. LookupConn (9 request methods)
**Primary use:** Symbol discovery and option chains

#### Symbol Discovery Methods (3 methods):
- `request_symbols_by_filter()` - Search symbols by text filter
- `request_symbols_by_naic()` - Find symbols by NAIC industry code
- `request_symbols_by_sic()` - Find symbols by SIC industry code

#### Option Chain Methods (4 methods):
- `request_equity_option_chain()` - Equity option chains
- `request_futures_option_chain()` - Futures option chains
- `request_futures_chain()` - Futures contract chains
- `request_futures_spread_chain()` - Futures spread chains

#### Market Data Methods (2 methods):
- `request_5MD()` - 5-minute delayed market data
- `request_FDS()` - Fundamental data service

### 3. QuoteConn (2 request methods)
**Primary use:** Real-time quote streaming with administrative functions

#### Administrative Methods:
- `request_stats()` - Connection statistics
- `request_watches()` - List watched symbols

#### Stream Management Methods (non-request):
- `watch()` / `unwatch()` - Subscribe/unsubscribe to symbols
- `regional_watch()` - Regional quote data
- `trades_watch()` - Trade data streaming
- `news_on()` / `news_off()` - News feed control

### 4. NewsConn (4 request methods)
**Primary use:** Financial news data

#### News Data Methods:
- `request_news_headlines()` - Get news headlines with filters
- `request_news_story()` - Retrieve full story by ID
- `request_story_counts()` - Count stories by symbol/source
- `request_news_config()` - News service configuration

**Key Features:**
- Filter by sources, symbols, dates
- Limit number of results
- Email story functionality

### 5. BarConn (1 request method)
**Primary use:** Real-time bar data streaming

#### Administrative Method:
- `request_watches()` - List subscribed symbols

#### Stream Management (non-request):
- `watch()` - Subscribe to bar updates with interval specification
- `unwatch()` / `unwatch_all()` - Unsubscribe from bars

### 6. AdminConn (0 request methods)
**Primary use:** Administrative functions and client management

#### Configuration Methods (non-request):
- `set_admin_variables()` - Configure credentials
- `register_client_app()` / `remove_client_app()` - App management
- `client_stats_on()` / `client_stats_off()` - Statistics monitoring
- `save_login_info()` - Credential persistence

### 7. TableConn (0 request methods)
**Primary use:** Reference data tables

#### Data Retrieval Methods (non-request):
- `get_markets()` - Available markets
- `get_security_types()` - Security type definitions
- `get_naic_codes()` / `get_sic_codes()` - Industry classifications
- `get_trade_conditions()` - Trade condition codes
- `update_tables()` - Refresh reference data

### 8. FeedConn (0 request methods)
**Primary use:** Base connection functionality

Provides core connection management methods inherited by other classes.

## Common Connection Methods

All connection classes inherit these standard methods:
- `connect()` / `disconnect()` - Connection management
- `connected()` - Connection status
- `add_listener()` / `remove_listener()` - Event listeners
- `start_runner()` / `stop_runner()` - Background processing

## API Usage Patterns

### Historical Data Pattern:
```python
service = iq.FeedService(product="APP", version="1.0", login=user, password=pwd)
service.launch(headless=True)

hist_conn = iq.HistoryConn(name="historical")
hist_conn.connect()

# Get tick data
ticks = hist_conn.request_ticks("AAPL", max_ticks=100)

# Get bar data
bars = hist_conn.request_bars("AAPL", interval_len=60, interval_type='s', max_bars=100)
```

### Real-time Data Pattern:
```python
quote_conn = iq.QuoteConn(name="quotes")
quote_conn.add_listener(my_listener)
quote_conn.connect()
quote_conn.watch("AAPL")  # Start receiving quotes
```

### Symbol Discovery Pattern:
```python
lookup_conn = iq.LookupConn(name="lookup")
lookup_conn.connect()

# Find technology stocks
symbols = lookup_conn.request_symbols_by_filter("technology", search_field='d')

# Get option chain
options = lookup_conn.request_equity_option_chain("AAPL", opt_type='pc')
```

## Known Issues and Limitations

Based on testing, several issues were identified:

1. **Tick Data Issues:**
   - `request_ticks()` fails with array ambiguity error
   - `request_ticks_for_days()` returns "NO_DATA" errors
   - `request_ticks_in_period()` has incorrect parameter names

2. **Bar Data Issues:**
   - `request_bars_for_days()` returns "NO_DATA" errors for recent data
   - May require market hours or different symbols for testing

3. **Documentation:**
   - Many methods lack proper docstrings
   - Parameter documentation is minimal

## Recommendations

1. **For Historical Data:**
   - Use `request_daily_data()` for reliable daily data
   - Test tick methods with different symbols or date ranges
   - Implement proper error handling for "NO_DATA" responses

2. **For Real-time Data:**
   - Implement proper listeners for Quote and Bar connections
   - Use watch/unwatch methods to manage subscriptions
   - Monitor connection status and implement reconnection logic

3. **For Symbol Discovery:**
   - LookupConn methods appear most reliable
   - Use for building symbol universes and option chain analysis

4. **Testing Strategy:**
   - Test with different symbols (some may have limited historical data)
   - Verify market hours and data availability
   - Implement comprehensive error handling

## Total API Surface

- **8 Connection Classes**
- **26 Request Methods** (data retrieval)
- **50+ Additional Methods** (streaming, configuration, utilities)
- **Complete Coverage** of IQFeed functionality

This audit provides a comprehensive map of all PyIQFeed capabilities for building robust market data applications.