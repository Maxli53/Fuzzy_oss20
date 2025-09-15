#!/usr/bin/env python3
"""
PyIQFeed Usage Analysis - Comprehensive Assessment
"""

# PyIQFeed Connection Types and Their Capabilities
PYIQFEED_CAPABILITIES = {
    "HistoryConn": {
        "methods": [
            "request_ticks",  # ✅ USING
            "request_ticks_for_days",  # ✅ USING
            "request_ticks_in_period",  # ✅ USING
            "request_bars",  # ✅ USING
            "request_bars_for_days",  # ✅ USING
            "request_bars_in_period",  # ✅ USING
            "request_daily_data",  # ✅ USING
            "request_daily_data_for_dates",  # ✅ USING
            "request_weekly_data",  # ❌ NOT USING
            "request_monthly_data",  # ❌ NOT USING
        ],
        "usage": "80%"
    },

    "LookupConn": {
        "methods": [
            "request_symbols_by_filter",  # ✅ USING
            "request_symbols_by_sic",  # ❌ NOT USING (SIC code lookup)
            "request_symbols_by_naic",  # ❌ NOT USING (NAIC code lookup)
            "request_futures_chain",  # ✅ USING
            "request_futures_spread_chain",  # ❌ NOT USING
            "request_futures_option_chain",  # ❌ NOT USING
            "request_equity_option_chain",  # ✅ USING
        ],
        "usage": "43%"
    },

    "QuoteConn": {
        "methods": [
            "watch",  # ✅ USING (real-time quotes)
            "unwatch",  # ✅ USING
            "request_watches",  # ✅ USING
            "unwatch_all",  # ✅ USING
        ],
        "usage": "100%"
    },

    "BarConn": {
        "methods": [
            "watch",  # ✅ USING (real-time bars)
            "unwatch",  # ✅ USING
            "request_watches",  # ✅ USING
        ],
        "usage": "100%"
    },

    "AdminConn": {
        "methods": [
            "register_client_app",  # ✅ USING (implicit in FeedService)
            "request_stats",  # ❌ NOT USING
            "set_log_levels",  # ❌ NOT USING
        ],
        "usage": "33%"
    },

    "NewsConn": {
        "methods": [
            "request_news_config",  # ✅ USING
            "request_news_headlines",  # ✅ USING
            "request_news_story",  # ✅ USING
            "request_story_counts",  # ❌ NOT USING
        ],
        "usage": "75%"
    },

    "TableConn": {
        "methods": [
            "connect",  # ❌ NOT USING (market depth/Level 2)
            "disconnect",  # ❌ NOT USING
        ],
        "usage": "0%"
    }
}

# Our IQFeedCollector Methods
OUR_METHODS = {
    "Historical Data": [
        "get_tick_data",  # HistoryConn.request_ticks
        "get_daily_data",  # HistoryConn.request_daily_data
        "get_daily_data_for_dates",  # HistoryConn.request_daily_data_for_dates
        "get_intraday_bars",  # HistoryConn.request_bars
        "get_bars_for_period",  # HistoryConn.request_bars_in_period
    ],

    "Real-time Streaming": [
        "stream_quotes",  # QuoteConn.watch
        "stream_bars",  # BarConn.watch
        "stop_streaming",  # unwatch methods
    ],

    "Lookups": [
        "search_symbols",  # LookupConn.request_symbols_by_filter
        "get_futures_chain",  # LookupConn.request_futures_chain
        "get_option_chain",  # LookupConn.request_equity_option_chain
    ],

    "News": [
        "get_news_headlines",  # NewsConn.request_news_headlines
        "get_news_story",  # NewsConn.request_news_story
    ],

    "Bulk Operations": [
        "collect_multiple_tickers_tick_data",
        "collect_multiple_tickers_daily_data",
    ]
}

# Features We're NOT Using
NOT_USING = {
    "Weekly/Monthly Data": [
        "request_weekly_data",
        "request_monthly_data"
    ],
    "Advanced Lookups": [
        "request_symbols_by_sic",  # Industry classification
        "request_symbols_by_naic",  # Insurance classification
    ],
    "Futures Advanced": [
        "request_futures_spread_chain",
        "request_futures_option_chain",
    ],
    "Market Depth": [
        "TableConn (Level 2 data)"
    ],
    "Administrative": [
        "request_stats",
        "set_log_levels",
    ],
    "News Analytics": [
        "request_story_counts"
    ]
}

def calculate_usage():
    """Calculate overall PyIQFeed usage percentage."""
    total_methods = 0
    using_methods = 0

    print("=" * 80)
    print("PYIQFEED USAGE ANALYSIS - DETAILED ASSESSMENT")
    print("=" * 80)

    for conn_type, info in PYIQFEED_CAPABILITIES.items():
        methods = info["methods"]
        total_methods += len(methods)
        used = sum(1 for m in methods if "✅" in str(m))
        using_methods += used

        print(f"\n{conn_type}: {info['usage']} usage")
        print(f"  Using {used}/{len(methods)} methods")

    overall_percentage = (using_methods / total_methods) * 100

    print("\n" + "=" * 80)
    print(f"OVERALL PYIQFEED USAGE: {overall_percentage:.1f}%")
    print(f"  Total PyIQFeed methods: {total_methods}")
    print(f"  Methods we're using: {using_methods}")
    print(f"  Methods not using: {total_methods - using_methods}")

    print("\n" + "=" * 80)
    print("KEY GAPS IN USAGE:")
    print("-" * 40)

    for category, items in NOT_USING.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  - {item}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("-" * 40)
    print("""
1. Weekly/Monthly Data (Easy Win):
   - Add get_weekly_data() and get_monthly_data()
   - Simple addition, same pattern as daily data

2. Market Depth (TableConn):
   - Add Level 2 data support for order book visibility
   - Critical for advanced trading strategies

3. Industry Classification Lookups:
   - Add SIC/NAIC code searches for sector analysis
   - Useful for portfolio screening

4. Advanced Futures:
   - Add spread chain and futures options support
   - Important for futures traders

5. Administrative Stats:
   - Add connection health monitoring
   - Useful for production monitoring
    """)

    print("=" * 80)
    print(f"FINAL ASSESSMENT: {overall_percentage:.1f}% PyIQFeed utilization")
    print("Good coverage of core features, room for specialized expansions")
    print("=" * 80)

if __name__ == "__main__":
    calculate_usage()