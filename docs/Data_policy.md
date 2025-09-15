# Fuzzy OSS20 Data Policy

## User-Controlled Data Collection Workflow

### Step 1: Trading Days Selection
- User specifies number of trading days to download
- Respects market hours policy (8 days max during market, 180 days after close)

### Step 2: Bar Type Selection
**Time-Based Bars**: ticks, 1s, 5s, 1m, 5m, 15m, 1h, daily
**Advanced Bars**: tick_N, volume_N, dollar_N, imbalance, volatility, range, renko

### Step 3: Frequency/Threshold Selection
- **Time bars**: interval selection (1m, 5m, etc.)
- **Tick bars**: N ticks (10, 50, 100, 200, 500, 1000)
- **Volume bars**: N shares (1K, 5K, 10K, 50K, 100K)
- **Dollar bars**: $N amount ($1K, $10K, $50K, $100K, $500K, $1M)
- **Imbalance bars**: imbalance threshold (10%, 20%, 30%)
- **Volatility bars**: volatility accumulation threshold
- **Range bars**: price movement points (0.1, 0.5, 1.0, 2.0)
- **Renko bars**: brick size (0.1, 0.25, 0.5, 1.0)

### Step 4: Market Hours (Default)
- **Default**: Regular hours only (09:30-16:00 ET)
- **Optional**: Include pre-market (04:00-09:30 ET)
- **Optional**: Include after-hours (16:00-20:00 ET)
- **Advanced**: Custom time range selection

### Data Quality Standards
- **Institutional Smart Fallback**: Weekend/holiday requests automatically use last trading day
- **NO_DATA Error Elimination**: Period-based PyIQFeed requests prevent weekend data failures
- **NO synthetic substitutes**: If adjusted data unavailable, we wait/retry, never substitute fake data

### Historical Data Limits
- **During market hours**: 8 trading days maximum
- **After market close**: 180 calendar days maximum
- **Rationale**: Balance storage costs with analytical needs

## Data Sources Architecture

### IQFeed (85% PyIQFeed API Utilization - Production Ready)
```
stage_01_data_engine/
├── collectors/
│   └── iqfeed_collector.py        # Enhanced PyIQFeed implementation (825+ lines)
│       ├── Historical Data:       # Tick, daily, weekly, monthly, intraday bars
│       ├── Real-time Streaming:   # Live quotes, trades, regional, bars
│       ├── Lookups:              # Option chains, futures chains, symbol search, SIC/NAIC
│       ├── Reference Data:       # Markets, security types, trade conditions
│       ├── News Feeds:           # Headlines, full stories, story counts analytics
│       └── Administrative:       # Connection stats, log levels, health monitoring
├── session_manager.py             # Market hours & trading day logic
├── iqfeed_constraints.py          # IQFeed limitations (simplified, no custom rate limiting)
├── storage/
│   ├── tick_storage.py           # ArcticDB tick data
│   ├── indicator_storage.py      # ArcticDB DTN indicators
│   └── bar_builder.py            # All bar type construction
└── config/
    └── indicator_config.yaml     # DTN symbol mappings
```

### IQFeed Capabilities (85% PyIQFeed Coverage)
- **Historical Data (100%)**:
  - Tick data with 180-day weekend optimization
  - Daily, weekly, monthly OHLCV data
  - Intraday bars with smart fallback
  - Period-based requests with weekend/holiday handling
- **Real-time Streaming (100%)**:
  - QuoteConn, BarConn with VerboseListeners
  - Complete watch/unwatch functionality
- **Market Lookups (71%)**:
  - Equity options, futures chains
  - Symbol search by filter
  - NEW: SIC/NAIC industry classification lookups
- **News Integration (100%)**:
  - Headlines, full stories
  - NEW: Story counts analytics for sentiment
- **Administrative (100%)**:
  - NEW: Connection health monitoring
  - NEW: Dynamic log level configuration
  - Client statistics and diagnostics
- **Business Logic**:
  - Session-aware collection
  - Native PyIQFeed rate limiting (no custom implementation)
  - Asset class detection
  - Smart date fallback to last trading day

### Not Implemented (15% - Specialized Features)
- **TableConn**: Level 2 market depth (not needed for current strategies)
- **Futures Advanced**: Spread chains, options on futures (niche instruments)
- **5MD/FDS**: Specialized futures data (not required)

### Arctic Storage Architecture
Complete structure based on DTNCalculatedIndicators.pdf analysis:

```
arctic_storage/
├── iqfeed/
│   ├── base/
│   │   ├── tick_50/{symbol}/{date}
│   │   ├── time_5s/{symbol}/{date}
│   │   └── daily/{symbol}/{date}
│   │
│   ├── dtn_calculated/
│   │   ├── equities_index/          # Category 1: Equities/Index Stats
│   │   │   ├── issues/              # Page 2 - TINT.Z, TIQT.Z (Total Issues)
│   │   │   ├── volume/              # Page 3 - VINT.Z, VIQT.Z (Market Volume)
│   │   │   ├── tick/                # Page 4 - JTNT.Z, JTQT.Z (Net Tick)
│   │   │   ├── trin/                # Page 5 - RINT.Z, RIQT.Z (TRIN Index)
│   │   │   ├── highs_lows/          # Page 6 - H1NH.Z, H1NL.Z (New Highs/Lows)
│   │   │   ├── avg_price/           # Page 7 - Average Price Indicators
│   │   │   ├── moving_avg/          # Page 8 - M506V.Z, M2006V.Z (Moving Averages)
│   │   │   ├── premium/             # Page 9 - PREM.Z, PRNQ.Z (Market Premium)
│   │   │   ├── ratio/               # Page 10 - Market Ratios
│   │   │   └── net/                 # Page 11 - Net Indicators
│   │   │
│   │   ├── options/                 # Category 2: Options Stats
│   │   │   ├── tick/                # Page 12 - TCOEA.Z, TPOEA.Z (Options Tick)
│   │   │   ├── issues/              # Page 13 - ICOEA.Z, IPOEA.Z (Options Issues)
│   │   │   ├── open_interest/       # Page 14 - OCOET.Z, OPOET.Z (Open Interest)
│   │   │   ├── volume/              # Page 15 - VCOET.Z, VPOET.Z (Options Volume)
│   │   │   ├── trin/                # Page 16 - SCOET.Z, SPOET.Z (Options TRIN)
│   │   │   └── grok_derived/        # 20 Grok calculations from raw options data
│   │   │       ├── pcr/             # Put-Call Ratio = VPOET.Z / VCOET.Z
│   │   │       ├── dollar_pcr/      # Dollar PCR = DPOET.Z / DCOET.Z
│   │   │       ├── oi_pcr/          # OI PCR = OPOET.Z / OCOET.Z
│   │   │       ├── net_tick_sentiment/
│   │   │       ├── volume_spread/
│   │   │       ├── sizzle_indices/
│   │   │       ├── gamma_flow/
│   │   │       ├── dark_pool_sentiment/
│   │   │       ├── institutional_flow/
│   │   │       ├── retail_sentiment/
│   │   │       ├── volatility_skew/
│   │   │       ├── term_structure/
│   │   │       ├── momentum_indicators/
│   │   │       ├── contrarian_signals/
│   │   │       ├── fear_greed_index/
│   │   │       ├── liquidity_metrics/
│   │   │       ├── smart_money_flow/
│   │   │       ├── vix_structure/
│   │   │       └── cross_asset_signals/
│   │   │
│   │   └── metadata/
│   │       ├── symbol_mappings.json  # Maps readable names to DTN symbols
│   │       └── thresholds.json       # Alert thresholds for each indicator
│   │
│   └── derived/                      # Bar types derived from base data
│       ├── volume_bars/
│       ├── dollar_bars/
│       ├── imbalance_bars/
│       ├── volatility_bars/
│       ├── range_bars/
│       └── renko_bars/
│
├── polygon/
│   ├── news/{symbol}/{date}
│   ├── analyst_ratings/{symbol}/{date}
│   ├── earnings/{symbol}/{date}
│   └── insider_trading/{symbol}/{date}
│
└── calibration/
    └── adaptive_thresholds/{symbol}/{date}
```

### Key DTN Symbol Mappings

#### Equities/Index Statistics (Raw from IQFeed)
- **Issues**: TINT.Z (NYSE Total), TIQT.Z (NASDAQ Total)
- **Volume**: VINT.Z (NYSE Volume), VIQT.Z (NASDAQ Volume)
- **Tick**: JTNT.Z (NYSE Net Tick), JTQT.Z (NASDAQ Net Tick)
- **TRIN**: RINT.Z (NYSE TRIN), RIQT.Z (NASDAQ TRIN), RI6T.Z (S&P500), RI1T.Z (DOW)
- **Highs/Lows**: H1NH.Z, H1NL.Z (1-day), H30NH.Z, H30NL.Z (30-day)
- **Moving Averages**: M506V.Z (S&P500 above 50MA), M2006V.Z (above 200MA)
- **Premium**: PREM.Z (S&P Premium), PRNQ.Z (NASDAQ Premium), PRYM.Z (DOW)

#### Options Statistics (Raw from IQFeed)
- **Volume**: VCOET.Z (Call Volume), VPOET.Z (Put Volume)
- **Dollar Volume**: DCOET.Z (Call $Volume), DPOET.Z (Put $Volume)
- **Open Interest**: OCOET.Z (Call OI), OPOET.Z (Put OI)
- **Tick**: TCOEA.Z (Call Advances), TPOEA.Z (Put Advances)
- **Issues**: ICOEA.Z (Call Issues Adv), IPOEA.Z (Put Issues Adv)
- **TRIN**: SCOET.Z (Call TRIN), SPOET.Z (Put TRIN)

#### Grok Derived Metrics (Calculated from Options Raw)
1. **PCR** = VPOET.Z / VCOET.Z
2. **Dollar PCR** = DPOET.Z / DCOET.Z
3. **OI PCR** = OPOET.Z / OCOET.Z
4. **Net Tick Sentiment** = (TCOEA.Z - TCOED.Z) - (TPOEA.Z - TPOED.Z)
5. **Volume Spread** = VCOET.Z - VPOET.Z
6. **Sizzle Index** = Current Volume / 20-day Average Volume
7. **Gamma Flow** = Delta-adjusted volume flow
8. **Dark Pool Sentiment** = Off-exchange volume analysis
9. **Institutional Flow** = Large block transaction analysis
10. **Retail Sentiment** = Small lot transaction analysis
11. **Volatility Skew** = IV skew across strikes
12. **Term Structure** = IV across expiration dates
13. **Momentum Indicators** = Volume-price momentum
14. **Contrarian Signals** = Extreme sentiment reversals
15. **Fear Greed Index** = Composite sentiment score
16. **Liquidity Metrics** = Bid-ask spread analysis
17. **Smart Money Flow** = Unusual options activity
18. **VIX Structure** = VIX term structure analysis
19. **Cross Asset Signals** = Bond/equity/currency correlations
20. **Flow Imbalance** = Directional flow analysis

### Sector SPDRs Integration
- **XLK** (Technology), **XLF** (Financial), **XLE** (Energy)
- **XLV** (Healthcare), **XLI** (Industrial), **XLB** (Materials)
- **XLP** (Consumer Staples), **XLY** (Consumer Discretionary), **XLU** (Utilities)

### Polygon API (Complementary Data)
```
stage_01_data_engine/
├── collectors/
│   └── polygon_collector.py
└── storage/
    └── news_storage.py
```

**Polygon Data Types**:
- Real-time news with sentiment scores
- Analyst ratings and price targets
- Social sentiment indicators
- Insider trading data
- Financial statements updates

## Bar Type Specifications

### Supported Bar Types
- **Tick Bars**: Every N ticks (default: 50)
- **Volume Bars**: Every N shares traded
- **Dollar Bars**: Every $N traded
- **Imbalance Bars**: When buy/sell imbalance exceeds threshold
- **Volatility Bars**: When price volatility accumulates
- **Range Bars**: When price moves N points from last close
- **Renko Bars**: Fixed price movement bricks

### Bar Construction Rules
1. All bars built from base data (50-tick or 5-second bars)
2. OHLCV + metadata (num_ticks, vwap, volume_profile)
3. Bar-specific metadata (thresholds, imbalance ratios, etc.)
4. Timestamp = completion time of bar

## Data Quality Standards

### Collection Parameters
- **Update frequency**: 1 second during market hours
- **Batch size**: 100 indicators per request
- **Retry attempts**: 3 with exponential backoff
- **Timeout**: 30 seconds per request
- **Stale data threshold**: 60 seconds

### Validation Rules
- Range validation for all indicators
- Outlier detection (3-sigma rule)
- Missing data tolerance: 5% maximum
- Cross-validation between related indicators

## Market Hours & Collection Schedule

### Market Hours (Eastern Time)
- **Pre-market**: 04:00 - 09:30
- **Regular hours**: 09:30 - 16:00
- **After-hours**: 16:00 - 20:00

### Collection Schedule
- **Breadth indicators**: All sessions
- **Options sentiment**: Regular hours only
- **Moving averages**: End of day calculation
- **Market premium**: Regular hours only
- **Bulk downloads**: After market close (16:30-17:00)

## Alert Thresholds

### Market Stress Indicators
```yaml
NYSE_TICK:
  extreme_positive: 1500
  extreme_negative: -1500

NYSE_TRIN:
  oversold: 2.0
  overbought: 0.5
  extreme_oversold: 3.0
  extreme_overbought: 0.3

TOTAL_PC_RATIO:
  extreme_fear: 1.5
  extreme_greed: 0.5
```

## Storage & Performance

### ArcticDB Configuration
- **Single database**: All data in one ArcticDB instance
- **Compression**: LZ4 for optimal performance
- **Indexing**: Symbol + timestamp composite keys
- **Partitioning**: By symbol and date for query optimization

### Data Retention
- **Tick data**: 8 trading days (market hours) / 180 days (after hours)
- **Daily bars**: 5 years
- **DTN indicators**: 2 years
- **News/sentiment**: 1 year

## Implementation Priority

### Phase 1: Core Infrastructure
1. Fix connector.py (add get_lookup_connection)
2. Create .env with credentials
3. Implement DTNCalculatedCollector
4. Test data loading verification

### Phase 2: Enhanced Collection
1. Implement PolygonCollector
2. Add daily bulk downloader
3. Create comprehensive bar builder tests
4. Implement alert thresholds

### Phase 3: Optimization
1. Performance tuning for ArcticDB
2. Advanced error handling and recovery
3. Data quality monitoring dashboard
4. Automated calibration snapshots

## API Credentials
```bash
# .env file
IQFEED_USERNAME=487854
IQFEED_PASSWORD=t1wnjnuz
POLYGON_API_KEY=lXc6yVaITFsV9S1pZ8L9u30yMbiV5VOi
```

## No Fallback Policy
- **Never substitute synthetic data**
- **Never mix data sources for same metric**
- **Fail fast and retry rather than degrade quality**
- **Alert immediately on data source failures**

## Exploratory Quantitative Research Mode

### Dynamic Symbol Discovery
The system now supports **exploratory mode** where ANY symbol can be fetched and stored without preconfiguration:

```python
# Fetch ANY symbol on the fly
engine = DataEngine()
data = engine.fetch_any(['AAPL', 'JTNT.Z', 'ESU24', 'EURUSD', 'UNKNOWN_TICKER'])

# Explore unknown symbols before fetching
exploration = engine.explore(['MYSTERIOUS_SYMBOL'], deep_analysis=True)

# Discover new symbols from news mentions
new_symbols = engine.discover_new_symbols(method='news')
```

### Flexible Storage Architecture

#### Auto-Categorization System
Based on DTN symbol patterns, the system automatically routes symbols to appropriate storage:

```
Flexible Arctic Storage:
├── iqfeed_equity_stocks/          # AAPL, MSFT, TSLA
├── iqfeed_equity_etfs/            # SPY, QQQ, XLK
├── iqfeed_dtn_sentiment/          # JTNT.Z, RINT.Z (market sentiment)
├── iqfeed_dtn_options/            # TCOEA.Z, VCOET.Z (options flow)
├── iqfeed_options_aapl/           # AAPL240315C00150000
├── iqfeed_futures_es/             # ESU24, ESZ24
├── iqfeed_forex/                  # EURUSD, GBPJPY
└── iqfeed_unknown_xyz/            # Symbols not yet categorized
```

#### Dynamic Namespace Generation
Storage paths are generated automatically based on symbol characteristics:

- **Equities**: `iqfeed/equity/stocks/AAPL/ticks/2024-01-15`
- **DTN Indicators**: `iqfeed/dtn/sentiment/JTNT.Z/indicators/2024-01-15`
- **Options**: `iqfeed/options/AAPL/AAPL240315C00150000/quotes/2024-01-15`
- **Futures**: `iqfeed/futures/ES/ESU24/ticks/2024-01-15`
- **Forex**: `iqfeed/forex/EUR/EURUSD/ticks/2024-01-15`

### Symbol Discovery Pipeline

#### 1. Pattern Recognition
Uses DTN Calculated Indicators PDF patterns to identify:
- **DTN Indicators** (.Z suffix): JTNT.Z → NYSE Net Tick
- **Options** (OCC format): AAPL240315C00150000 → AAPL Call $150 exp 3/15/24
- **Futures** (Month codes): ESU24 → E-mini S&P Sept 2024
- **Forex** (Currency pairs): EURUSD → Euro/Dollar

#### 2. Smart Storage Routing
Automatically determines:
- **Library name**: Based on category and subcategory
- **Namespace**: Hierarchical path for organization
- **Metadata**: Exchange, expiration, strike price, etc.
- **Retention policy**: Based on data type and importance

#### 3. Universe Management
Tracks discovered symbols for research expansion:
- **Symbol registry**: All discovered symbols with metadata
- **Access patterns**: Most frequently accessed data
- **Related symbols**: Suggestions based on analysis
- **Universe statistics**: Diversity and coverage metrics

### Data Quality in Exploratory Mode

#### Validation Without Preconfiguration
- **Pattern-based validation**: Use symbol type to determine expected data ranges
- **Statistical outlier detection**: 3-sigma rule across all data types
- **Cross-validation**: Check related symbols for consistency
- **Graceful degradation**: Store with quality flags if uncertain

#### Metadata Enrichment
Every stored symbol includes comprehensive metadata:
```yaml
symbol_metadata:
  symbol: "AAPL"
  category: "equity"
  subcategory: "common_stock"
  exchange: "NASDAQ"
  storage_namespace: "iqfeed/equity/stocks"
  discovery_timestamp: "2024-01-15T10:30:00Z"
  access_count: 42
  related_symbols: ["MSFT", "GOOGL", "QQQ"]
  data_quality_score: 0.95
```

### Research Workflow Integration

#### Exploratory Research Pattern
```python
# 1. Discover interesting symbols
engine = DataEngine()
news_symbols = engine.discover_new_symbols(method='news')

# 2. Explore before committing resources
for symbol in news_symbols:
    exploration = engine.explore([symbol])
    if exploration[symbol]['sample_data_info']['data_quality'] == 'GOOD':
        # 3. Fetch comprehensive data
        data = engine.fetch_any([symbol], lookback_days=30, include_news=True)

        # 4. Data is automatically stored with smart routing
        # No manual storage configuration needed
```

#### Universe Evolution
The system learns and adapts:
- **Symbol frequency tracking**: Popular symbols get priority
- **Category expansion**: New categories discovered automatically
- **Storage optimization**: Frequently accessed data gets better indexing
- **Research suggestions**: Related symbols recommended based on patterns

### Performance & Scalability

#### Lazy Library Creation
- Libraries created only when first symbol stored
- No wasted storage for unused categories
- Automatic cleanup of empty libraries

#### Intelligent Caching
- Symbol metadata cached for fast access
- Access patterns tracked for optimization
- Related symbol suggestions precomputed

#### Resource Management
```yaml
limits:
  max_libraries: 1000
  max_symbols_per_library: 10000
  storage_alert_threshold: 10GB
  query_timeout: 30s
```

### Migration from Rigid to Flexible

#### Backwards Compatibility
- Existing data accessible through legacy methods
- Gradual migration without data loss
- Dual storage support during transition

#### Migration Strategy
1. **Phase 1**: Deploy flexible storage alongside existing
2. **Phase 2**: Route new symbols to flexible storage
3. **Phase 3**: Migrate existing symbols with metadata enrichment
4. **Phase 4**: Retire rigid storage components

This policy ensures hedge fund-grade data quality with clear separation of concerns and professional data handling standards, while enabling true exploratory quantitative research capabilities.

## Tick Data Processing Pipeline (Implemented)

### IQFeed Tick Data Structure
Based on extensive analysis, IQFeed provides 14 fields in tick data:

```python
# NumPy structured array from PyIQFeed
dtype = [
    ('tick_id', '<u8'),      # Unique tick identifier
    ('date', '<M8[D]'),      # Trade date
    ('time', '<m8[us]'),     # Microseconds since midnight ET
    ('last', '<f8'),         # Trade price
    ('last_sz', '<u8'),      # Trade size (volume)
    ('last_type', 'S1'),     # Exchange code (bytes)
    ('mkt_ctr', '<u4'),      # Market center ID
    ('tot_vlm', '<u8'),      # Cumulative daily volume
    ('bid', '<f8'),          # Best bid at trade time
    ('ask', '<f8'),          # Best ask at trade time
    ('cond1', 'u1'),         # Trade condition 1
    ('cond2', 'u1'),         # Trade condition 2
    ('cond3', 'u1'),         # Trade condition 3
    ('cond4', 'u1')          # Trade condition 4
]
```

**CRITICAL LIMITATION**: IQFeed tick data does NOT include bid_size or ask_size, limiting certain microstructure calculations.

### Data Processing Optimizations (Implemented)

#### 1. NumPy Array Validation
**Comprehensive validation with chunked processing support:**
- **Structure validation**: Verifies all 14 expected fields present
- **Data range validation**: Checks prices > 0, spreads >= 0, dates match
- **Smart chunking**: Automatically processes arrays > 1M ticks in chunks
- **Detailed logging**: Info/warning/error levels for debugging

#### 2. DataFrame Conversion Optimization
**Memory-efficient conversion achieving 40% reduction:**

| Data Type | Original | Optimized | Reduction |
|-----------|----------|-----------|-----------|
| Prices | float64 | float32 | 50% |
| Volumes | int64 | uint32 | 50% |
| Market Center | int64 | uint16 | 75% |
| Exchange | object | category | 90% |
| Conditions | int64 | uint8 | 87.5% |

**Performance Results (100k ticks):**
- Speed: 2.8x faster (103ms vs 283ms)
- Memory: 41.6% reduction (16MB vs 27MB peak)
- DataFrame size: 62% smaller (5.25MB vs 13.92MB)

### Essential Metrics (Pre-computed)

Based on institutional quantitative hedge fund requirements, we pre-compute essential metrics during ingestion:

#### Tier 1: Always Computed
```python
# Price/Spread Metrics
spread = ask - bid                          # Liquidity indicator
midpoint = (bid + ask) / 2                  # Fair value estimate
spread_bps = (spread / midpoint) * 10000    # Basis points
spread_pct = spread / midpoint               # Percentage

# Trade Metrics
dollar_volume = price * volume              # Dollar amount traded
effective_spread = 2 * |price - midpoint|   # Actual transaction cost

# Lee-Ready Trade Classification
trade_sign = {
    +1 if price > midpoint,   # Buyer-initiated
    -1 if price < midpoint,   # Seller-initiated
    tick_test if price == midpoint
}

# Price Movement
log_return = log(price / prev_price)        # Logarithmic return
tick_direction = sign(price - prev_price)   # Up/down/unchanged

# Volume Metrics
volume_rate = total_volume - prev_total_volume
trade_pct_of_day = volume / total_volume

# Condition Flags
is_extended_hours = (condition_1 == 135)
is_odd_lot = (condition_3 == 23)
is_regular = all conditions == 0
```

#### Metrics NOT Computed (Missing Data)
Due to lack of bid_size/ask_size in IQFeed tick data:
- **microprice** (weighted midpoint formula)
- **quote_imbalance** (bid vs ask size)
- **depth_ratio** (bid_size / ask_size)
- **order_book_imbalance** metrics

#### Complex Metrics (Future Stage 2)
Deferred to separate analytics engine:
- **VPIN** (Volume-Synchronized PIN)
- **Kyle's Lambda** (price impact regression)
- **PIN/AdjPIN** (probability of informed trading)
- **Realized spread** (requires 5-min future prices)
- **Amihud illiquidity** (requires daily aggregation)

### Context Handling for Chunked Processing

For large tick arrays (>1M ticks), the system maintains context between chunks:

```python
# Context passed between chunks
context = {
    'last_price': float,         # For log_return calculation
    'last_midpoint': float,      # For tick test
    'last_total_volume': int,    # For volume_rate
}

# Ensures metrics continuity across chunk boundaries
```

### Trade Condition Codes Reference

Common condition codes and their meanings:

| Code | Meaning | Field |
|------|---------|-------|
| 135 | Extended hours trading | cond1 |
| 23 | Odd lot (<100 shares) | cond3 |
| 61 | Trade qualifier | cond2 |
| 0 | Regular trade | all |

### Exchange Codes Reference

| Code | Exchange |
|------|----------|
| O | NYSE Arca |
| Q | NASDAQ |
| N | NYSE |
| A | NYSE American |
| C | NSX |
| D | FINRA ADF |

### Storage Impact Analysis

With essential metrics pre-computed:
- **Raw tick data**: ~5.0 MB per 100k ticks
- **+Essential metrics**: ~2.0 MB per 100k ticks
- **Total**: ~7.0 MB (40% increase uncompressed)
- **With LZ4 compression**: ~20% actual increase

### Query Performance Benefits

Pre-computing metrics provides:
- **10-100x faster** filtering on spread/volume/conditions
- **Instant** trade classification (no recalculation)
- **Consistent** calculations across all analyses
- **Reduced** CPU usage during research

### Implementation Architecture

```
Stage 1: Data Engine (CURRENT)
├── IQFeedCollector
│   └── Returns NumPy arrays (no conversion)
├── TickStore
│   ├── Validation (structure, ranges, chunking)
│   ├── Conversion (NumPy → DataFrame)
│   ├── Essential metrics computation
│   └── Storage to ArcticDB
│
Stage 2: Analytics Engine (FUTURE)
├── Complex metrics (VPIN, Kyle's lambda)
├── Aggregated statistics
└── Machine learning features
```

### Future Enhancements

1. **Level 2 Data Integration**: If bid/ask sizes become available, enable:
   - Microprice calculation
   - Order book imbalance metrics
   - Depth analysis

2. **Options Tick Data**: Separate pipeline for options with:
   - Strike/expiration metadata
   - Greeks calculation
   - Implied volatility

3. **Real-time Streaming**: Extend to live tick processing:
   - Incremental metric updates
   - Real-time trade classification
   - Live alert generation

### Decision Log

**2024-01-15**: Key architectural decisions made:
1. **Pre-compute essential metrics**: Better performance than on-demand calculation
2. **Use float32/uint32**: Sufficient precision with 40% memory savings
3. **Implement chunking at 1M ticks**: Balance between memory and efficiency
4. **Defer complex metrics to Stage 2**: Keep ingestion pipeline fast
5. **Document data limitations**: No bid/ask sizes affects some metrics
6. **Maintain context between chunks**: Ensures metric continuity

### Performance Benchmarks

| Operation | Ticks | Time | Memory | Notes |
|-----------|-------|------|--------|-------|
| Validation | 100k | 15ms | 2MB | Structure + ranges |
| Conversion | 100k | 103ms | 16MB | With all metrics |
| Storage | 100k | 200ms | 5MB | LZ4 compressed |
| Full Pipeline | 100k | 318ms | 23MB | End-to-end |
| Chunked | 2M | 6.4s | 25MB | Processes in chunks |

## Metadata Layer Architecture (Institutional Grade)

### Overview: Two-Tier Storage Strategy

Our system implements a sophisticated two-tier data architecture that separates detailed tick data from aggregated metadata, enabling institutional-grade analytics with optimal performance:

```
┌─────────────────────────────────────────────────────────────┐
│                     ArcticDB Storage                         │
├─────────────────────────┬───────────────────────────────────┤
│    DataFrame (Heavy)    │      Metadata (Light)             │
├─────────────────────────┼───────────────────────────────────┤
│ • Per-tick data         │ • Summary statistics              │
│ • Essential metrics     │ • Aggregated indicators           │
│ • 100k-2M rows/day     │ • Single document                 │
│ • Loaded for analysis   │ • Loaded for discovery           │
│ • ~7MB per 100k ticks  │ • ~10KB per day                   │
└─────────────────────────┴───────────────────────────────────┘
```

### Data Separation Philosophy

**DataFrame Contains (Per-Tick):**
- All original IQFeed fields (price, volume, bid, ask, etc.)
- Essential computed metrics (spread, midpoint, trade_sign, dollar_volume)
- Condition flags (is_extended_hours, is_odd_lot, is_regular)
- Price movements (log_return, tick_direction)

**Metadata Contains (Aggregated):**
- Statistical summaries of DataFrame columns
- Market microstructure indicators
- Liquidity and execution quality metrics
- Regime detection and anomaly flags
- Access patterns and data lineage

### Phase 1: Essential Metadata (Immediate Implementation)

These metadata fields will be computed and stored with every tick data write:

#### 1.1 Basic Statistics (Already Implemented)
```python
metadata['basic_stats'] = {
    'symbol': str,
    'date': str,
    'total_ticks': int,
    'first_tick_time': datetime,
    'last_tick_time': datetime,
    'price_open': float,
    'price_high': float,
    'price_low': float,
    'price_close': float,
    'volume_total': int,
    'vwap': float,
    'dollar_volume': float
}
```

#### 1.2 Spread Statistics (To Implement)
```python
metadata['spread_stats'] = {
    'mean_bps': float,        # Average spread in basis points
    'median_bps': float,      # Median spread (typical conditions)
    'std_bps': float,         # Spread volatility
    'min_bps': float,         # Tightest spread observed
    'max_bps': float,         # Widest spread observed
    'p25_bps': float,         # 25th percentile
    'p75_bps': float,         # 75th percentile
    'p95_bps': float,         # 95th percentile (stress indicator)
    'p99_bps': float,         # 99th percentile (extreme conditions)
    'zero_spread_count': int, # Locked market instances
    'inverted_count': int     # Crossed market instances
}
```

#### 1.3 Trade Classification Summary
```python
metadata['trade_classification'] = {
    'buy_count': int,              # Trades classified as buys
    'sell_count': int,             # Trades classified as sells
    'neutral_count': int,          # Trades at midpoint
    'buy_volume': int,             # Total buy volume
    'sell_volume': int,            # Total sell volume
    'buy_dollar_volume': float,    # Dollar value of buys
    'sell_dollar_volume': float,   # Dollar value of sells
    'buy_sell_ratio': float,       # Buy/sell imbalance
    'volume_weighted_sign': float, # Net directional flow
    'large_buy_count': int,        # Block buys (>10k shares)
    'large_sell_count': int        # Block sells (>10k shares)
}
```

#### 1.4 Liquidity Profile
```python
metadata['liquidity_profile'] = {
    'quote_intensity': float,      # Updates per second
    'avg_trade_size': float,       # Mean trade size
    'median_trade_size': float,    # Typical trade size
    'trade_frequency': float,      # Trades per minute
    'effective_tick_size': float,  # Minimum price movement
    'price_levels_count': int,     # Unique prices traded
    'time_between_trades_ms': float, # Avg milliseconds between trades
    'liquidity_score': float       # Composite liquidity metric [0-100]
}
```

#### 1.5 Execution Quality Metrics
```python
metadata['execution_metrics'] = {
    'effective_spread_mean': float,    # Actual vs quoted spread
    'effective_spread_median': float,  # Typical execution cost
    'price_improvement_rate': float,   # % trades inside spread
    'at_midpoint_rate': float,        # % trades at midpoint
    'at_bid_rate': float,              # % trades at bid
    'at_ask_rate': float,              # % trades at ask
    'outside_quote_rate': float,       # % trades outside NBBO
    'odd_lot_rate': float,             # % odd lots (<100 shares)
    'block_rate': float                # % blocks (>10k shares)
}
```

### Phase 2: Advanced Metadata (Near Term Implementation)

These institutional-grade metrics will be added after Phase 1 is stable:

#### 2.1 Order Flow Toxicity Indicators
```python
metadata['toxicity_metrics'] = {
    'adverse_selection_score': float,  # Post-trade price movement
    'information_asymmetry': float,    # Spread widening patterns
    'toxic_minutes': int,              # High-toxicity period count
    'reversion_rate_1min': float,     # Price reversion after trades
    'reversion_rate_5min': float,     # Longer-term reversion
    'permanent_impact_estimate': float # Persistent price impact
}
```

#### 2.2 Market Impact Predictors
```python
metadata['impact_features'] = {
    'kyle_lambda_estimate': float,      # Price impact coefficient
    'temporary_impact_halflife': float, # Seconds for impact decay
    'avg_trade_impact_bps': float,     # Typical market impact
    'large_trade_impact_bps': float,   # Block trade impact
    'cumulative_impact_bps': float,    # Day's total impact
    'resilience_score': float          # Speed of recovery [0-100]
}
```

#### 2.3 Microstructure Patterns
```python
metadata['microstructure_patterns'] = {
    'quote_stuffing_score': float,     # Abnormal quote/trade ratio
    'momentum_ignition_events': int,   # Rapid price movements
    'mini_flash_crashes': int,         # Sudden drops > 50bps
    'tick_clustering': float,          # Price discreteness
    'autocorrelation_1min': float,     # Return predictability
    'autocorrelation_5min': float,     # Medium-term patterns
    'hurst_exponent': float,           # Trending vs mean-reverting
    'microstructure_noise': float      # High-frequency noise level
}
```

#### 2.4 Institutional Flow Indicators
```python
metadata['institutional_flow'] = {
    'block_trade_count': int,          # Trades > 10k shares
    'block_volume_pct': float,         # % volume from blocks
    'sweep_order_count': int,          # Multi-venue executions
    'odd_lot_ratio': float,            # Retail vs institutional
    'average_trade_value': float,      # Dollar value per trade
    'institutional_participation': float, # Estimated institutional %
    'smart_money_indicator': float,    # Predictive large trades
    'accumulation_distribution': float  # Wyckoff accumulation score
}
```

#### 2.5 Regime Detection
```python
metadata['market_regime'] = {
    'volatility_regime': str,    # 'low', 'normal', 'high', 'extreme'
    'liquidity_state': str,      # 'thick', 'normal', 'thin', 'dried_up'
    'trend_state': str,          # 'strong_up', 'up', 'neutral', 'down', 'strong_down'
    'microstructure_regime': str, # 'hft_dominant', 'mixed', 'fundamental'
    'stress_indicator': float,   # Market stress level [0-100]
    'regime_change_detected': bool, # Structural break flag
    'regime_duration_minutes': int  # Time in current regime
}
```

### Phase 3: Specialized Metadata (Future Enhancement)

Advanced features for sophisticated strategies:

#### 3.1 HFT Activity Measures
```python
metadata['hft_activity'] = {
    'message_rate': float,             # Updates per second
    'cancel_to_trade_ratio': float,    # Order book churn
    'quote_flicker_rate': float,       # Rapid quote changes
    'latency_arbitrage_ops': int,      # Exploitable delays
    'speed_bump_effect': float,        # IEX-style delay impact
    'colocation_advantage': float,     # Speed advantage value
    'hft_participation_pct': float,    # Estimated HFT volume %
    'predatory_algo_score': float      # Aggressive HFT detection
}
```

#### 3.2 Machine Learning Features
```python
metadata['ml_features'] = {
    'feature_vector': list[float],     # Top 50 engineered features
    'anomaly_score': float,            # Isolation forest score
    'cluster_id': int,                 # Market structure cluster
    'embedding_vector': list[float],   # Learned representation (dim=32)
    'predictive_power': dict,          # Feature importance scores
    'pattern_similarity': dict,        # Similar historical days
    'ml_regime_prediction': str,       # Next regime forecast
    'confidence_score': float          # Prediction confidence [0-1]
}
```

#### 3.3 Cross-Asset Signals
```python
metadata['cross_asset_context'] = {
    'spy_correlation': float,          # vs S&P 500 ETF
    'sector_correlation': float,       # vs sector ETF
    'vix_correlation': float,          # vs volatility index
    'dxy_correlation': float,          # vs dollar index
    'rates_sensitivity': float,        # vs 10-year yield
    'commodity_beta': float,           # vs commodity index
    'crypto_correlation': float,       # vs BTC (if applicable)
    'global_sync_score': float         # Cross-market synchronization
}
```

#### 3.4 Regulatory & Compliance
```python
metadata['regulatory_metrics'] = {
    'reg_nms_compliance': bool,        # Best execution flag
    'mifid_ii_flags': dict,           # European compliance
    'cat_reportable_events': int,      # Consolidated Audit Trail
    'suspicious_patterns': int,        # Potential manipulation
    'wash_trade_candidates': int,      # Self-trading patterns
    'layering_score': float,          # Spoofing detection
    'marking_the_close': bool,        # End-of-day manipulation
    'audit_flags': list[str]          # Compliance warnings
}
```

### Implementation Architecture

#### Metadata Computation Pipeline
```python
def compute_metadata(df: pd.DataFrame, symbol: str, date: str) -> dict:
    """
    Compute all metadata from tick DataFrame.
    Called AFTER DataFrame creation but BEFORE storage.
    """
    metadata = {}

    # Phase 1: Essential (always computed)
    metadata['basic_stats'] = compute_basic_stats(df)
    metadata['spread_stats'] = compute_spread_stats(df)
    metadata['trade_classification'] = compute_trade_classification(df)
    metadata['liquidity_profile'] = compute_liquidity_profile(df)
    metadata['execution_metrics'] = compute_execution_metrics(df)

    # Phase 2: Advanced (if enabled)
    if config.ADVANCED_METRICS_ENABLED:
        metadata['toxicity_metrics'] = compute_toxicity_metrics(df)
        metadata['impact_features'] = compute_impact_features(df)
        metadata['microstructure_patterns'] = compute_patterns(df)
        metadata['institutional_flow'] = compute_institutional_flow(df)
        metadata['market_regime'] = detect_regime(df)

    # Phase 3: Specialized (if enabled)
    if config.SPECIALIZED_METRICS_ENABLED:
        metadata['hft_activity'] = compute_hft_metrics(df)
        metadata['ml_features'] = compute_ml_features(df)
        metadata['cross_asset_context'] = compute_cross_asset(df, symbol)
        metadata['regulatory_metrics'] = compute_regulatory_metrics(df)

    return metadata
```

### Storage Integration with ArcticDB

```python
# Writing data with metadata
def store_ticks_with_metadata(symbol: str, date: str, df: pd.DataFrame):
    # Compute metadata from DataFrame
    metadata = compute_metadata(df, symbol, date)

    # Store in ArcticDB with metadata
    storage_key = f"{symbol}/{date}"
    arctic_lib.write(
        storage_key,
        df,
        metadata=metadata  # Metadata stored alongside data
    )

# Reading just metadata (fast)
def get_metadata(symbol: str, date: str) -> dict:
    storage_key = f"{symbol}/{date}"
    # This only loads metadata, not the DataFrame
    return arctic_lib.read_metadata(storage_key)

# Reading full data (slower)
def get_tick_data(symbol: str, date: str) -> pd.DataFrame:
    storage_key = f"{symbol}/{date}"
    # This loads the entire DataFrame
    return arctic_lib.read(storage_key).data
```

### Use Cases and Benefits

#### 1. Quick Data Quality Assessment
```python
# Check data quality without loading ticks
meta = get_metadata('AAPL', '2024-01-15')
if meta['spread_stats']['p95_bps'] > 50:
    print("Warning: Wide spreads detected")
if meta['liquidity_profile']['liquidity_score'] < 30:
    print("Warning: Low liquidity day")
```

#### 2. GUI Dashboard Population
```python
# Instant dashboard cards without data loading
meta = get_metadata('AAPL', '2024-01-15')
dashboard.display_card('Avg Spread', f"{meta['spread_stats']['mean_bps']:.1f} bps")
dashboard.display_card('Buy/Sell Ratio', f"{meta['trade_classification']['buy_sell_ratio']:.2f}")
dashboard.display_card('Liquidity Score', f"{meta['liquidity_profile']['liquidity_score']:.0f}/100")
```

#### 3. Research Discovery Patterns
```python
# Find interesting days for detailed analysis
interesting_days = []
for date in date_range:
    meta = get_metadata('AAPL', date)
    # High toxicity days
    if meta.get('toxicity_metrics', {}).get('toxic_minutes', 0) > 30:
        interesting_days.append(('high_toxicity', date))
    # Regime changes
    if meta.get('market_regime', {}).get('regime_change_detected', False):
        interesting_days.append(('regime_change', date))
    # Unusual institutional flow
    if meta.get('institutional_flow', {}).get('smart_money_indicator', 0) > 0.8:
        interesting_days.append(('smart_money', date))
```

#### 4. Adaptive Loading Strategy
```python
# Load data intelligently based on metadata
meta = get_metadata('AAPL', '2024-01-15')

# High activity day - load everything
if meta['liquidity_profile']['quote_intensity'] > 100:
    df = get_tick_data('AAPL', '2024-01-15')

# Normal day - load sampled data
elif meta['liquidity_profile']['quote_intensity'] > 10:
    df = get_tick_data('AAPL', '2024-01-15', sample_rate=10)

# Low activity - load aggregated bars instead
else:
    df = get_1min_bars('AAPL', '2024-01-15')
```

#### 5. Cross-Sectional Analysis
```python
# Compare symbols without loading data
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
date = '2024-01-15'

spread_comparison = {}
for symbol in symbols:
    meta = get_metadata(symbol, date)
    spread_comparison[symbol] = meta['spread_stats']['median_bps']

# Find symbol with best liquidity
best_liquidity = max(symbols,
    key=lambda s: get_metadata(s, date)['liquidity_profile']['liquidity_score'])
```

### Performance Benefits

#### Speed Improvements
- **Metadata-only queries**: 1000x faster (10ms vs 10s)
- **Dashboard population**: Instant (<50ms total)
- **Data discovery**: Scan 1000 days in <10 seconds
- **Conditional loading**: Load only what's needed

#### Storage Efficiency
- **Metadata size**: ~10KB per symbol-day
- **Compression ratio**: Metadata highly compressible
- **Index optimization**: Metadata fields can be indexed
- **Cache friendly**: Entire metadata set fits in memory

#### Research Productivity
- **Quick hypothesis testing**: Test ideas without full data loads
- **Pattern discovery**: Find anomalies across time/symbols
- **Smart sampling**: Load detailed data only when needed
- **Batch analysis**: Process metadata for entire universe

### Migration Path

#### Phase 1 (Week 1-2): Essential Metadata
1. Implement basic statistics computation
2. Add spread and trade classification stats
3. Integrate with existing TickStore
4. Update GUI to use metadata

#### Phase 2 (Week 3-4): Advanced Metadata
1. Add toxicity and impact metrics
2. Implement regime detection
3. Add institutional flow indicators
4. Create metadata-based alerts

#### Phase 3 (Month 2): Specialized Features
1. Add HFT activity measures
2. Implement ML feature extraction
3. Add cross-asset correlations
4. Integrate compliance metrics

### Best Practices

1. **Compute Once, Use Many**: Calculate metadata during ingestion, not on-demand
2. **Version Control**: Track metadata schema versions for compatibility
3. **Incremental Updates**: Support adding new metrics without reprocessing
4. **Fail Gracefully**: Missing metadata fields shouldn't break queries
5. **Monitor Performance**: Track metadata computation time and size

### Conclusion

This institutional-grade metadata architecture transforms our tick data storage from a simple historical record into an intelligent, queryable system that enables:
- Instant insights without data loading
- Sophisticated pattern discovery
- Adaptive analysis strategies
- Regulatory compliance tracking
- Machine learning integration

By separating heavy tick data from lightweight metadata, we achieve the best of both worlds: detailed data when needed, instant insights always available.

## Advanced Metadata Concepts

### 1. Temporal Metadata Relationships

Track intraday evolution of metrics in hourly buckets:

```python
metadata['intraday_evolution'] = {
    '09:30-10:30': {  # First hour
        'spread_mean_bps': 15.2,
        'volume_pct': 18.5,  # % of daily volume
        'volatility': 0.8,
        'trade_count': 12543,
        'buy_sell_ratio': 1.2
    },
    '10:30-11:30': {
        'spread_mean_bps': 12.1,
        'volume_pct': 12.3,
        'volatility': 0.6,
        'trade_count': 8234,
        'buy_sell_ratio': 0.95
    },
    # ... for each hour
    '15:00-16:00': {  # Power hour
        'spread_mean_bps': 18.5,
        'volume_pct': 22.1,
        'volatility': 1.2,
        'trade_count': 18765,
        'buy_sell_ratio': 1.1
    }
}
```

### 2. Event-Triggered Metadata

Capture market impact of significant events:

```python
metadata['event_impacts'] = {
    'large_trades': [  # For each >50k share trade
        {
            'timestamp': '10:32:15',
            'size': 75000,
            'price': 236.45,
            'spread_before_bps': 10,
            'spread_after_bps': 25,
            'price_impact_bps': 8,
            'recovery_time_ms': 1500,
            'following_trades': 23  # Trades in next 5 seconds
        }
    ],
    'spread_jumps': [  # Sudden spread widenings >3x normal
        {
            'timestamp': '14:15:30',
            'from_bps': 12,
            'to_bps': 45,
            'duration_ms': 3000,
            'volume_during': 125000,
            'likely_cause': 'news'  # 'news', 'large_trade', 'market_wide', 'unknown'
        }
    ],
    'price_gaps': [  # Price jumps > 0.1%
        {
            'timestamp': '11:45:22',
            'from_price': 236.50,
            'to_price': 237.00,
            'gap_bps': 21,
            'filled': False,
            'time_to_fill': None
        }
    ]
}
```

### 3. Relative Metadata (vs Benchmarks)

Compare symbol performance to peers and market:

```python
metadata['relative_metrics'] = {
    'vs_sector': {  # vs XLK for tech stocks
        'spread_ratio': 0.8,  # Our spread is 80% of sector avg
        'volume_ratio': 1.5,  # We trade 50% more volume
        'volatility_ratio': 1.1,
        'correlation': 0.85,
        'beta': 1.15
    },
    'vs_market': {  # vs SPY
        'spread_percentile': 30,   # Tighter spread than 70% of stocks
        'volume_percentile': 75,   # 75th percentile by volume
        'volatility_percentile': 60,
        'correlation': 0.72,
        'beta': 1.2
    },
    'peer_rank': {  # Among similar market cap stocks
        'liquidity_rank': 12,  # 12th most liquid
        'spread_rank': 8,      # 8th tightest spread
        'volume_rank': 15,
        'universe_size': 50    # Out of 50 peers
    },
    'performance_vs_peers': {
        'spread_improvement': -2.3,  # bps better than yesterday
        'volume_change': +15.2,      # % change vs avg
        'relative_strength': 0.65    # 0-1 score
    }
}
```

### 4. Predictive Metadata

Forward-looking metrics based on current patterns:

```python
metadata['predictions'] = {
    'next_hour_spread': {
        'forecast_bps': 14.5,
        'confidence_interval': [12.0, 17.0],
        'confidence_level': 0.85,
        'model': 'ARIMA',
        'features_used': ['time_of_day', 'recent_volatility', 'volume_rate']
    },
    'eod_volume': {
        'forecast': 2500000,
        'current_pace_pct': 102,  # Running 2% ahead of normal
        'confidence': 0.85,
        'typical_eod': 2450000
    },
    'volatility_forecast': {
        '1hr': 0.015,  # 1.5% expected move
        '4hr': 0.028,  # Until close
        'confidence': 0.75,
        'regime_change_prob': 0.15
    },
    'spread_forecast': {
        'next_15min': 12.5,
        'next_30min': 13.0,
        'next_60min': 14.0,
        'trend': 'widening'
    }
}
```

### 5. Trade Network/Clustering Metadata

Identify related trading patterns:

```python
metadata['trade_networks'] = {
    'trade_clusters': [  # Groups of related trades
        {
            'cluster_id': 1,
            'start_time': '10:15:30',
            'end_time': '10:15:31',
            'trade_count': 45,
            'total_volume': 125000,
            'avg_trade_size': 2778,
            'price_range': [236.45, 236.48],
            'likely_origin': 'algo_execution',  # 'algo', 'sweep', 'block', 'retail'
            'fragmentation': 0.75  # Across multiple venues
        }
    ],
    'momentum_chains': [  # Self-reinforcing price moves
        {
            'chain_id': 1,
            'start_time': '11:15:00',
            'initial_trade': 5000,
            'initial_price': 236.50,
            'follower_trades': 15,
            'total_momentum_volume': 85000,
            'price_move_bps': 35,
            'duration_seconds': 45
        }
    ],
    'liquidity_clusters': [  # Periods of concentrated liquidity
        {
            'period_start': '14:30:00',
            'period_end': '14:35:00',
            'avg_spread_bps': 8,
            'trade_frequency': 125,  # trades per minute
            'avg_trade_size': 500
        }
    ]
}
```

### 6. Data Quality Metadata

Track data completeness and reliability:

```python
metadata['data_quality'] = {
    'completeness_score': 0.98,  # 98% of expected ticks present
    'confidence_level': 'high',  # 'high', 'medium', 'low'
    'tick_gaps': [  # Suspicious gaps in data
        {
            'from': '10:15:30',
            'to': '10:15:45',
            'missing_ticks_estimate': 50,
            'last_price_before': 236.45,
            'first_price_after': 236.48
        }
    ],
    'anomalies': {
        'count': 3,
        'details': [
            {'time': '14:32:10', 'type': 'price_spike', 'deviation_sigma': 4.5},
            {'time': '15:45:22', 'type': 'zero_spread', 'duration_ms': 100}
        ]
    },
    'crossed_markets': {
        'count': 2,
        'total_duration_ms': 250,
        'max_cross_bps': 5
    },
    'data_source_quality': {
        'latency_ms': 45,
        'packet_loss': 0.001,
        'reconnections': 0
    }
}
```

### 7. Execution Venue Metadata

Analyze where trades are executing:

```python
metadata['venue_analysis'] = {
    'exchange_distribution': {
        'Q': 0.45,  # 45% on NASDAQ
        'N': 0.30,  # 30% on NYSE
        'O': 0.15,  # 15% on ARCA
        'D': 0.10   # 10% on ADF (dark)
    },
    'venue_quality': {
        'Q': {'avg_spread_bps': 10, 'fill_rate': 0.95, 'avg_size': 450},
        'N': {'avg_spread_bps': 12, 'fill_rate': 0.92, 'avg_size': 650},
        'O': {'avg_spread_bps': 11, 'fill_rate': 0.93, 'avg_size': 550},
        'D': {'avg_spread_bps': 0, 'fill_rate': 1.0, 'avg_size': 5000}
    },
    'fragmentation_score': 0.65,  # 0=concentrated, 1=fragmented
    'dark_pool_percentage': 10.5,
    'best_execution_venue': 'Q',  # Based on spread and fill rate
    'venue_migration': {  # How venue usage changed during day
        'morning': {'Q': 0.50, 'N': 0.35},
        'midday': {'Q': 0.45, 'N': 0.30},
        'close': {'Q': 0.40, 'N': 0.25, 'D': 0.20}
    }
}
```

### 8. Adaptive Learning Metadata

System learns normal patterns for each symbol:

```python
metadata['adaptive_baselines'] = {
    'typical_spread': {
        'monday': {'open': 15, 'midday': 10, 'close': 12},
        'tuesday': {'open': 14, 'midday': 9, 'close': 11},
        'wednesday': {'open': 13, 'midday': 9, 'close': 11},
        'thursday': {'open': 13, 'midday': 10, 'close': 12},
        'friday': {'open': 14, 'midday': 11, 'close': 15}
    },
    'volume_patterns': {
        'normal_day': 2000000,
        'earnings_day': {'multiplier': 3.5, 'last_occurrence': '2024-01-10'},
        'opex_friday': {'multiplier': 1.8, 'next_occurrence': '2024-01-19'},
        'fed_day': {'multiplier': 2.2, 'next_occurrence': '2024-01-31'},
        'holiday_eve': {'multiplier': 0.7}
    },
    'learned_thresholds': {
        'unusual_spread': 25,  # 95th percentile over 30 days
        'unusual_volume': 5000000,  # 95th percentile
        'regime_change_trigger': 0.03,  # 3% move
        'large_trade': 10000,  # 95th percentile trade size
        'tick_gap_threshold': 5000  # ms before considered gap
    },
    'pattern_recognition': {
        'typical_open_volatility': 1.2,
        'typical_close_volatility': 1.5,
        'lunch_lull_period': ['12:00', '13:00'],
        'most_active_period': ['09:30', '10:00']
    }
}
```

### 9. Historical Context Metadata

How today compares to historical patterns:

```python
metadata['historical_context'] = {
    'percentiles': {
        'spread_1d': 65,    # Today's spread is 65th percentile vs yesterday
        'spread_5d': 70,    # 70th percentile vs last 5 days
        'spread_30d': 73,   # 73rd percentile vs last month
        'spread_1y': 45,    # 45th percentile vs last year
        'volume_1d': 110,   # 10% above yesterday
        'volume_30d': 125,  # 25% above monthly avg
        'volatility_30d': 85  # 85th percentile volatility
    },
    'z_scores': {
        'volume_zscore_30d': 1.2,  # 1.2 std devs above mean
        'spread_zscore_30d': 0.8,
        'volatility_zscore_30d': 1.5
    },
    'similar_days': [  # Historical days with similar patterns
        {'date': '2024-01-03', 'similarity_score': 0.92, 'next_day_move': +0.8},
        {'date': '2023-12-15', 'similarity_score': 0.89, 'next_day_move': -0.3},
        {'date': '2023-11-20', 'similarity_score': 0.87, 'next_day_move': +1.2}
    ],
    'trend_position': {
        'spread_trend': 'widening',  # vs 20-day MA
        'spread_trend_strength': 0.7,  # 0-1
        'volume_trend': 'increasing',
        'volume_trend_strength': 0.4,
        'volatility_trend': 'stable',
        'volatility_trend_strength': 0.1
    },
    'records': {
        'is_highest_volume_30d': False,
        'is_widest_spread_30d': False,
        'is_most_volatile_30d': True,
        'unusual_metrics': ['volatility']  # Metrics > 95th percentile
    }
}
```

## Metadata Computation Architecture

### Computation Timing Strategy

We use a **post-processing approach** where metadata is computed AFTER data storage:

```python
# Workflow:
1. IQFeed → NumPy array
2. NumPy → DataFrame with essential metrics (spread, trade_sign, etc.)
3. DataFrame → ArcticDB storage (FAST write)
4. Background job → Compute metadata → Update ArcticDB metadata (doesn't touch data)
```

**Benefits:**
- Fast ingestion (no metadata computation blocking writes)
- Can recompute metadata if we add new metrics
- Can fix bugs in metadata without reprocessing data
- Metadata computation can be parallelized

### Metadata Versioning Strategy

We implement **"Latest Only with Changelog"** approach:

```python
# Store only latest metadata version
'AAPL/2024-01-15/metadata' = {
    'version': '1.2',
    'computed_at': '2024-01-17 10:00',
    'spread_mean': 12.3,
    # ... all current metrics
}

# Separate changelog for audit trail
metadata_changelog = {
    '2024-01-15': 'Initial computation v1.0',
    '2024-01-16': 'Added institutional flow metrics v1.1',
    '2024-01-17': 'Fixed spread calculation bug v1.2'
}
```

### Cross-Symbol Dependencies

For metrics requiring multiple symbols (correlations, rankings), we use a **two-phase computation**:

```python
class MetadataComputer:
    def compute_metadata_phase1(self, symbol, date, df):
        """Independent metrics - runs immediately after storage"""
        return {
            'spread_stats': compute_spread_stats(df),
            'volume_stats': compute_volume_stats(df),
            'trade_classification': compute_trade_classification(df)
        }

    def compute_metadata_phase2(self, symbol, date):
        """Relative metrics - runs after all symbols processed"""
        return {
            'relative_metrics': compute_relative_to_peers(symbol, date),
            'market_rank': compute_market_rankings(symbol, date),
            'correlation_matrix': compute_correlations(symbol, date)
        }

# Schedule:
# 16:00 - Market closes
# 16:00-17:00 - Phase 1 metadata for all symbols (parallel)
# 17:00-17:30 - Phase 2 relative metadata (needs all Phase 1 complete)
```

### Background Job Scheduling

**Hybrid approach** for optimal performance:

```python
METADATA_SCHEDULE = {
    'immediate': {
        'trigger': 'on_data_stored',
        'delay_seconds': 120,
        'metrics': ['spread_stats', 'volume_stats', 'trade_classification']
    },
    'end_of_day': {
        'trigger': '16:30',
        'metrics': ['institutional_flow', 'regime_detection', 'ml_features']
    },
    'overnight': {
        'trigger': '20:00',
        'metrics': ['relative_metrics', 'correlation_matrix', 'peer_rankings']
    }
}
```

## Non-24/7 Operation Support

### System State Persistence

The system maintains persistent state to handle intermittent availability:

```python
# pipeline_state.json
{
    'last_successful_ingestion': {
        'AAPL': '2024-01-15 16:00:00',
        'MSFT': '2024-01-15 16:00:00'
    },
    'metadata_computed': {
        'AAPL/2024-01-15': {'phase1': true, 'phase2': false},
        'MSFT/2024-01-15': {'phase1': true, 'phase2': false}
    },
    'pending_tasks': [],
    'last_startup': '2024-01-16 08:30:00'
}
```

### Intelligent Catch-Up on Startup

When the system starts after being offline:

#### 1. Gap Identification
```python
def identify_missing_data():
    gaps = {
        'missing_ingestions': [],    # Data not yet fetched
        'missing_metadata': [],       # Metadata not computed
        'incomplete_processing': []   # Partial completions
    }

    for symbol in tracked_symbols:
        last_stored = get_last_stored_date(symbol)
        missing_days = get_trading_days_between(last_stored, now)

        for day in missing_days:
            priority = calculate_priority(symbol, day)
            gaps['missing_ingestions'].append({
                'symbol': symbol,
                'date': day,
                'priority': priority
            })
```

#### 2. Prioritized Backfill Plan
```python
BACKFILL_PRIORITIES = {
    'immediate': [],      # Yesterday's data for key symbols
    'high_priority': [],  # Last 2-8 days for important symbols
    'batch': [],         # Last 30 days, process when idle
    'skip': []           # Older than 30 days, ignore unless requested
}

# Priority scoring considers:
# - Recency (newer = higher priority)
# - Symbol importance (AAPL, SPY, QQQ = high)
# - Special events (earnings, Fed days = high)
# - Day of week (Monday/Friday = higher)
```

#### 3. Execution Strategy
- **Phase 1**: Immediate tasks (blocking, must complete)
- **Phase 2**: High priority tasks (parallel execution)
- **Phase 3**: Batch tasks (background, when resources available)

### Lazy Metadata Computation

Metadata can be computed on-demand if missing:

```python
def get_metadata(symbol, date):
    # Try to load existing
    metadata = load_metadata(symbol, date)

    if metadata is None:
        # Compute on-demand
        print(f"Computing missing metadata for {symbol}/{date}...")
        df = load_tick_data(symbol, date)
        metadata = compute_and_store_metadata(df, symbol, date)

    elif is_outdated(metadata):
        # Recompute if version mismatch
        metadata = recompute_metadata(symbol, date)

    return metadata
```

## Startup Progress Monitoring

### Visual Progress Dashboard

The system provides comprehensive startup progress monitoring:

```python
class StartupProgressMonitor:
    def display_dashboard(self):
        """
        Shows real-time startup progress
        """
        print("=" * 80)
        print("          FUZZY OSS20 - STARTUP PROGRESS")
        print("=" * 80)

        # Overall progress bar
        print_progress_bar("SYSTEM READINESS", system_readiness_pct)

        # Component progress
        print_progress_bar("Data Ingestion", data_completeness_pct)
        print_progress_bar("Metadata Coverage", metadata_coverage_pct)
        print_progress_bar("Index Building", index_progress_pct)
        print_progress_bar("Cache Warming", cache_progress_pct)

        # Current operation
        print(f"CURRENT: {current_operation}")

        # ETA
        print(f"TIME REMAINING: {estimated_time}")
```

### System Assessment on Startup

```python
def assess_system_state():
    """
    Comprehensive assessment showing:
    """
    return {
        'data_coverage': {
            'AAPL': 95.5,  # % of expected data present
            'MSFT': 92.3,
            'SPY': 100.0
        },
        'metadata_coverage': {
            'complete': 145,     # Fully computed
            'partial': 12,       # Phase 1 only
            'missing': 8         # Not computed
        },
        'system_health': {
            'storage_usage': '45GB / 100GB',
            'last_successful_run': '2024-01-15 16:45',
            'errors_last_24h': 2,
            'warnings_last_24h': 15
        },
        'estimated_catch_up_time': '12 minutes',
        'recommendations': [
            'Fetch yesterday\'s data for AAPL',
            'Recompute metadata for 8 symbol-days',
            'Consider archiving data older than 180 days'
        ]
    }
```

### Startup Summary Report

After startup completes:

```
================================================================================
                    ✅ STARTUP COMPLETE
================================================================================

⏱️  TIMING:
  • Started: 08:30:15
  • Completed: 08:42:37
  • Duration: 12m 22s

📊 DATA STATISTICS:
  • Symbols tracked: 15
  • Days fetched: 3
  • Ticks processed: 1,234,567
  • Metadata computed: 45

🎯 SYSTEM STATUS:
  ✅ Data Engine: ready
  ✅ Metadata Computer: ready
  ✅ Storage (ArcticDB): ready (45.2GB used)
  ⚠️ IQFeed Connection: degraded (high latency)

📈 COVERAGE REPORT:
  • Data completeness: 98.5%
  • Metadata coverage: 95.2%
  • Overall readiness: 96.8%

⚠️  ISSUES DETECTED:
  • IQFeed latency above normal (125ms vs 50ms typical)
  • 2 symbols missing Friday's data (will retry)

💡 RECOMMENDATIONS:
  • Schedule backfill for missing Friday data
  • Monitor IQFeed connection stability

================================================================================
System ready for use. Happy trading! 🚀
================================================================================
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
1. Implement state persistence (pipeline_state.json)
2. Create gap identification system
3. Build basic catch-up routine
4. Add progress monitoring

### Phase 2: Metadata System (Week 2)
1. Implement Phase 1 metadata computation
2. Add lazy computation fallback
3. Create metadata versioning
4. Build changelog system

### Phase 3: Advanced Features (Week 3-4)
1. Add Phase 2 relative metadata
2. Implement smart prioritization
3. Create visual dashboards
4. Add startup assessment

### Phase 4: Optimization (Month 2)
1. Parallel processing optimization
2. Caching strategies
3. Resource management
4. Performance monitoring

## Alert System (Deferred)

Alert mechanisms are deferred to later development stages. When implemented, will follow a three-tier approach:
- **Critical**: Hard-coded thresholds (immediate attention)
- **Warning**: Learned thresholds (investigation needed)
- **Info**: User-configured (monitoring/logging)

## Best Practices for Non-24/7 Systems

1. **Always persist state** between sessions
2. **Prioritize recent data** when catching up
3. **Use lazy computation** for missing metadata
4. **Implement progress monitoring** for user feedback
5. **Handle partial failures gracefully**
6. **Log all catch-up activities** for debugging
7. **Provide clear startup summaries**
8. **Allow manual override** of automatic decisions