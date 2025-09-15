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