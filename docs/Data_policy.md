# Fuzzy OSS20 Data Policy

## Base Data Strategy

### Primary Data Collection
- **50-tick bars**: Main data unit for tick-based analysis
- **5-second bars**: Main data unit for time-based analysis
- **NO raw ticks**: We aggregate immediately to prevent storage bloat
- **NO fallbacks**: If primary data unavailable, we wait/retry, never substitute

### Historical Data Limits
- **During market hours**: 8 trading days maximum
- **After market close**: 180 calendar days maximum
- **Rationale**: Balance storage costs with analytical needs

## Data Sources Architecture

### IQFeed (Primary Market Data)
```
stage_01_data_engine/
├── collectors/
│   ├── tick_collector.py          # 50-tick & 5-second bars
│   ├── dtn_indicators_collector.py # DTN Calculated Indicators
│   └── iqfeed_historical.py       # Daily bulk downloads
├── storage/
│   ├── tick_storage.py            # ArcticDB tick data
│   ├── indicator_storage.py       # ArcticDB DTN indicators
│   └── bar_builder.py             # All bar type construction
└── config/
    └── indicator_config.yaml      # DTN symbol mappings
```

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

This policy ensures hedge fund-grade data quality with clear separation of concerns and professional data handling standards.