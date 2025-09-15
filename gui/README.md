# 🖥️ Streamlit GUI for Tick Data Analysis

## Overview
Professional-grade GUI for exploring and analyzing tick data stored in ArcticDB with pre-computed microstructure metrics. Focus on exploratory data analysis rather than trading operations.

## Current Implementation Status
- ✅ Basic Streamlit structure with mock data
- ✅ TradingView-style charting with Plotly
- ✅ Data interface layer (partial)
- ⏳ Real data connection to TickStore
- ⏳ Metrics visualization
- ⏳ Microstructure analysis tools

## 🎯 Core Functionality

### Data Available for Visualization
Each tick in our ArcticDB storage contains:

**Original IQFeed Fields:**
- timestamp, price, volume, bid, ask
- exchange, market_center, total_volume
- condition codes (1-4)

**Pre-computed Metrics:**
- spread, midpoint, spread_bps, spread_pct
- dollar_volume, effective_spread
- trade_sign (Lee-Ready classification)
- log_return, tick_direction
- volume_rate, trade_pct_of_day
- is_extended_hours, is_odd_lot, is_regular

## 📊 Proposed Streamlit GUI Structure
### Sidebar (Control Panel)
- **Connection Status Panel**
  - ArcticDB connection (green/red indicator)
  - IQFeed status (if live data enabled)
  - Available symbols dropdown (from database)
  - Date range selector

- **Data Selection**
  - Symbol input/selector
  - Date picker (single or range)
  - Time range filter (e.g., 9:30-16:00)
  - Data type (Ticks with metrics / Aggregated bars)

- **Filter Options**
  - Trade conditions (regular/extended/odd lot)
  - Spread range (min/max basis points)
  - Volume threshold
  - Trade sign filter (buy/sell/all)

### Main Tabs
#### 1. **Data Discovery** 🔍
- **Storage Overview**
  - List all available symbols in ArcticDB
  - Data coverage calendar (which dates have data)
  - Storage statistics (rows, memory usage, compression ratio)
  - Latest update timestamps

- **Quick Preview**
  - Last 100 ticks for selected symbol
  - Summary statistics (OHLC, volume, spread stats)
  - Data quality indicators

#### 2. **Tick Explorer** 📈
- **Data Grid View**
  - Paginated tick data with ALL fields and metrics
  - Sortable/filterable columns
  - Color coding for trade conditions
  - Trade sign indicators (↑ buy / ↓ sell)

- **Time Series Charts**
  - Price with bid/ask bands
  - Volume bars with trade sign coloring
  - Spread evolution (basis points)
  - Cumulative volume profile

- **Export Options**
  - Filtered data to CSV
  - Full day to Parquet
  - Selected metrics only

#### 3. **Microstructure Analysis** 🔬
- **Spread Analytics**
  - Intraday spread patterns
  - Effective vs quoted spread comparison
  - Spread distribution histogram
  - Time-weighted average spread

- **Trade Classification**
  - Lee-Ready accuracy metrics
  - Buy/sell pressure indicators
  - Trade sign distribution pie chart
  - Cumulative trade imbalance

- **Market Quality Metrics**
  - Dollar volume by hour
  - Trade size distribution
  - Odd lot percentage over time
  - Extended hours activity analysis

#### 4. **Metrics Dashboard** 📊
- **Pre-computed Metrics Visualization**
  - Log returns distribution
  - Tick direction patterns
  - Volume rate analysis
  - Trade intensity heatmap

- **Condition Analysis**
  - Extended hours vs regular comparison
  - Odd lot impact on spreads
  - Condition code frequency table

#### 5. **Multi-Symbol Comparison** 🔄
- **Cross-Symbol Analysis**
  - Relative spread comparison
  - Volume patterns alignment
  - Correlation matrices for metrics
  - Synchronized time series plots

- **Sector/Peer Analysis**
  - Compare similar stocks
  - Identify outliers in metrics
  - Benchmark against averages

#### 6. **Data Quality Monitor** ✅
- **Validation Dashboard**
  - Missing data detection
  - Outlier identification
  - Gap analysis (time between ticks)
  - Data completeness scores

- **Pipeline Health**
  - Last successful ingestion
  - Error logs viewer
  - Chunk processing statistics
  - Memory usage tracking

⚡ Technical Details

IQFeed integration: via pyiqfeed Level1 (real-time) + HistoryProvider

Streaming handling: async callbacks → Streamlit st.session_state

Charts: Plotly (candlesticks, line, volume)

Tables: AgGrid (sortable/filterable live tables)

Storage: ArcticDB (high-performance) with stats exposed via GUI

✅ This way, your dev team gets:

A professional trader-style dashboard

A test harness for IQFeed + storage

A GUI for monitoring + visualization