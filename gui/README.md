# Stage 1 Data Engine - Testing GUI

Real-time Streamlit interface for testing the flexible storage system with actual market data.

## Features

### 🔍 Symbol Testing
- Parse ANY financial symbol (stocks, futures, forex, options, DTN indicators)
- See automatic categorization and routing decisions
- View expected storage paths
- Test symbol validity

### 📂 Storage Inspector
- Browse all stored symbols across all backends
- View storage statistics and performance metrics
- Inspect stored data with interactive charts
- Monitor connection status to all components

### 🔄 Round-Trip Testing
- Complete pipeline testing: Parse → Fetch → Store → Retrieve → Verify
- Automated data integrity verification
- Visual confirmation of storage functionality
- Step-by-step success/failure reporting

### 📈 Live Data Testing
- Fetch real market data from IQFeed/Polygon
- Store data using flexible storage system
- Immediately retrieve and compare data
- Visual data previews and verification

## Quick Start

### 1. Launch GUI
```bash
cd C:\Users\maxli\PycharmProjects\PythonProject\Fuzzy_oss20
streamlit run gui/streamlit_app.py
```

### 2. Test Symbol Parsing
- Go to "Symbol Testing" tab
- Enter any symbol (e.g., AAPL, @ES#, EUR/USD)
- Click "Parse Symbol" to see categorization

### 3. Fetch Real Data
- Go to "Live Data Testing" tab
- Enter a symbol (e.g., AAPL)
- Click "Fetch Real Data"
- Watch the complete pipeline in action

### 4. Verify Storage
- Use "Storage Inspector" to browse stored data
- View performance metrics and statistics
- Check data integrity

## Symbol Examples

### Stocks
- AAPL, MSFT, TSLA, GOOGL, SPY, QQQ

### Futures
- @ES# (S&P 500), @NQ# (Nasdaq), @CL# (Oil), @GC# (Gold)

### Forex
- EUR/USD, GBP/USD, USD/JPY, AUD/USD

### Options
- AAPL230120C00150000 (AAPL Call)
- SPY231215P00400000 (SPY Put)

### DTN Indicators
- $TICK, $TRIN, $VIX, $ADVN

## GUI Sections

### Sidebar
- **Connection Status**: Live status of all components
- **Quick Actions**: Refresh connections, view stats

### Main Tabs

#### Symbol Testing
- Symbol input with examples
- Real-time parsing results
- Storage routing recommendations
- Expected storage paths

#### Storage Inspector
- Storage system statistics
- Performance metrics
- Stored symbols browser
- Data visualization

#### Round-Trip Testing
- Complete pipeline verification
- Automated testing
- Step-by-step results
- Data integrity checks

#### Live Data Testing
- Real market data fetching
- Storage verification
- Data comparison
- Visual confirmation

## Troubleshooting

### Connection Issues
- Check if IQFeed/Polygon services are running
- Verify API credentials in config files
- Check network connectivity

### No Data Found
- Market might be closed
- Symbol might not exist
- Data source might be unavailable

### Storage Errors
- Check ArcticDB installation
- Verify storage directory permissions
- Check available disk space

## Technical Details

### Architecture
- **GUIDataInterface**: Wrapper around DataEngine
- **Symbol Parser Widget**: DTN symbol parsing display
- **Storage Viewer**: Data inspection and visualization
- **Streamlit App**: Main application orchestration

### Data Flow
1. User enters symbol → DTN Symbol Parser
2. Real data fetch → IQFeed/Polygon collectors
3. Data storage → FlexibleArcticStore via StorageRouter
4. Data retrieval → Same storage system
5. Verification → Compare original vs retrieved

### File Structure
```
gui/
├── streamlit_app.py              # Main Streamlit application
├── data_interface.py             # DataEngine wrapper
├── components/                   # Reusable UI components
│   ├── symbol_parser_widget.py  # Symbol parsing display
│   ├── storage_viewer.py         # Storage inspection tools
│   └── __init__.py
└── README.md                     # This file
```

## Configuration

The GUI automatically uses your existing configuration:
- `stage_01_data_engine/config/` - DataEngine configuration
- `config/` - Global project configuration
- Environment variables for API keys

## Performance

### Caching
- DataEngine instance is cached across sessions
- Connection status is checked efficiently
- Large datasets are paginated for display

### Resource Usage
- Minimal memory footprint
- Efficient data loading
- Automatic cleanup of temporary data

## Support

If you encounter issues:
1. Check the Streamlit console for error messages
2. Verify all dependencies are installed
3. Ensure configuration files are present
4. Test individual components outside the GUI

The GUI provides immediate visual confirmation that your flexible storage system is working correctly with real market data.