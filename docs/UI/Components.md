# Frontend Components Documentation

## Table of Contents
1. [Component Architecture](#component-architecture)
2. [Core Components](#core-components)
3. [Data Display Components](#data-display-components)
4. [Chart Components](#chart-components)
5. [Form Components](#form-components)
6. [Layout Components](#layout-components)
7. [Utility Components](#utility-components)
8. [Hooks](#hooks)
9. [Context Providers](#context-providers)
10. [Component Guidelines](#component-guidelines)

## Component Architecture

### Component Structure
```
src/
├── components/
│   ├── common/        # Shared components
│   ├── charts/        # Chart components
│   ├── data/          # Data display components
│   ├── forms/         # Form components
│   ├── layout/        # Layout components
│   └── validation/    # Validation components
├── pages/             # Page components
├── hooks/             # Custom React hooks
├── context/           # Context providers
└── utils/             # Utility functions
```

## Core Components

### Dashboard Component
Primary dashboard displaying real-time market overview.

```typescript
interface DashboardProps {
  symbols: string[];
  refreshInterval?: number;
  layout?: 'grid' | 'list';
}

const Dashboard: React.FC<DashboardProps> = ({
  symbols,
  refreshInterval = 5000,
  layout = 'grid'
}) => {
  // Component implementation
}
```

**Features:**
- Real-time price updates
- Market statistics
- System health monitoring
- Active collection status

**Usage:**
```tsx
<Dashboard
  symbols={['AAPL', 'GOOGL', 'MSFT']}
  refreshInterval={3000}
  layout="grid"
/>
```

### DataExplorer Component
Advanced data exploration with AG-Grid integration.

```typescript
interface DataExplorerProps {
  symbol: string;
  dataType: 'tick' | 'bar';
  barType?: 'time' | 'volume' | 'dollar';
  interval?: string;
  streaming?: boolean;
}

const DataExplorer: React.FC<DataExplorerProps> = ({
  symbol,
  dataType,
  barType = 'time',
  interval = '1m',
  streaming = false
}) => {
  // Component implementation
}
```

**Features:**
- AG-Grid for tabular data
- Real-time streaming updates
- Export functionality
- Advanced filtering

**Usage:**
```tsx
<DataExplorer
  symbol="AAPL"
  dataType="tick"
  streaming={true}
/>
```

### ValidationDashboard Component
Data validation monitoring and alerts.

```typescript
interface ValidationDashboardProps {
  symbols: string[];
  autoRefresh?: boolean;
  showHistory?: boolean;
}

const ValidationDashboard: React.FC<ValidationDashboardProps> = ({
  symbols,
  autoRefresh = true,
  showHistory = false
}) => {
  // Component implementation
}
```

**Validation Types:**
1. Session Alignment
2. Storage Location
3. Timezone Consistency
4. Data Continuity

**Usage:**
```tsx
<ValidationDashboard
  symbols={['AAPL', 'GOOGL']}
  autoRefresh={true}
  showHistory={true}
/>
```

## Data Display Components

### TickGrid Component
Display tick data in a high-performance grid.

```typescript
interface TickGridProps {
  data: TickData[];
  columns?: ColDef[];
  onRowClick?: (row: TickData) => void;
  streaming?: boolean;
}

const TickGrid: React.FC<TickGridProps> = ({
  data,
  columns = defaultTickColumns,
  onRowClick,
  streaming = false
}) => {
  return (
    <div className="ag-theme-alpine-dark">
      <AgGridReact
        rowData={data}
        columnDefs={columns}
        onRowClicked={onRowClick}
        animateRows={streaming}
      />
    </div>
  );
}
```

**Default Columns:**
- Timestamp
- Price
- Size
- Bid/Ask
- Spread
- Exchange

### BarGrid Component
Display bar data with customizable intervals.

```typescript
interface BarGridProps {
  data: BarData[];
  interval: string;
  showVWAP?: boolean;
  showVolume?: boolean;
}

const BarGrid: React.FC<BarGridProps> = ({
  data,
  interval,
  showVWAP = true,
  showVolume = true
}) => {
  // Component implementation
}
```

### MarketDepth Component
Real-time order book visualization.

```typescript
interface MarketDepthProps {
  symbol: string;
  levels?: number;
  updateInterval?: number;
}

const MarketDepth: React.FC<MarketDepthProps> = ({
  symbol,
  levels = 10,
  updateInterval = 1000
}) => {
  // Component implementation
}
```

## Chart Components

### CandlestickChart Component
Interactive candlestick chart using Plotly.

```typescript
interface CandlestickChartProps {
  data: BarData[];
  symbol: string;
  interval: string;
  height?: number;
  showVolume?: boolean;
  indicators?: string[];
}

const CandlestickChart: React.FC<CandlestickChartProps> = ({
  data,
  symbol,
  interval,
  height = 400,
  showVolume = true,
  indicators = []
}) => {
  const chartData = {
    data: [{
      type: 'candlestick',
      x: data.map(d => d.timestamp),
      open: data.map(d => d.open),
      high: data.map(d => d.high),
      low: data.map(d => d.low),
      close: data.map(d => d.close)
    }],
    layout: {
      title: `${symbol} - ${interval}`,
      height: height
    }
  };

  return <Plot {...chartData} />;
}
```

**Supported Indicators:**
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- Bollinger Bands
- RSI
- MACD
- Volume

### VolumeChart Component
Volume histogram with color coding.

```typescript
interface VolumeChartProps {
  data: BarData[];
  height?: number;
  colorScheme?: 'default' | 'heatmap';
}

const VolumeChart: React.FC<VolumeChartProps> = ({
  data,
  height = 200,
  colorScheme = 'default'
}) => {
  // Component implementation
}
```

### SpreadChart Component
Bid-ask spread visualization over time.

```typescript
interface SpreadChartProps {
  data: TickData[];
  displayType?: 'absolute' | 'bps';
  showMean?: boolean;
}

const SpreadChart: React.FC<SpreadChartProps> = ({
  data,
  displayType = 'bps',
  showMean = true
}) => {
  // Component implementation
}
```

### HeatmapChart Component
Multi-symbol correlation heatmap.

```typescript
interface HeatmapChartProps {
  symbols: string[];
  metric: 'price' | 'volume' | 'correlation';
  timeframe: string;
}

const HeatmapChart: React.FC<HeatmapChartProps> = ({
  symbols,
  metric,
  timeframe
}) => {
  // Component implementation
}
```

## Form Components

### SymbolSelector Component
Autocomplete symbol selection.

```typescript
interface SymbolSelectorProps {
  value: string;
  onChange: (symbol: string) => void;
  multiple?: boolean;
  placeholder?: string;
}

const SymbolSelector: React.FC<SymbolSelectorProps> = ({
  value,
  onChange,
  multiple = false,
  placeholder = 'Enter symbol...'
}) => {
  const [options, setOptions] = useState<string[]>([]);

  return (
    <Autocomplete
      value={value}
      onChange={(_, newValue) => onChange(newValue)}
      options={options}
      multiple={multiple}
      renderInput={(params) => (
        <TextField {...params} placeholder={placeholder} />
      )}
    />
  );
}
```

### DateRangePicker Component
Date range selection for data queries.

```typescript
interface DateRangePickerProps {
  startDate: Date;
  endDate: Date;
  onChange: (start: Date, end: Date) => void;
  maxRange?: number;
}

const DateRangePicker: React.FC<DateRangePickerProps> = ({
  startDate,
  endDate,
  onChange,
  maxRange = 30
}) => {
  // Component implementation
}
```

### IntervalSelector Component
Bar interval selection.

```typescript
interface IntervalSelectorProps {
  value: string;
  onChange: (interval: string) => void;
  barType: string;
}

const IntervalSelector: React.FC<IntervalSelectorProps> = ({
  value,
  onChange,
  barType
}) => {
  const intervals = {
    time: ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
    tick: ['100', '500', '1000', '5000'],
    volume: ['1000', '5000', '10000', '50000'],
    dollar: ['10000', '50000', '100000', '500000']
  };

  return (
    <Select value={value} onChange={(e) => onChange(e.target.value)}>
      {intervals[barType].map(interval => (
        <MenuItem key={interval} value={interval}>
          {interval}
        </MenuItem>
      ))}
    </Select>
  );
}
```

## Layout Components

### Layout Component
Main application layout with navigation.

```typescript
interface LayoutProps {
  children: React.ReactNode;
  user?: User;
}

const Layout: React.FC<LayoutProps> = ({ children, user }) => {
  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar position="fixed">
        <Toolbar>
          <Typography variant="h6">Fuzzy OSS20</Typography>
          <Navigation />
          {user && <UserMenu user={user} />}
        </Toolbar>
      </AppBar>
      <Sidebar />
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        {children}
      </Box>
    </Box>
  );
}
```

### Sidebar Component
Navigation sidebar with collapsible menu.

```typescript
interface SidebarProps {
  open?: boolean;
  onToggle?: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({
  open = true,
  onToggle
}) => {
  const menuItems = [
    { label: 'Dashboard', path: '/dashboard', icon: <DashboardIcon /> },
    { label: 'Data Explorer', path: '/data-explorer', icon: <DataIcon /> },
    { label: 'Validation', path: '/validation', icon: <CheckIcon /> },
    { label: 'Backfill', path: '/backfill', icon: <BackfillIcon /> },
    { label: 'Settings', path: '/settings', icon: <SettingsIcon /> }
  ];

  return (
    <Drawer variant="persistent" open={open}>
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.path} component={Link} to={item.path}>
            <ListItemIcon>{item.icon}</ListItemIcon>
            <ListItemText primary={item.label} />
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
}
```

### TabPanel Component
Tabbed content container.

```typescript
interface TabPanelProps {
  tabs: Array<{
    label: string;
    content: React.ReactNode;
    disabled?: boolean;
  }>;
  defaultTab?: number;
}

const TabPanel: React.FC<TabPanelProps> = ({
  tabs,
  defaultTab = 0
}) => {
  const [value, setValue] = useState(defaultTab);

  return (
    <Box>
      <Tabs value={value} onChange={(_, v) => setValue(v)}>
        {tabs.map((tab, index) => (
          <Tab
            key={index}
            label={tab.label}
            disabled={tab.disabled}
          />
        ))}
      </Tabs>
      {tabs.map((tab, index) => (
        <Box key={index} hidden={value !== index}>
          {tab.content}
        </Box>
      ))}
    </Box>
  );
}
```

## Utility Components

### LoadingSpinner Component
Loading indicator with optional message.

```typescript
interface LoadingSpinnerProps {
  message?: string;
  size?: 'small' | 'medium' | 'large';
  fullScreen?: boolean;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Loading...',
  size = 'medium',
  fullScreen = false
}) => {
  const spinner = (
    <Box sx={{ textAlign: 'center' }}>
      <CircularProgress size={size} />
      {message && <Typography>{message}</Typography>}
    </Box>
  );

  return fullScreen ? (
    <Backdrop open={true}>{spinner}</Backdrop>
  ) : spinner;
}
```

### ErrorBoundary Component
Error boundary for graceful error handling.

```typescript
interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: Error }>;
}

class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  { hasError: boolean; error?: Error }
> {
  state = { hasError: false, error: undefined };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      const Fallback = this.props.fallback || DefaultErrorFallback;
      return <Fallback error={this.state.error!} />;
    }
    return this.props.children;
  }
}
```

### NotificationBar Component
Toast-style notifications.

```typescript
interface NotificationBarProps {
  message: string;
  severity: 'success' | 'error' | 'warning' | 'info';
  duration?: number;
  onClose?: () => void;
}

const NotificationBar: React.FC<NotificationBarProps> = ({
  message,
  severity,
  duration = 6000,
  onClose
}) => {
  return (
    <Snackbar
      open={true}
      autoHideDuration={duration}
      onClose={onClose}
    >
      <Alert severity={severity} onClose={onClose}>
        {message}
      </Alert>
    </Snackbar>
  );
}
```

## Hooks

### useWebSocket Hook
WebSocket connection management.

```typescript
interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnectInterval?: number;
}

function useWebSocket(options?: UseWebSocketOptions) {
  const [connected, setConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);

  const subscribe = useCallback((symbols: string[]) => {
    wsService.subscribeToSymbols(symbols);
  }, []);

  const unsubscribe = useCallback((symbols: string[]) => {
    wsService.unsubscribeFromSymbols(symbols);
  }, []);

  useEffect(() => {
    if (options?.autoConnect) {
      wsService.connect().then(() => setConnected(true));
    }

    return () => {
      wsService.disconnect();
    };
  }, []);

  return {
    connected,
    lastMessage,
    subscribe,
    unsubscribe
  };
}
```

### useTickData Hook
Fetch and manage tick data.

```typescript
interface UseTickDataOptions {
  symbol: string;
  limit?: number;
  streaming?: boolean;
}

function useTickData(options: UseTickDataOptions) {
  const [data, setData] = useState<TickData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    fetchTickData();
    if (options.streaming) {
      setupStreaming();
    }
  }, [options.symbol]);

  return { data, loading, error, refetch };
}
```

### useValidation Hook
Validation status management.

```typescript
function useValidation(symbols: string[]) {
  const [validations, setValidations] = useState<ValidationResult[]>([]);
  const [loading, setLoading] = useState(false);

  const validate = async () => {
    setLoading(true);
    const results = await apiService.validateSymbols(symbols);
    setValidations(results);
    setLoading(false);
  };

  return { validations, loading, validate };
}
```

### useTheme Hook
Theme management and switching.

```typescript
function useTheme() {
  const [theme, setTheme] = useState<'light' | 'dark'>('dark');

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light');
  };

  return { theme, toggleTheme };
}
```

## Context Providers

### AppContext Provider
Global application state.

```typescript
interface AppContextValue {
  user: User | null;
  settings: AppSettings;
  updateSettings: (settings: Partial<AppSettings>) => void;
}

const AppContext = createContext<AppContextValue | undefined>(undefined);

export const AppProvider: React.FC<{ children: ReactNode }> = ({
  children
}) => {
  const [user, setUser] = useState<User | null>(null);
  const [settings, setSettings] = useState<AppSettings>(defaultSettings);

  return (
    <AppContext.Provider value={{ user, settings, updateSettings }}>
      {children}
    </AppContext.Provider>
  );
}
```

### DataContext Provider
Centralized data management.

```typescript
interface DataContextValue {
  symbols: string[];
  selectedSymbol: string;
  tickData: Map<string, TickData[]>;
  barData: Map<string, BarData[]>;
  subscribeSymbol: (symbol: string) => void;
  unsubscribeSymbol: (symbol: string) => void;
}

const DataContext = createContext<DataContextValue | undefined>(undefined);
```

## Component Guidelines

### Best Practices

1. **Performance Optimization**
   - Use React.memo for expensive renders
   - Implement virtualization for large datasets
   - Debounce user inputs
   - Lazy load heavy components

2. **State Management**
   - Keep state as local as possible
   - Use context for cross-cutting concerns
   - Implement Redux for complex state

3. **Error Handling**
   - Wrap components in error boundaries
   - Provide meaningful error messages
   - Implement retry mechanisms

4. **Accessibility**
   - Use semantic HTML elements
   - Provide ARIA labels
   - Ensure keyboard navigation
   - Test with screen readers

5. **Testing**
   - Unit test individual components
   - Integration test component interactions
   - E2E test critical user flows

### Component Template

```typescript
import React, { useState, useEffect, useCallback, memo } from 'react';
import { Box, Typography } from '@mui/material';
import { useTranslation } from 'react-i18next';

interface ComponentNameProps {
  // Props definition
  required: string;
  optional?: number;
  children?: React.ReactNode;
}

const ComponentName: React.FC<ComponentNameProps> = memo(({
  required,
  optional = 0,
  children
}) => {
  const { t } = useTranslation();
  const [state, setState] = useState<any>(null);

  useEffect(() => {
    // Side effects
    return () => {
      // Cleanup
    };
  }, [/* dependencies */]);

  const handleAction = useCallback(() => {
    // Event handler
  }, [/* dependencies */]);

  return (
    <Box>
      <Typography>{t('component.title')}</Typography>
      {children}
    </Box>
  );
});

ComponentName.displayName = 'ComponentName';

export default ComponentName;
```

### Styling Guidelines

1. **Material-UI Theme**
   ```typescript
   const theme = createTheme({
     palette: {
       mode: 'dark',
       primary: { main: '#00ff41' },
       secondary: { main: '#00bcd4' }
     }
   });
   ```

2. **CSS-in-JS**
   ```typescript
   const useStyles = makeStyles((theme) => ({
     root: {
       padding: theme.spacing(2),
       backgroundColor: theme.palette.background.paper
     }
   }));
   ```

3. **Responsive Design**
   ```typescript
   <Grid container spacing={2}>
     <Grid item xs={12} md={6} lg={4}>
       {/* Content */}
     </Grid>
   </Grid>
   ```

---

*Component Library Version: 1.0.0*
*Last Updated: January 17, 2025*