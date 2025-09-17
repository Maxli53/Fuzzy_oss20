import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  Paper,
  Grid,
  TextField,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tab,
  Tabs,
  Typography,
  IconButton,
  Chip,
  Alert
} from '@mui/material';
import { AgGridReact } from 'ag-grid-react';
import { ColDef, GridApi, GridReadyEvent } from 'ag-grid-community';
import Plot from 'react-plotly.js';
import { format, parseISO } from 'date-fns';
import RefreshIcon from '@mui/icons-material/Refresh';
import DownloadIcon from '@mui/icons-material/Download';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine-dark.css';

import { apiService } from '../services/api.service';
import { wsService, MessageType, TickData } from '../services/websocket.service';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const DataExplorer: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [barType, setBarType] = useState('time');
  const [interval, setInterval] = useState('1m');
  const [tabValue, setTabValue] = useState(0);
  const [tickData, setTickData] = useState<any[]>([]);
  const [barData, setBarData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);

  const gridApiRef = useRef<GridApi | null>(null);

  // AG-Grid column definitions for tick data
  const tickColumns: ColDef[] = [
    {
      field: 'timestamp',
      headerName: 'Time',
      width: 180,
      valueFormatter: (params) => {
        return params.value ? format(parseISO(params.value), 'HH:mm:ss.SSS') : '';
      },
      sortable: true,
      filter: true
    },
    {
      field: 'price',
      headerName: 'Price',
      width: 100,
      valueFormatter: (params) => `$${params.value?.toFixed(2)}`,
      cellClass: (params) => {
        if (!params.data.prevPrice) return '';
        return params.value > params.data.prevPrice ? 'price-up' : 'price-down';
      }
    },
    { field: 'size', headerName: 'Size', width: 80 },
    {
      field: 'bid',
      headerName: 'Bid',
      width: 100,
      valueFormatter: (params) => `$${params.value?.toFixed(2)}`
    },
    {
      field: 'ask',
      headerName: 'Ask',
      width: 100,
      valueFormatter: (params) => `$${params.value?.toFixed(2)}`
    },
    {
      field: 'spread',
      headerName: 'Spread',
      width: 90,
      valueGetter: (params) => {
        const spread = params.data.ask - params.data.bid;
        return spread.toFixed(4);
      }
    },
    {
      field: 'spread_bps',
      headerName: 'Spread (bps)',
      width: 110,
      valueGetter: (params) => {
        const spread = params.data.ask - params.data.bid;
        const bps = (spread / params.data.price) * 10000;
        return bps.toFixed(2);
      }
    },
    { field: 'exchange', headerName: 'Exchange', width: 100 },
    { field: 'conditions', headerName: 'Conditions', width: 120 }
  ];

  // AG-Grid column definitions for bar data
  const barColumns: ColDef[] = [
    {
      field: 'timestamp',
      headerName: 'Time',
      width: 180,
      valueFormatter: (params) => {
        return params.value ? format(parseISO(params.value), 'yyyy-MM-dd HH:mm') : '';
      }
    },
    {
      field: 'open',
      headerName: 'Open',
      width: 100,
      valueFormatter: (params) => `$${params.value?.toFixed(2)}`
    },
    {
      field: 'high',
      headerName: 'High',
      width: 100,
      valueFormatter: (params) => `$${params.value?.toFixed(2)}`
    },
    {
      field: 'low',
      headerName: 'Low',
      width: 100,
      valueFormatter: (params) => `$${params.value?.toFixed(2)}`
    },
    {
      field: 'close',
      headerName: 'Close',
      width: 100,
      valueFormatter: (params) => `$${params.value?.toFixed(2)}`
    },
    { field: 'volume', headerName: 'Volume', width: 120 },
    {
      field: 'vwap',
      headerName: 'VWAP',
      width: 100,
      valueFormatter: (params) => `$${params.value?.toFixed(2)}`
    }
  ];

  const loadTickData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.getTickData(symbol, {
        limit: 1000
      });
      setTickData(response.data);
    } catch (err: any) {
      setError(err.message || 'Failed to load tick data');
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  const loadBarData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiService.getBarData(symbol, barType, interval, {
        limit: 500
      });
      setBarData(response.data);
    } catch (err: any) {
      setError(err.message || 'Failed to load bar data');
    } finally {
      setLoading(false);
    }
  }, [symbol, barType, interval]);

  const toggleStreaming = () => {
    if (streaming) {
      wsService.unsubscribeFromSymbols([symbol]);
      setStreaming(false);
    } else {
      wsService.subscribeToSymbols([symbol]);
      setStreaming(true);
    }
  };

  const handleExport = () => {
    if (gridApiRef.current) {
      gridApiRef.current.exportDataAsCsv({
        fileName: `${symbol}_${tabValue === 0 ? 'ticks' : 'bars'}_${Date.now()}.csv`
      });
    }
  };

  useEffect(() => {
    if (tabValue === 0) {
      loadTickData();
    } else {
      loadBarData();
    }
  }, [symbol, tabValue, loadTickData, loadBarData]);

  useEffect(() => {
    // Subscribe to WebSocket tick updates
    const unsubscribe = wsService.onMessage(MessageType.TICK, (tick: TickData) => {
      if (tick.symbol === symbol && streaming) {
        setTickData(prev => {
          const newData = [tick, ...prev.slice(0, 999)];
          return newData;
        });
      }
    });

    return unsubscribe;
  }, [symbol, streaming]);

  const onGridReady = (params: GridReadyEvent) => {
    gridApiRef.current = params.api;
  };

  // Candlestick chart for bar data
  const candlestickChart = {
    data: [
      {
        type: 'candlestick',
        x: barData.map(d => d.timestamp),
        open: barData.map(d => d.open),
        high: barData.map(d => d.high),
        low: barData.map(d => d.low),
        close: barData.map(d => d.close),
        increasing: { line: { color: '#00ff41' } },
        decreasing: { line: { color: '#ff0041' } }
      }
    ],
    layout: {
      title: `${symbol} - ${interval} Bars`,
      xaxis: { title: 'Time', type: 'date' },
      yaxis: { title: 'Price ($)' },
      paper_bgcolor: '#1a1f2e',
      plot_bgcolor: '#0a0e1a',
      font: { color: '#ffffff' },
      height: 400
    }
  };

  // Volume chart
  const volumeChart = {
    data: [
      {
        type: 'bar',
        x: barData.map(d => d.timestamp),
        y: barData.map(d => d.volume),
        marker: { color: '#00bcd4' }
      }
    ],
    layout: {
      title: 'Volume',
      xaxis: { title: 'Time', type: 'date' },
      yaxis: { title: 'Volume' },
      paper_bgcolor: '#1a1f2e',
      plot_bgcolor: '#0a0e1a',
      font: { color: '#ffffff' },
      height: 200
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Paper sx={{ p: 2, mb: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} md={2}>
            <TextField
              fullWidth
              label="Symbol"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  tabValue === 0 ? loadTickData() : loadBarData();
                }
              }}
            />
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Bar Type</InputLabel>
              <Select
                value={barType}
                onChange={(e) => setBarType(e.target.value)}
                label="Bar Type"
              >
                <MenuItem value="time">Time</MenuItem>
                <MenuItem value="tick">Tick</MenuItem>
                <MenuItem value="volume">Volume</MenuItem>
                <MenuItem value="dollar">Dollar</MenuItem>
                <MenuItem value="renko">Renko</MenuItem>
                <MenuItem value="range">Range</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Interval</InputLabel>
              <Select
                value={interval}
                onChange={(e) => setInterval(e.target.value)}
                label="Interval"
              >
                <MenuItem value="1m">1 Minute</MenuItem>
                <MenuItem value="5m">5 Minutes</MenuItem>
                <MenuItem value="15m">15 Minutes</MenuItem>
                <MenuItem value="30m">30 Minutes</MenuItem>
                <MenuItem value="1h">1 Hour</MenuItem>
                <MenuItem value="4h">4 Hours</MenuItem>
                <MenuItem value="1d">Daily</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Button
                variant="contained"
                onClick={() => tabValue === 0 ? loadTickData() : loadBarData()}
                startIcon={<RefreshIcon />}
              >
                Refresh
              </Button>
              <Button
                variant={streaming ? "contained" : "outlined"}
                onClick={toggleStreaming}
                color={streaming ? "error" : "primary"}
              >
                {streaming ? 'Stop Streaming' : 'Start Streaming'}
              </Button>
              <IconButton onClick={handleExport} color="primary">
                <DownloadIcon />
              </IconButton>
              {streaming && (
                <Chip
                  label="LIVE"
                  color="error"
                  size="small"
                  sx={{ animation: 'pulse 1.5s infinite' }}
                />
              )}
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Paper>
        <Tabs value={tabValue} onChange={(e, v) => setTabValue(v)}>
          <Tab label="Tick Data" />
          <Tab label="Bar Data" />
          <Tab label="Charts" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <div className="ag-theme-alpine-dark" style={{ height: 600, width: '100%' }}>
            <AgGridReact
              rowData={tickData}
              columnDefs={tickColumns}
              onGridReady={onGridReady}
              defaultColDef={{
                resizable: true,
                sortable: true,
                filter: true
              }}
              animateRows={true}
              rowSelection="multiple"
            />
          </div>
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <div className="ag-theme-alpine-dark" style={{ height: 600, width: '100%' }}>
            <AgGridReact
              rowData={barData}
              columnDefs={barColumns}
              onGridReady={onGridReady}
              defaultColDef={{
                resizable: true,
                sortable: true,
                filter: true
              }}
            />
          </div>
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <Box>
            <Plot {...candlestickChart} style={{ width: '100%' }} />
            <Plot {...volumeChart} style={{ width: '100%' }} />
          </Box>
        </TabPanel>
      </Paper>
    </Box>
  );
};

export default DataExplorer;