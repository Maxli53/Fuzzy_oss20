import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Provider } from 'react-redux';

import { store } from './store/store';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import DataExplorer from './pages/DataExplorer';
import ValidationDashboard from './pages/ValidationDashboard';
import BackfillManager from './pages/BackfillManager';
import Settings from './pages/Settings';

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ff41',
    },
    secondary: {
      main: '#00bcd4',
    },
    background: {
      default: '#0a0e1a',
      paper: '#1a1f2e',
    },
  },
  typography: {
    fontFamily: '"Roboto Mono", "Courier New", monospace',
  },
});

function App() {
  return (
    <Provider store={store}>
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <Router>
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/data-explorer" element={<DataExplorer />} />
              <Route path="/validation" element={<ValidationDashboard />} />
              <Route path="/backfill" element={<BackfillManager />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Layout>
        </Router>
      </ThemeProvider>
    </Provider>
  );
}

export default App;