import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme, CssBaseline } from '@mui/material';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Provider } from 'react-redux';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import { store } from './store';
import { WebSocketProvider } from './contexts/WebSocketContext';
import Layout from './components/Layout';
import LandingPage from './pages/Landing';
import StockFetcher from './pages/StockFetcher';
import Analysis from './pages/Analysis';
import { AuthProvider } from './contexts/AuthContext';

// Create dark theme
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00ffff',
    },
    secondary: {
      main: '#ff00ff',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#999999',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 300,
      letterSpacing: '0.05em',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 300,
      letterSpacing: '0.03em',
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 0,
          textTransform: 'none',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 0,
        },
      },
    },
  },
});

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  const [showLanding, setShowLanding] = useState(true);

  useEffect(() => {
    // Check if user has seen landing page
    const hasSeenLanding = localStorage.getItem('hasSeenLanding');
    if (hasSeenLanding === 'true') {
      setShowLanding(false);
    }
  }, []);

  const handleContinue = () => {
    localStorage.setItem('hasSeenLanding', 'true');
    setShowLanding(false);
  };

  if (showLanding) {
    return (
      <ThemeProvider theme={darkTheme}>
        <CssBaseline />
        <LandingPage onContinue={handleContinue} />
      </ThemeProvider>
    );
  }

  return (
    <Provider store={store}>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={darkTheme}>
          <CssBaseline />
          <AuthProvider>
            <WebSocketProvider>
              <Router>
                <Layout>
                  <Routes>
                    <Route path="/" element={<Navigate to="/stock-fetcher" />} />
                    <Route path="/stock-fetcher" element={<StockFetcher />} />
                    <Route path="/analysis/:tabId?" element={<Analysis />} />
                  </Routes>
                </Layout>
              </Router>
              <ToastContainer
                position="bottom-right"
                autoClose={5000}
                hideProgressBar={false}
                newestOnTop={false}
                closeOnClick
                rtl={false}
                pauseOnFocusLoss
                draggable
                pauseOnHover
                theme="dark"
              />
            </WebSocketProvider>
          </AuthProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </Provider>
  );
}

export default App;