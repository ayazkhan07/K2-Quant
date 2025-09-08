import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  TextField,
  Button,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  LinearProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Checkbox,
  Tooltip,
} from '@mui/material';
import {
  Download as DownloadIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
  CloudDownload as FetchIcon,
  Clear as ClearIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';

import { useAppDispatch, useAppSelector } from '../../hooks/redux';
import { fetchStockData, deleteTable, deleteAllTables } from '../../store/slices/stockSlice';
import { stockApi } from '../../services/api/stock';
import { formatNumber, formatDate } from '../../utils/formatters';

const TIME_RANGES = [
  { value: '1D', label: '1 Day' },
  { value: '1W', label: '1 Week' },
  { value: '1M', label: '1 Month' },
  { value: '3M', label: '3 Months' },
  { value: '6M', label: '6 Months' },
  { value: '1Y', label: '1 Year' },
  { value: '2Y', label: '2 Years' },
  { value: '5Y', label: '5 Years' },
  { value: '10Y', label: '10 Years' },
  { value: '20Y', label: '20 Years' },
];

const FREQUENCIES = [
  { value: '1min', label: '1 Minute' },
  { value: '5min', label: '5 Minutes' },
  { value: '15min', label: '15 Minutes' },
  { value: '30min', label: '30 Minutes' },
  { value: '1H', label: '1 Hour' },
  { value: 'D', label: 'Daily' },
  { value: 'W', label: 'Weekly' },
  { value: 'M', label: 'Monthly' },
];

export default function StockFetcher() {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const { tables, loading, error } = useAppSelector((state) => state.stock);
  
  const [symbol, setSymbol] = useState('');
  const [timeRange, setTimeRange] = useState('1M');
  const [frequency, setFrequency] = useState('D');
  const [marketHoursOnly, setMarketHoursOnly] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  useEffect(() => {
    loadTables();
  }, []);

  const loadTables = async () => {
    try {
      const response = await stockApi.getTables();
      dispatch({ type: 'stock/setTables', payload: response.data });
    } catch (error) {
      toast.error('Failed to load tables');
    }
  };

  const handleFetch = async () => {
    if (!symbol) {
      toast.warning('Please enter a stock symbol');
      return;
    }

    try {
      const result = await dispatch(
        fetchStockData({
          symbol: symbol.toUpperCase(),
          timeRange,
          frequency,
          marketHoursOnly,
        })
      ).unwrap();

      toast.success(`Fetched ${formatNumber(result.total_records)} records for ${symbol}`);
      loadTables();
    } catch (error: any) {
      toast.error(error.message || 'Failed to fetch stock data');
    }
  };

  const handleDeleteTable = async (tableName: string) => {
    setDeleteTarget(tableName);
    setDeleteDialogOpen(true);
  };

  const confirmDelete = async () => {
    if (!deleteTarget) return;

    try {
      if (deleteTarget === 'ALL') {
        await dispatch(deleteAllTables()).unwrap();
        toast.success('All tables deleted successfully');
      } else {
        await dispatch(deleteTable(deleteTarget)).unwrap();
        toast.success(`Table ${deleteTarget} deleted successfully`);
      }
      loadTables();
    } catch (error: any) {
      toast.error(error.message || 'Failed to delete table');
    } finally {
      setDeleteDialogOpen(false);
      setDeleteTarget(null);
    }
  };

  const handleAnalyze = (tableName: string) => {
    navigate(`/analysis/0?model=${tableName}`);
  };

  const handleExport = async (tableName: string) => {
    try {
      const response = await stockApi.exportTable(tableName, 'csv');
      const blob = new Blob([response.data], { type: 'text/csv' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${tableName}.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
      toast.success('Export completed successfully');
    } catch (error) {
      toast.error('Failed to export table');
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex' }}>
      {/* Sidebar */}
      <Paper
        sx={{
          width: 320,
          p: 3,
          borderRadius: 0,
          borderRight: '1px solid #1a1a1a',
        }}
      >
        <Typography variant="h6" gutterBottom>
          Stock Data Fetcher
        </Typography>

        <Box sx={{ mt: 3 }}>
          <TextField
            fullWidth
            label="Stock Symbol"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            placeholder="e.g., AAPL"
            sx={{ mb: 2 }}
          />

          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              label="Time Range"
              onChange={(e) => setTimeRange(e.target.value)}
            >
              {TIME_RANGES.map((range) => (
                <MenuItem key={range.value} value={range.value}>
                  {range.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Frequency</InputLabel>
            <Select
              value={frequency}
              label="Frequency"
              onChange={(e) => setFrequency(e.target.value)}
            >
              {FREQUENCIES.map((freq) => (
                <MenuItem key={freq.value} value={freq.value}>
                  {freq.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControlLabel
            control={
              <Checkbox
                checked={marketHoursOnly}
                onChange={(e) => setMarketHoursOnly(e.target.checked)}
              />
            }
            label="Market Hours Only (9:30 AM - 4:00 PM)"
            sx={{ mb: 3 }}
          />

          <Button
            fullWidth
            variant="contained"
            onClick={handleFetch}
            disabled={loading || !symbol}
            startIcon={<FetchIcon />}
            sx={{ mb: 2 }}
          >
            {loading ? 'Fetching...' : 'Fetch Data'}
          </Button>

          {loading && <LinearProgress sx={{ mb: 2 }} />}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
        </Box>

        {/* Actions */}
        <Box sx={{ mt: 4 }}>
          <Typography variant="subtitle2" gutterBottom>
            Quick Actions
          </Typography>
          <Button
            fullWidth
            variant="outlined"
            onClick={loadTables}
            startIcon={<RefreshIcon />}
            sx={{ mb: 1 }}
          >
            Refresh Tables
          </Button>
          <Button
            fullWidth
            variant="outlined"
            color="error"
            onClick={() => handleDeleteTable('ALL')}
            startIcon={<ClearIcon />}
          >
            Clear All Data
          </Button>
        </Box>
      </Paper>

      {/* Main Content */}
      <Box sx={{ flex: 1, p: 3, overflow: 'auto' }}>
        <Typography variant="h5" gutterBottom>
          Available Data Tables
        </Typography>

        {tables.length === 0 ? (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <Typography variant="body1" color="text.secondary">
              No data tables available. Fetch some stock data to get started.
            </Typography>
          </Paper>
        ) : (
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Range</TableCell>
                  <TableCell>Frequency</TableCell>
                  <TableCell align="right">Records</TableCell>
                  <TableCell>Date Range</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {tables.map((table) => (
                  <TableRow key={table.table_name}>
                    <TableCell>
                      <Chip label={table.symbol} color="primary" size="small" />
                    </TableCell>
                    <TableCell>{table.range}</TableCell>
                    <TableCell>{table.timespan}</TableCell>
                    <TableCell align="right">
                      {formatNumber(table.total_records)}
                    </TableCell>
                    <TableCell>{table.date_range}</TableCell>
                    <TableCell>{formatDate(table.created_at)}</TableCell>
                    <TableCell align="center">
                      <Tooltip title="Analyze">
                        <IconButton
                          size="small"
                          onClick={() => handleAnalyze(table.table_name)}
                        >
                          <AnalyticsIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Export CSV">
                        <IconButton
                          size="small"
                          onClick={() => handleExport(table.table_name)}
                        >
                          <DownloadIcon />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => handleDeleteTable(table.table_name)}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Box>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <Typography>
            {deleteTarget === 'ALL'
              ? 'Are you sure you want to delete all tables? This action cannot be undone.'
              : `Are you sure you want to delete table ${deleteTarget}? This action cannot be undone.`}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={confirmDelete} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}