import React, { useState, useEffect, useRef } from 'react';
import { Box, Paper, Typography, Divider } from '@mui/material';
import { useParams, useSearchParams } from 'react-router-dom';
import { toast } from 'react-toastify';

import LeftPane from './components/LeftPane';
import MiddlePane from './components/MiddlePane';
import RightPane from './components/RightPane';
import { useAppDispatch, useAppSelector } from '../../hooks/redux';
import { loadModel, applyIndicator, removeIndicator, applyStrategy, removeStrategy } from '../../store/slices/analysisSlice';
import { stockApi } from '../../services/api/stock';
import { indicatorApi } from '../../services/api/indicator';
import { strategyApi } from '../../services/api/strategy';

interface AnalysisState {
  currentModel: string | null;
  currentData: any[];
  metadata: any;
  appliedIndicators: Map<string, any>;
  activeStrategy: string | null;
  chartAggregation: string;
  viewRange: { start: Date | null; end: Date | null };
}

export default function Analysis() {
  const { tabId = '0' } = useParams();
  const [searchParams] = useSearchParams();
  const dispatch = useAppDispatch();
  
  const [state, setState] = useState<AnalysisState>({
    currentModel: null,
    currentData: [],
    metadata: {},
    appliedIndicators: new Map(),
    activeStrategy: null,
    chartAggregation: 'day',
    viewRange: { start: null, end: null },
  });

  const [leftPaneWidth, setLeftPaneWidth] = useState(280);
  const [rightPaneWidth, setRightPaneWidth] = useState(380);
  const leftResizeRef = useRef<HTMLDivElement>(null);
  const rightResizeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Load model from URL params if present
    const modelParam = searchParams.get('model');
    if (modelParam) {
      handleModelSelect(modelParam);
    }
  }, [searchParams]);

  const handleModelSelect = async (tableName: string) => {
    try {
      const response = await stockApi.getTableData(tableName, 500);
      const metadata = await stockApi.getTableInfo(tableName);
      
      setState(prev => ({
        ...prev,
        currentModel: tableName,
        currentData: response.data,
        metadata: metadata,
        appliedIndicators: new Map(),
        activeStrategy: null,
      }));

      dispatch(loadModel({ tableName, data: response.data, metadata }));
      toast.success(`Loaded model: ${tableName}`);
    } catch (error) {
      toast.error('Failed to load model');
    }
  };

  const handleIndicatorToggle = async (indicatorName: string, enabled: boolean) => {
    if (!state.currentModel) {
      toast.warning('No model loaded');
      return;
    }

    try {
      if (enabled) {
        // Apply indicator
        const params = extractIndicatorParams(indicatorName);
        const response = await indicatorApi.calculateIndicator(
          state.currentModel,
          indicatorName,
          params
        );
        
        setState(prev => {
          const newIndicators = new Map(prev.appliedIndicators);
          newIndicators.set(indicatorName, response.data);
          return { ...prev, appliedIndicators: newIndicators };
        });

        dispatch(applyIndicator({ name: indicatorName, data: response.data }));
        toast.success(`Applied indicator: ${indicatorName}`);
      } else {
        // Remove indicator
        setState(prev => {
          const newIndicators = new Map(prev.appliedIndicators);
          newIndicators.delete(indicatorName);
          return { ...prev, appliedIndicators: newIndicators };
        });

        dispatch(removeIndicator(indicatorName));
        toast.info(`Removed indicator: ${indicatorName}`);
      }
    } catch (error) {
      toast.error(`Failed to ${enabled ? 'apply' : 'remove'} indicator`);
    }
  };

  const handleStrategyToggle = async (strategyName: string, enabled: boolean) => {
    if (!state.currentModel) {
      toast.warning('No model loaded');
      return;
    }

    try {
      if (enabled) {
        // Apply strategy
        const response = await strategyApi.applyStrategy(
          state.currentModel,
          strategyName
        );
        
        setState(prev => ({
          ...prev,
          activeStrategy: strategyName,
        }));

        dispatch(applyStrategy({ name: strategyName, projections: response.projections }));
        toast.success(`Applied strategy: ${strategyName}`);
        
        // Reload data to include projections
        handleModelSelect(state.currentModel);
      } else {
        // Remove strategy
        await strategyApi.removeProjections(state.currentModel, strategyName);
        
        setState(prev => ({
          ...prev,
          activeStrategy: null,
        }));

        dispatch(removeStrategy());
        toast.info(`Removed strategy: ${strategyName}`);
        
        // Reload data without projections
        handleModelSelect(state.currentModel);
      }
    } catch (error) {
      toast.error(`Failed to ${enabled ? 'apply' : 'remove'} strategy`);
    }
  };

  const extractIndicatorParams = (indicatorName: string): any => {
    // Extract parameters from indicator name (e.g., "SMA (20)" -> { period: 20 })
    const match = indicatorName.match(/\((\d+)\)/);
    if (match) {
      return { period: parseInt(match[1]) };
    }
    
    // Default parameters for specific indicators
    const defaults: Record<string, any> = {
      'RSI': { period: 14 },
      'MACD': { fast: 12, slow: 26, signal: 9 },
      'BOLLINGER BANDS': { period: 20, std: 2 },
      'STOCHASTIC': { k_period: 14, d_period: 3 },
    };
    
    const baseName = indicatorName.split('(')[0].trim().toUpperCase();
    return defaults[baseName] || {};
  };

  const handleResize = (pane: 'left' | 'right', delta: number) => {
    if (pane === 'left') {
      setLeftPaneWidth(prev => Math.max(200, Math.min(400, prev + delta)));
    } else {
      setRightPaneWidth(prev => Math.max(300, Math.min(500, prev + delta)));
    }
  };

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ height: 40, display: 'flex', alignItems: 'center', px: 2, borderRadius: 0 }}>
        <Typography variant="subtitle1" color="text.secondary">
          K2 QUANT - ANALYSIS (Tab {tabId})
        </Typography>
      </Paper>

      {/* Main Content */}
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left Pane */}
        <Box sx={{ width: leftPaneWidth, position: 'relative' }}>
          <LeftPane
            onModelSelect={handleModelSelect}
            onIndicatorToggle={handleIndicatorToggle}
            onStrategyToggle={handleStrategyToggle}
            currentModel={state.currentModel}
            appliedIndicators={Array.from(state.appliedIndicators.keys())}
            activeStrategy={state.activeStrategy}
          />
          {/* Resize Handle */}
          <Box
            ref={leftResizeRef}
            sx={{
              position: 'absolute',
              right: 0,
              top: 0,
              bottom: 0,
              width: 4,
              cursor: 'col-resize',
              backgroundColor: 'transparent',
              '&:hover': { backgroundColor: 'primary.main' },
            }}
            onMouseDown={(e) => {
              const startX = e.clientX;
              const startWidth = leftPaneWidth;
              
              const handleMouseMove = (e: MouseEvent) => {
                const delta = e.clientX - startX;
                setLeftPaneWidth(Math.max(200, Math.min(400, startWidth + delta)));
              };
              
              const handleMouseUp = () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
              };
              
              document.addEventListener('mousemove', handleMouseMove);
              document.addEventListener('mouseup', handleMouseUp);
            }}
          />
        </Box>

        {/* Middle Pane */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <MiddlePane
            data={state.currentData}
            metadata={state.metadata}
            indicators={state.appliedIndicators}
            aggregation={state.chartAggregation}
            onAggregationChange={(agg) => setState(prev => ({ ...prev, chartAggregation: agg }))}
            viewRange={state.viewRange}
            onViewRangeChange={(range) => setState(prev => ({ ...prev, viewRange: range }))}
          />
        </Box>

        {/* Right Pane */}
        <Box sx={{ width: rightPaneWidth, position: 'relative' }}>
          {/* Resize Handle */}
          <Box
            ref={rightResizeRef}
            sx={{
              position: 'absolute',
              left: 0,
              top: 0,
              bottom: 0,
              width: 4,
              cursor: 'col-resize',
              backgroundColor: 'transparent',
              '&:hover': { backgroundColor: 'primary.main' },
            }}
            onMouseDown={(e) => {
              const startX = e.clientX;
              const startWidth = rightPaneWidth;
              
              const handleMouseMove = (e: MouseEvent) => {
                const delta = startX - e.clientX;
                setRightPaneWidth(Math.max(300, Math.min(500, startWidth + delta)));
              };
              
              const handleMouseUp = () => {
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
              };
              
              document.addEventListener('mousemove', handleMouseMove);
              document.addEventListener('mouseup', handleMouseUp);
            }}
          />
          <RightPane
            currentModel={state.currentModel}
            metadata={state.metadata}
            onStrategyGenerated={(name, code) => {
              // Save strategy and apply it
              toast.info(`Strategy "${name}" generated`);
            }}
            onProjectionRequested={(params) => {
              // Generate projections
              toast.info('Projection requested');
            }}
          />
        </Box>
      </Box>

      {/* Status Bar */}
      <Paper sx={{ height: 32, display: 'flex', alignItems: 'center', px: 2, borderRadius: 0, borderTop: '1px solid #1a1a1a' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="caption" sx={{ color: '#4a4' }}>●</Typography>
          <Typography variant="caption" color="text.secondary">
            {state.currentModel ? `Model: ${state.currentModel}` : 'No model loaded'}
          </Typography>
          <Divider orientation="vertical" flexItem />
          <Typography variant="caption" color="text.secondary">
            Ready
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
}