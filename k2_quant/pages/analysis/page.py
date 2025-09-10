"""
K2 Quant Analysis Page - Harmonized Orchestrator

Coordinates the three pane components without containing UI logic.
Save as: k2_quant/pages/analysis/page.py
"""

import re
from typing import Dict, Optional, Any, List, Union
import uuid
import time
from datetime import datetime
import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QLabel, QStatusBar)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.services.technical_analysis_service import ta_service
from k2_quant.utilities.services.stock_data_service import stock_service
from k2_quant.utilities.data.saved_models_manager import saved_models_manager
from k2_quant.utilities.services.strategy_service import strategy_service
from k2_quant.utilities.services.dynamic_python_engine import dpe_service

# Import the three pane components
from k2_quant.pages.analysis.components.left_pane import LeftPaneWidget
from k2_quant.pages.analysis.components.middle_pane import MiddlePaneWidget
from k2_quant.pages.analysis.components.right_pane import RightPaneWidget


class AnalysisPageWidget(QWidget):
    """Main Analysis page widget - orchestrates three pane components"""
    
    # Signals
    back_to_stock_fetcher = pyqtSignal()
    
    def __init__(self, tab_id: int = 0, parent=None):
        super().__init__(parent)
        self.tab_id = tab_id
        self.current_model = None
        self.current_data = None
        self.current_metadata = {}
        self.applied_indicators = {}
        
        self.init_ui()
        self.setup_styling()
        self.load_saved_models()
        
        k2_logger.info(f"Analysis page initialized (Tab ID: {tab_id})", "ANALYSIS")
    
    def init_ui(self):
        """Initialize the UI layout and create pane components"""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        
        # Create header
        self.create_header(main_layout)
        
        # Create splitter for three panes
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #1a1a1a;
            }
        """)
        
        # Create the three pane components
        self.left_pane = LeftPaneWidget()
        self.middle_pane = MiddlePaneWidget()
        self.right_pane = RightPaneWidget()
        
        # Wire up signals
        self.setup_connections()
        
        # Add to splitter
        self.splitter.addWidget(self.left_pane)
        self.splitter.addWidget(self.middle_pane)
        self.splitter.addWidget(self.right_pane)
        
        # Set sizes - left: 280px, middle: flexible, right: 513px
        self.splitter.setSizes([280, 607, 513])
        
        main_layout.addWidget(self.splitter)
        
        # Create status bar
        self.status_widget = self.create_status_bar()
        main_layout.addWidget(self.status_widget)
        
        # Load initial data
        self.refresh_left_pane_data()
    
    def create_header(self, parent_layout):
        """Create header with title"""
        header = QWidget()
        header.setFixedHeight(40)
        header.setObjectName("analysisHeader")
        
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 0, 20, 0)
        header.setLayout(header_layout)
        
        title = QLabel(f"K2 QUANT - ANALYSIS (Tab {self.tab_id})")
        title.setFont(QFont("Arial", 14))
        title.setStyleSheet("color: #999; letter-spacing: 1px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        parent_layout.addWidget(header)
    
    def create_status_bar(self) -> QWidget:
        """Create status bar"""
        widget = QWidget()
        widget.setFixedHeight(32)
        widget.setObjectName("analysisStatusBar")
        
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        widget.setLayout(layout)
        
        # Status indicator
        indicator = QLabel("●")
        indicator.setStyleSheet("color: #4a4; font-size: 8px;")
        layout.addWidget(indicator)
        
        # Model info
        self.model_label = QLabel("No model loaded")
        self.model_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.model_label)
        
        layout.addStretch()
        
        # System status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.status_label)
        
        return widget
    
    def setup_connections(self):
        """Wire up all signals between components"""
        # Left pane connections
        self.left_pane.model_selected.connect(self.load_model_by_table)
        self.left_pane.strategy_toggled.connect(self.on_strategy_toggled)
        self.left_pane.indicator_toggled.connect(self.on_indicator_toggled)
        
        # Middle pane connections
        self.middle_pane.column_toggled.connect(self.on_column_toggled)
        self.middle_pane.view_mode_changed.connect(self.on_view_mode_changed)
        self.middle_pane.projection_requested.connect(self.on_projection_requested)
        self.middle_pane.indicator_applied.connect(self.on_indicator_applied_from_middle)
        self.middle_pane.data_exported.connect(self.on_data_exported)
        
        # Right pane connections
        self.right_pane.message_sent.connect(self.on_ai_message_sent)
        self.right_pane.strategy_generated.connect(self.on_strategy_generated)
        self.right_pane.projection_requested.connect(self.on_projection_requested_from_ai)
    
    def refresh_left_pane_data(self):
        """Refresh all data in the left pane"""
        try:
            # Load saved models
            models = saved_models_manager.get_saved_models()
            self.left_pane.populate_models(models)
            
            # Load strategies
            try:
                strategies = strategy_service.get_all_strategies()
                self.left_pane.populate_strategies(strategies)
            except Exception:
                self.left_pane.populate_strategies([])
            
            k2_logger.info("Left pane data refreshed", "ANALYSIS")
            
        except Exception as e:
            k2_logger.error(f"Failed to refresh left pane data: {str(e)}", "ANALYSIS")
    
    def load_model_by_table(self, table_name: str):
        """Load model by table name"""
        k2_logger.info(f"Loading model: {table_name}", "ANALYSIS")
        
        try:
            # Get data from table (limit to last 500 rows for performance)
            rows, total_count = stock_service.get_display_data(table_name, limit=500)
            # DIAGNOSTIC: display fetch shape and schema
            try:
                total_rows = len(rows)
                first_row_cols = len(rows[0]) if rows else 0
                k2_logger.debug(f"DISPLAY_FETCH: table={table_name} rows={total_rows} first_row_cols={first_row_cols}", "ANALYSIS")
                if rows:
                    k2_logger.debug(f"DISPLAY_FETCH: first_row_preview={rows[0]}", "ANALYSIS")
                try:
                    ind_cols = stock_service.db.get_indicator_columns(table_name) if hasattr(stock_service, 'db') else []
                    k2_logger.debug(f"DISPLAY_FETCH: indicator_columns={ind_cols}", "ANALYSIS")
                except Exception as ex:
                    k2_logger.debug(f"DISPLAY_FETCH: indicator_columns not available ({ex})", "ANALYSIS")
            except Exception as ex:
                k2_logger.error(f"DISPLAY_FETCH diagnostics failed: {ex}", "ANALYSIS")
            
            if rows:
                # Update current state
                self.current_model = table_name
                self.current_data = rows
                
                # Gather metadata
                base_metadata = saved_models_manager.get_model_metadata(table_name) or {'symbol': table_name}
                table_info = stock_service.get_table_info(table_name) or {}
                date_range = table_info.get('date_range')
                
                # Merge metadata while keeping existing keys stable
                self.current_metadata = dict(base_metadata)
                self.current_metadata.update({
                    'records': total_count,
                    'table_name': table_name,
                    'total_records': total_count,
                    'date_range': date_range
                })
                
                # Update status
                self.model_label.setText(f"Model: {table_name} ({total_count:,} records)")
                self.status_label.setText("Model loaded")
                
                # Load into middle pane (chart uses limited data; table uses limited data)
                self.middle_pane.load_data(rows, self.current_metadata)
                
                # Build a small DF sample for AI grounding
                try:
                    sample_rows, sample_cols, _ = stock_service.get_display_data_with_indicators(table_name, limit=50)
                    df_sample = pd.DataFrame(sample_rows, columns=sample_cols)
                except Exception:
                    df_sample = None

                # Update AI context (include df sample so the AI can see columns/recent rows)
                ctx = {
                    'symbol': self.current_metadata.get('symbol', table_name),
                    'records': total_count,
                    'table_name': table_name,
                    'date_range': date_range,
                    'df': df_sample
                }
                self.right_pane.set_data_context(ctx)
                
                # Clear any existing indicators/strategies when loading new model
                self.left_pane.clear_all_indicators()
                self.left_pane.clear_all_strategies()
                
                # Restore model state if available
                try:
                    state = saved_models_manager.get_model_state(table_name)
                    if state and self.middle_pane.chart_widget:
                        k2_logger.info(f"Model state available for {table_name}", "ANALYSIS")
                        # Restore aggregation and view range if present
                        cw = self.middle_pane.chart_widget
                        agg = state.get('aggregation') if isinstance(state, dict) else None
                        vr  = state.get('view_range') if isinstance(state, dict) else None
                        if vr:
                            cw.set_view_range(vr)
                        if agg:
                            cw.change_timeframe(agg)
                except Exception as e:
                    k2_logger.debug(f"No model state available: {e}", "ANALYSIS")

                # Wire persistence of aggregation and view range
                try:
                    if self.middle_pane.chart_widget:
                        cw = self.middle_pane.chart_widget
                        cw.timeframe_changed.connect(
                            lambda tf, tn=table_name: saved_models_manager.set_model_state(
                                tn, indicators=None, active_strategy=None, chart_range=None
                            )
                        )
                        cw.view_range_changed.connect(
                            lambda vr, tn=table_name: saved_models_manager.set_model_state(
                                tn, indicators=None, active_strategy=None, chart_range=None
                            )
                        )
                except Exception:
                    pass
                    
        except Exception as e:
            k2_logger.error(f"Failed to load model: {str(e)}", "ANALYSIS")
            self.status_label.setText("Error loading model")
    
    def on_indicator_toggled(self, indicator_name: str, enabled: bool):
        """Handle indicator toggle from left pane"""
        if not self.current_model:
            k2_logger.warning("No model loaded for indicator toggle", "ANALYSIS")
            return
        
        k2_logger.info(f"Indicator '{indicator_name}' toggled to {enabled}", "ANALYSIS")
        cid = self._new_cid()
        self._log_indicator("indicator_toggle", cid, indicator=indicator_name, enabled=enabled)
        
        if enabled:
            # Apply indicator
            params = self.extract_default_indicator_params(indicator_name)
            self.apply_indicator(indicator_name, params, cid=cid)
        else:
            # Remove indicator
            self.remove_indicator(indicator_name, cid=cid)
    
    def extract_default_indicator_params(self, indicator_name: str) -> Dict:
        """Normalize UI label to TA-Lib parameter keys."""
        params: Dict[str, Any] = {}
        base = indicator_name.split("(")[0].strip().upper()

        # Parse number in parentheses as timeperiod
        period = None
        if "(" in indicator_name and ")" in indicator_name:
            match = re.search(r'\((\d+)\)', indicator_name)
            if match:
                period = int(match.group(1))

        if base in ("SMA", "EMA", "WMA", "KAMA", "T3", "TEMA", "DEMA", "RSI", "CCI", "ATR", "WILLR", "MFI", "TRIX"):
            params["timeperiod"] = period if period is not None else {
                "SMA": 20, "EMA": 20, "RSI": 14, "CCI": 14, "ATR": 14, "WILLR": 14, "MFI": 14,
                "KAMA": 30, "T3": 5, "TEMA": 30, "DEMA": 30, "TRIX": 30, "WMA": 20
            }.get(base, 20)
        elif base == "MACD":
            params.update({"fastperiod": 12, "slowperiod": 26, "signalperiod": 9})
        elif base in ("BBANDS", "BOLLINGER BANDS"):
            params.update({"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2})
        elif base in ("STOCH", "STOCHASTIC"):
            params.update({"fastk_period": 5, "slowk_period": 3, "slowd_period": 3})
        elif base == "VWAP":
            pass
        elif base == "SAR":
            params.update({"acceleration": 0.02, "maximum": 0.2})
        else:
            if period is not None:
                params["timeperiod"] = period

        return params
    
    def apply_indicator(self, indicator_name: str, params: Dict, cid: str | None = None):
        """Apply indicator with correct routing, persistence, and table refresh."""
        try:
            df = stock_service.get_full_dataframe(self.current_model)
            # Debug: capture raw timestamp dtype from DB fetch
            try:
                k2_logger.info(str({'event': 'df_timestamp_dtype', 'dtype': str(df['timestamp'].dtype)}), "INDICATOR")
            except Exception:
                pass
            if df is None or df.empty:
                return

            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC')
            df.set_index('datetime', inplace=True)

            base = indicator_name.split("(")[0].strip().upper()
            base = "BBANDS" if base == "BOLLINGER BANDS" else base
            if base == "STOCHASTIC":  # normalize for consistent pane/columns
                base = "STOCH"
            # Base columns should not persist/drop
            if base in ("VOLUME", "VWAP"):
                # Render from existing data only
                if base == "VOLUME":
                    series = df['volume'] if 'volume' in df.columns else None
                    if series is not None:
                        self.middle_pane.add_indicator_to_pane("VOLUME", {"Volume": series}, family="VOLUME")
                    else:
                        k2_logger.warning("Volume column not found; cannot render Volume pane", "ANALYSIS")
                else:
                    series = df['vwap'] if 'vwap' in df.columns else None
                    if series is not None:
                        self.middle_pane.add_indicator_overlay("VWAP", series)
                    else:
                        k2_logger.warning("VWAP column not found; cannot render VWAP overlay", "ANALYSIS")
                self.applied_indicators[indicator_name] = params or {}
                return

            meta = ta_service.get_indicator_info(base)
            start_compute = time.perf_counter()
            calc = ta_service.calculate_indicator(df, base, params)
            if calc is None:
                return
            compute_ms = round((time.perf_counter() - start_compute) * 1000, 2)
            try:
                if isinstance(calc, dict):
                    outputs = []
                    for k, s in calc.items():
                        try:
                            outputs.append({'name': k, 'len': int(len(s)), 'nan': int(getattr(s, 'isna', lambda: [])().sum())})
                        except Exception:
                            outputs.append({'name': k})
                    self._log_indicator("indicator_computed", cid, indicator=indicator_name, base=base, outputs=outputs, compute_ms=compute_ms)
                else:
                    try:
                        n = int(len(calc)); nan = int(getattr(calc, 'isna', lambda: [])().sum())
                    except Exception:
                        n = None; nan = None
                    self._log_indicator("indicator_computed", cid, indicator=indicator_name, base=base, len=n, nan=nan, compute_ms=compute_ms)
            except Exception:
                pass

            # Persist first, then capture state, then plot; refresh table in finally
            try:
                pre_cols = set(stock_service.db.get_indicator_columns(self.current_model))
            except Exception:
                pre_cols = set()
            if isinstance(calc, dict):
                if base == "BBANDS":
                    for out_name in ("upper", "middle", "lower"):
                        series = calc.get(out_name)
                        if series is None:
                            continue
                        col = self._indicator_column_name(base, params, output=out_name)
                        affected = self._persist_indicator_series(col, df['timestamp'], series)
                        self._log_indicator("indicator_persist", cid, indicator=indicator_name, column=col, rows_updated=int(affected))
                else:
                    for out_name, series in calc.items():
                        col = self._indicator_column_name(base, params, output=out_name)
                        affected = self._persist_indicator_series(col, df['timestamp'], series)
                        self._log_indicator("indicator_persist", cid, indicator=indicator_name, column=col, rows_updated=int(affected))
            else:
                col = self._indicator_column_name(base, params)
                affected = self._persist_indicator_series(col, df['timestamp'], calc)
                self._log_indicator("indicator_persist", cid, indicator=indicator_name, column=col, rows_updated=int(affected))

            # Record state (even if plotting fails later)
            self.applied_indicators[indicator_name] = params or {}

            try:
                if isinstance(calc, dict):
                    if base == "BBANDS":
                        for out_name in ("upper", "middle", "lower"):
                            series = calc.get(out_name)
                            if series is None:
                                continue
                            overlay_label = f"BBANDS ({out_name.title()})"
                            self.middle_pane.add_indicator_overlay(overlay_label, series, cid=cid)
                            self._log_indicator("indicator_render_overlay", cid, indicator=overlay_label, where="main", status="success")
                    else:
                        if base == "MACD":
                            series_map = {"MACD Line": calc.get("line"),
                                          "MACD Signal": calc.get("signal"),
                                          "MACD Hist": calc.get("hist")}
                            series_map = {k: v for k, v in series_map.items() if v is not None}
                        elif base in ("STOCH", "STOCHRSI"):
                            series_map = {"%K": calc.get("k"), "%D": calc.get("d")}
                            series_map = {k: v for k, v in series_map.items() if v is not None}
                        else:
                            series_map = {f"{base}-{k}".upper(): v for k, v in calc.items() if v is not None}

                        if series_map:
                            self.middle_pane.add_indicator_to_pane(
                                pane_key=base,
                                series_map=series_map,
                                family=base,
                                cid=cid
                            )
                            self._log_indicator("indicator_render_pane", cid, pane_key=base, series=list(series_map.keys()), status="success")
                else:
                    if meta and getattr(meta, "pane", "main") == "main":
                        self.middle_pane.add_indicator_overlay(indicator_name, calc, cid=cid)
                        self._log_indicator("indicator_render_overlay", cid, indicator=indicator_name, where="main", status="success")
                    else:
                        self.middle_pane.add_indicator_to_pane(
                            pane_key=base,
                            series_map={indicator_name: calc},
                            family=base,
                            cid=cid
                        )
                        self._log_indicator("indicator_render_pane", cid, pane_key=base, series=[indicator_name], status="success")
            finally:
                try:
                    self._refresh_table_only(limit=500)
                    try:
                        post_cols = set(stock_service.db.get_indicator_columns(self.current_model))
                    except Exception:
                        post_cols = set()
                    added = sorted(list(post_cols - pre_cols))
                    if added:
                        self._log_indicator("table_columns_added", cid, added=added, now=sorted(list(post_cols)))
                except AttributeError:
                    self._reload_display_data()

        except Exception as e:
            k2_logger.error(f"Error applying indicator {indicator_name}: {e}", "ANALYSIS")
    
    def remove_indicator(self, indicator_name: str, cid: str | None = None):
        """Remove indicator from UI and drop DB columns; then refresh table."""
        try:
            base = indicator_name.split("(")[0].strip().upper()
            # Normalize aliases to canonical keys used by the chart
            base = "BBANDS" if base == "BOLLINGER BANDS" else base
            base = "STOCH" if base == "STOCHASTIC" else base
            if base in ("VOLUME", "VWAP"):
                # UI cleanup only
                self.middle_pane.remove_indicator(indicator_name if base == "VWAP" else "Volume", family=base, cid=cid)
                if indicator_name in self.applied_indicators:
                    del self.applied_indicators[indicator_name]
                self._log_indicator("indicator_ui_removed", cid, indicator=indicator_name, base=base, status="success")
                return

            # UI cleanup for non-base
            if base == "BBANDS":
                # Remove all BBANDS overlays explicitly
                for label in ("BBANDS (Upper)", "BBANDS (Middle)", "BBANDS (Lower)"):
                    self.middle_pane.remove_indicator(label, family=None, cid=cid)
                self._log_indicator("indicator_ui_removed", cid, indicator="BBANDS overlays", base=base, status="success")
            elif base in ("STOCH", "STOCHRSI"):
                # Atomically remove entire pane to avoid reapply races
                if self.middle_pane and getattr(self.middle_pane, "chart_widget", None):
                    try:
                        self.middle_pane.chart_widget.remove_indicator_pane(base, cid=cid)
                    except TypeError:
                        self.middle_pane.chart_widget.remove_indicator_pane(base)
                self._log_indicator("indicator_ui_removed", cid, indicator=f"{base} pane", base=base, status="success")
            elif base == "MACD":
                # Symmetric robust removal for MACD
                if self.middle_pane and getattr(self.middle_pane, "chart_widget", None):
                    try:
                        self.middle_pane.chart_widget.remove_indicator_pane("MACD", cid=cid)
                    except TypeError:
                        self.middle_pane.chart_widget.remove_indicator_pane("MACD")
                self._log_indicator("indicator_ui_removed", cid, indicator="MACD pane", base=base, status="success")
            else:
                self.middle_pane.remove_indicator(indicator_name, family=base, cid=cid)
                self._log_indicator("indicator_ui_removed", cid, indicator=indicator_name, base=base, status="success")

            # Robust DB cleanup: drop by family prefix if params missing
            try:
                pre_cols = set(stock_service.db.get_indicator_columns(self.current_model))
            except Exception:
                pre_cols = set()
            indicator_cols = stock_service.db.get_indicator_columns(self.current_model)
            params = self.applied_indicators.get(indicator_name, {})
            if base == "RSI":
                to_drop = [c for c in indicator_cols if c.startswith("rsi_")]
            elif base == "MACD":
                to_drop = [c for c in indicator_cols if c.startswith("macd_")]
            elif base == "STOCH":
                to_drop = [c for c in indicator_cols if c.startswith("stoch_") or c.startswith("stochastic_")]
            elif base == "STOCHRSI":
                to_drop = [c for c in indicator_cols if c.startswith("stochrsi_")]
            elif base == "BBANDS":
                to_drop = [c for c in indicator_cols if c.startswith("bbands_") or c.startswith("bollinger")]
            elif base == "OBV":
                # OBV column name is exactly 'obv' (no suffix)
                to_drop = ["obv"] if "obv" in indicator_cols else []
            else:
                exact = self._indicator_column_name(base, params) if params else None
                to_drop = [exact] if exact and exact in indicator_cols else [c for c in indicator_cols if c.startswith(base.lower() + "_")]

            self._log_indicator("indicator_columns_dropping", cid, to_drop=to_drop)
            for col in to_drop:
                stock_service.drop_indicator_column(self.current_model, col)

            if indicator_name in self.applied_indicators:
                del self.applied_indicators[indicator_name]

            # Refresh table headers
            try:
                self._refresh_table_only(limit=500)
                try:
                    post_cols = set(stock_service.db.get_indicator_columns(self.current_model))
                except Exception:
                    post_cols = set()
                removed = sorted(list(pre_cols - post_cols))
                if removed:
                    self._log_indicator("table_columns_removed", cid, removed=removed, now=sorted(list(post_cols)))
            except AttributeError:
                self._reload_display_data()

        except Exception as e:
            k2_logger.error(f"Failed to remove indicator {indicator_name}: {e}", "ANALYSIS")
    
    def _persist_indicator_series(self, col_name: str, ts_series: pd.Series, val_series: pd.Series) -> int:
        """Ensure and bulk-update indicator column values. Returns affected row count.
        Adds debug logging and normalizes timestamps to epoch-ms for reliable DB join.
        """
        stock_service.ensure_indicator_column(self.current_model, col_name, sql_type="NUMERIC")

        # Pre-persist debug
        try:
            k2_logger.info(str({
                'event': 'persist_debug_pre',
                'column': col_name,
                'len_ts': int(len(ts_series)),
                'len_vals': int(len(val_series)),
                'ts_head': [str(ts_series.iloc[i]) for i in range(min(3, len(ts_series)))],
            }), "INDICATOR")
        except Exception:
            pass

        # Normalize timestamps to epoch-ms int64
        try:
            if hasattr(ts_series, "dt"):
                ts_ms = (ts_series.view("int64") // 1_000_000).astype("int64")
            else:
                ts_numeric = pd.to_numeric(ts_series, errors="coerce")
                ts_ms = ts_numeric.astype("Int64").astype("int64")
        except Exception:
            ts_ms = ts_series.astype("int64")

        # Post-normalization debug
        try:
            k2_logger.info(str({
                'event': 'persist_debug_post',
                'column': col_name,
                'ts_ms_head': [int(ts_ms.iloc[i]) for i in range(min(3, len(ts_ms)))],
                'ts_ms_min_max': (int(ts_ms.min()), int(ts_ms.max()))
            }), "INDICATOR")
        except Exception:
            pass

        affected = stock_service.update_indicator_column(self.current_model, col_name, ts_ms, val_series)

        try:
            k2_logger.info(str({
                'event': 'persist_result',
                'column': col_name,
                'rows_updated': int(affected)
            }), "INDICATOR")
        except Exception:
            pass

        return affected

    def _new_cid(self) -> str:
        return f"ind-{uuid.uuid4().hex[:8]}"

    def _log_indicator(self, event: str, cid: str, **fields):
        try:
            base_ctx = {
                'tab_id': self.tab_id,
                'model': self.current_model,
                'event': event,
                'cid': cid
            }
            out = {**base_ctx, **fields}
            k2_logger.info(str(out), "INDICATOR")
        except Exception:
            pass
    
    def _indicator_column_name(self, name: str, params: Dict, output: str = None) -> str:
        """Consistent column naming: e.g., macd_fastperiod_12_slowperiod_26_signalperiod_9_hist"""
        name = name.lower().replace(" ", "")
        if params:
            parts = [f"{k}_{params[k]}" for k in sorted(params.keys())]
            base = "_".join([name] + parts)
        else:
            base = name
        return f"{base}_{output}" if output else base

    def _reload_display_data(self):
        """Reload rows from DB including indicator columns and refresh the middle pane."""
        try:
            rows, total = stock_service.get_display_data(self.current_model, limit=500)
            # DIAGNOSTIC for reload path as well
            try:
                table_name = self.current_model
                total_rows = len(rows)
                first_row_cols = len(rows[0]) if rows else 0
                k2_logger.debug(f"DISPLAY_FETCH(RELOAD): table={table_name} rows={total_rows} first_row_cols={first_row_cols}", "ANALYSIS")
                if rows:
                    k2_logger.debug(f"DISPLAY_FETCH(RELOAD): first_row_preview={rows[0]}", "ANALYSIS")
            except Exception:
                pass
            self.middle_pane.load_data(rows, self.current_metadata)
            self.model_label.setText(f"Model: {self.current_model} ({total:,} records)")
        except Exception as e:
            k2_logger.error(f"Display data reload failed: {e}", "ANALYSIS")

    # ADD: table-only refresh helper
    def _refresh_table_only(self, limit: int = 500):
        """Refresh only the data table with base + indicator columns; do not reload chart."""
        try:
            table_name = self.current_model
            if not table_name:
                return

            rows, columns, total = stock_service.get_display_data_with_indicators(table_name, limit=limit)
            k2_logger.debug(f"TABLE_ONLY_REFRESH: cols={columns}", "ANALYSIS")
            if rows:
                k2_logger.debug(f"TABLE_ONLY_REFRESH: first_row_cols={len(rows[0])}", "ANALYSIS")

            import pandas as pd
            df = pd.DataFrame(rows, columns=columns)
            # Persist table state so later UI events don't wipe indicator columns
            try:
                self.middle_pane.current_data = df
            except Exception:
                pass
            self.middle_pane.load_data_into_table(df)

            self.model_label.setText(f"Model: {table_name} ({total:,} records)")
            self.status_label.setText("Indicators updated")
        except Exception as e:
            k2_logger.error(f"Table-only refresh failed: {e}", "ANALYSIS")

    # NEW: after model load or param changes, update indicator gating
    def _refresh_indicator_availability(self):
        try:
            total = 0
            try:
                total = int((self.current_metadata or {}).get('total_records') or 0)
            except Exception:
                pass
            if total <= 0:
                return
            # Gate canonical indicators by required lookback
            from k2_quant.utilities.services.technical_analysis_service import ta_service
            for name, cfg in ta_service.indicators.items():
                req = ta_service.required_lookback(name, cfg.parameters)
                enabled = total >= req
                reason = None if enabled else f"Needs ≥ {req} data points; current: {total}"
                try:
                    self.left_pane.set_indicator_enabled(name, enabled, reason)
                except Exception:
                    pass
        except Exception as e:
            k2_logger.error(f"Indicator availability refresh failed: {e}", "ANALYSIS")
    
    def on_strategy_toggled(self, strategy_name: str, enabled: bool):
        """Handle strategy toggle from left pane"""
        if not self.current_model:
            k2_logger.warning("No model loaded for strategy toggle", "ANALYSIS")
            return
        
        k2_logger.info(f"Strategy '{strategy_name}' toggled to {enabled}", "ANALYSIS")
        
        try:
            if enabled:
                self.apply_strategy(strategy_name)
            else:
                self.remove_strategy(strategy_name)
        except Exception as e:
            k2_logger.error(f"Strategy toggle failed: {str(e)}", "ANALYSIS")
    
    def apply_strategy(self, strategy_name: str):
        """Apply strategy projections"""
        table_name = self.current_model
        
        # Fetch strategy code
        code = strategy_service.get_strategy_code(strategy_name)
        if not code:
            k2_logger.warning(f"Strategy code not found: {strategy_name}", "ANALYSIS")
            return
        
        # Load full dataset
        rows, _ = stock_service.get_display_data(table_name, limit=10**9)
        df = pd.DataFrame(rows, columns=['Date','Time','Open','High','Low','Close','Volume','VWAP'])
        
        # Convert to strategy format
        df['date_time_market'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume','VWAP':'vwap'})
        
        # Execute strategy
        result = dpe_service.execute_strategy(code, df)
        if not result.get('success'):
            k2_logger.error(f"Strategy execution failed: {result.get('error')}", "ANALYSIS")
            return
        
        result_df = result.get('data') if isinstance(result.get('data'), pd.DataFrame) else df
        
        # Get projection rows
        proj_df = result_df.iloc[len(df):].copy() if len(result_df) > len(df) else pd.DataFrame()
        
        if not proj_df.empty:
            # Insert projections
            stock_service.delete_projections(table_name, strategy_name)
            stock_service.insert_projections(table_name, proj_df, strategy_name)
            
            # Update state
            saved_models_manager.set_model_state(
                table_name,
                indicators=None,
                active_strategy=strategy_name,
                chart_range=None
            )
            
            # Reload view
            rows, total_count = stock_service.get_display_data(table_name, limit=500)
            self.middle_pane.load_data(rows, self.current_metadata)
            self.model_label.setText(f"Model: {table_name} ({total_count:,} records)")
    
    def remove_strategy(self, strategy_name: str):
        """Remove strategy projections"""
        table_name = self.current_model
        
        stock_service.delete_projections(table_name, strategy_name)
        saved_models_manager.set_model_state(
            table_name,
            indicators=None,
            active_strategy=None,
            chart_range=None
        )
        
        # Reload view
        rows, total_count = stock_service.get_display_data(table_name, limit=500)
        self.middle_pane.load_data(rows, self.current_metadata)
        self.model_label.setText(f"Model: {table_name} ({total_count:,} records)")
    
    # Middle pane event handlers
    def on_column_toggled(self, column: str, visible: bool):
        """Handle column visibility toggle"""
        k2_logger.info(f"Column '{column}' toggled to {'visible' if visible else 'hidden'}", "ANALYSIS")
    
    def on_view_mode_changed(self, mode: str):
        """Handle view mode change"""
        k2_logger.info(f"View mode changed to {mode}", "ANALYSIS")
    
    def on_projection_requested(self):
        """Handle projection request from middle pane"""
        k2_logger.info("Projection requested from middle pane", "ANALYSIS")
        # Could trigger AI to generate projections
    
    def on_indicator_applied_from_middle(self, indicator_type: str, params: dict):
        """Handle indicator application from middle pane"""
        k2_logger.info(f"Indicator {indicator_type} applied with params {params}", "ANALYSIS")
        self.apply_indicator(indicator_type, params)
    
    def on_data_exported(self, format: str, data: pd.DataFrame):
        """Handle data export"""
        k2_logger.info(f"Data exported in {format} format", "ANALYSIS")
    
    # Right pane event handlers
    def on_ai_message_sent(self, message: str):
        """Handle AI message"""
        k2_logger.info(f"AI message: {message}", "ANALYSIS")
        # In full implementation, would process with AI service
    
    def on_strategy_generated(self, name: str, code: str):
        """Handle strategy generation from AI"""
        k2_logger.info(f"Strategy generated: {name}", "ANALYSIS")
        # Could save strategy and apply it
    
    def on_projection_requested_from_ai(self, params: dict):
        """Handle [QUERY] code emitted by the AI widget and return results to chat."""
        k2_logger.info(f"Projection requested from AI: {params}", "ANALYSIS")

        if not isinstance(params, dict) or params.get('type') != 'QUERY':
            return

        code = params.get('code') or ""
        if not code:
            try:
                self.right_pane.chat.add_system_message("⚠️ No code provided for lookup.")
            except Exception:
                pass
            return

        if not self.current_model:
            try:
                self.right_pane.chat.add_system_message("⚠️ No model loaded.")
            except Exception:
                pass
            return

        # Build a DataFrame from current model (bounded for performance)
        try:
            rows, cols, _ = stock_service.get_display_data_with_indicators(self.current_model, limit=5000)
            df = pd.DataFrame(rows, columns=cols)
        except Exception as e:
            k2_logger.error(f"Failed to prepare DF for QUERY: {e}", "ANALYSIS")
            try:
                self.right_pane.chat.add_system_message("⚠️ Could not prepare data for lookup.")
            except Exception:
                pass
            return

        # Wrap snippet to capture a result via query(df)
        wrapper = f"""
{code}

try:
    result = query(df)
except Exception as _e:
    result = {{"error": str(_e)}}
""".strip()

        out = dpe_service.execute_strategy(wrapper, df)
        if not out.get('success'):
            err = out.get('error') or "Lookup failed."
            try:
                self.right_pane.chat.add_system_message(f"❌ QUERY error: {err}")
            except Exception:
                pass
            return

        result = out.get('data')
        # Format result back into chat
        try:
            if isinstance(result, pd.DataFrame):
                head = result.head(20)
                self.right_pane.chat.add_system_message(
                    f"✅ Lookup returned {len(result)} rows. Showing first {len(head)}:\n{head.to_string(index=False)}"
                )
            elif isinstance(result, dict):
                pretty = "\n".join([f"- {k}: {v}" for k, v in result.items()])
                self.right_pane.chat.add_system_message(f"✅ Lookup result:\n{pretty}")
            else:
                self.right_pane.chat.add_system_message(f"✅ Lookup result: {str(result)}")
        except Exception:
            try:
                self.right_pane.chat.add_system_message(f"✅ Lookup result: {str(result)}")
            except Exception:
                pass
    
    def load_saved_models(self):
        """Load saved models into the left pane"""
        self.left_pane.refresh_models()
    
    def setup_styling(self):
        """Apply consistent styling"""
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0a0a;
                color: #ffffff;
            }
            
            #analysisHeader {
                background-color: #0f0f0f;
                border-bottom: 1px solid #1a1a1a;
            }
            
            #analysisStatusBar {
                background-color: #0f0f0f;
                border-top: 1px solid #1a1a1a;
            }
        """)
    
    def cleanup(self):
        """Clean up resources when closing tab"""
        try:
            # Clear left pane states
            self.left_pane.clear_all_indicators()
            self.left_pane.clear_all_strategies()
            
            # Clear middle pane
            self.middle_pane.cleanup()
            
            # Clear right pane
            self.right_pane.cleanup()
            
            # Clear current data
            self.current_model = None
            self.current_data = None
            
            k2_logger.info(f"Analysis page cleaned up (Tab ID: {self.tab_id})", "ANALYSIS")
        except Exception as e:
            k2_logger.error(f"Error during cleanup: {str(e)}", "ANALYSIS")
    
    def reset_after_database_cleared(self):
        """Reset UI and state after DB deletion"""
        try:
            # Clear all panes
            self.left_pane.populate_models([])
            self.left_pane.clear_all_indicators()
            self.left_pane.clear_all_strategies()
            
            # Clear middle pane
            self.middle_pane.clear_data()
            
            # Reset labels
            self.model_label.setText("No model loaded")
            self.status_label.setText("Ready")
            
            # Clear AI chat
            self.right_pane.clear_chat()
            
            # Reset internal state
            self.current_model = None
            self.current_data = None
            
            k2_logger.info(f"Analysis tab {self.tab_id} reset after DB clear", "ANALYSIS")
        except Exception as e:
            k2_logger.error(f"Analysis reset failed: {str(e)}", "ANALYSIS")