"""
Stream Window Component — content widget for each MDI sub-window.

Layout (top-to-bottom):
  60% — Chart (pyqtgraph)
  40% — Tabbed: THINKSPACE | OUTPUTS | FORECAST DATA

Each window is fully isolated: its own chart, AI session, outputs, and forecast.
"""

import json
import re
from typing import Dict, Optional, Any, List

import pandas as pd
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.services.technical_analysis_service import ta_service
from k2_quant.utilities.services.stock_data_service import stock_service
from k2_quant.utilities.data.saved_models_manager import saved_models_manager
from k2_quant.utilities.services.strategy_service import strategy_service
from k2_quant.utilities.report_helpers import format_blocks_plain
from k2_quant.utilities.numeric_rounding import round_dataframe_numeric_columns
from k2_quant.utilities.services.dynamic_python_engine import dpe_service

from k2_quant.pages.analysis.widgets.chart import ChartWidget
from k2_quant.pages.analysis.widgets.data_tabs_widget import DataTabsWidget
from k2_quant.pages.analysis.components.right_pane import RightPaneWidget
from k2_quant.pages.analysis.components.outputs_panel import OutputsPanel


_INTRADAY_TIMESPANS = {'minute', 'min', 'hour'}


def _should_filter_market_hours(metadata: dict) -> bool:
    if metadata.get('market_hours_only', False):
        return True
    ts = str(metadata.get('timespan', '')).lower()
    return any(ts.startswith(prefix) for prefix in _INTRADAY_TIMESPANS)


class StreamWindowWidget(QWidget):
    """Self-contained window content for one model inside the Stream MDI area."""

    closed = pyqtSignal(str)  # table_name — emitted when window is closing
    focused = pyqtSignal(str)  # table_name — emitted on focus/activation

    def __init__(self, table_name: str, parent=None):
        super().__init__(parent)
        self.table_name = table_name
        self.current_data = None
        self.current_metadata: Dict[str, Any] = {}
        self.applied_indicators: Dict[str, Dict] = {}
        self.applied_strategies: set = set()

        self._init_ui()
        self._load_model()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setHandleWidth(2)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: #1a1a1a; }")

        # --- Chart (top 60%) ---
        self.chart_widget = ChartWidget()
        self.splitter.addWidget(self.chart_widget)

        # --- Bottom tabs (40%) ---
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setObjectName("streamBottomTabs")

        self.thinkspace = RightPaneWidget()
        self.thinkspace.setMinimumWidth(0)
        self.thinkspace.setMaximumWidth(16777215)
        self.thinkspace.workspace_provider = self._get_workspace_state
        self.thinkspace.save_chat_callback = self._save_chat
        self.thinkspace.load_chat_callback = self._load_chat

        self.outputs_panel = OutputsPanel()

        self.data_tabs = DataTabsWidget()
        # Show only the Forecast Data content — hide the internal sub-tab bar
        self.data_tabs.tab_widget.tabBar().setVisible(False)
        self.data_tabs.tab_widget.setCurrentIndex(1)

        self.bottom_tabs.addTab(self.thinkspace, "THINKSPACE")
        self.bottom_tabs.addTab(self.outputs_panel, "OUTPUTS")
        self.bottom_tabs.addTab(self.data_tabs, "FORECAST DATA")

        self.bottom_tabs.setStyleSheet("""
            #streamBottomTabs { background: #0a0a0a; border: none; }
            #streamBottomTabs::pane { border: none; background: #0a0a0a; }
            #streamBottomTabs QTabBar::tab {
                background: #1a1a1a; color: #999;
                padding: 6px 18px; border: none;
                border-bottom: 2px solid transparent;
                font-size: 11px; font-weight: 600; letter-spacing: 1px;
            }
            #streamBottomTabs QTabBar::tab:selected {
                color: #4a9eff; border-bottom: 2px solid #4a9eff;
                background: #0a0a0a;
            }
            #streamBottomTabs QTabBar::tab:hover:!selected {
                color: #ccc; background: #151515;
            }
        """)
        self.bottom_tabs.currentChanged.connect(self._on_bottom_tab_changed)

        self.splitter.addWidget(self.bottom_tabs)
        self.splitter.setStretchFactor(0, 6)
        self.splitter.setStretchFactor(1, 4)

        layout.addWidget(self.splitter)

        self._connect_signals()

    def _connect_signals(self):
        self.thinkspace.strategy_generated.connect(self._on_strategy_generated)
        self.thinkspace.strategy_removed_remotely.connect(self._on_strategy_removed)
        self.thinkspace.data_modified.connect(self._on_ai_data_modified)
        self.thinkspace.tab_writes_ready.connect(self._on_tab_writes)
        self.outputs_panel.reference_in_chat.connect(self._on_reference_run_in_chat)
        self.data_tabs.forecast_column_toggled.connect(self._on_forecast_column_toggled)

    # ── Model loading ─────────────────────────────────────────────

    def _load_model(self):
        try:
            from k2_quant.utilities.data.db_manager import db_manager as _db

            base_metadata = saved_models_manager.get_model_metadata(self.table_name) or {
                'symbol': self.table_name
            }
            mkt_hours = _should_filter_market_hours(base_metadata)
            rows, total_count = stock_service.get_display_data(
                self.table_name, limit=500, market_hours_only=mkt_hours)

            if not rows:
                k2_logger.warning(f"No data for model {self.table_name}", "STREAM")
                return

            self.current_data = rows
            parts = self.table_name.split('_')
            symbol = parts[1].upper() if len(parts) > 1 else 'UNKNOWN'
            has_row_number = _db._check_column_exists(self.table_name, '#')

            self.current_metadata = dict(base_metadata)
            self.current_metadata.update({
                'records': total_count,
                'table_name': self.table_name,
                'total_records': total_count,
                'symbol': symbol,
                'has_row_number': has_row_number,
                'market_hours_only': mkt_hours,
            })

            if total_count > 0:
                self.chart_widget.load_data_from_table(
                    table_name=self.table_name,
                    total_records=total_count,
                    metadata=self.current_metadata,
                )

            self._load_data_tabs(rows, total_count)
            self._restore_tab_data()

            ctx = {
                'symbol': symbol,
                'records': total_count,
                'table_name': self.table_name,
                'market_hours_only': mkt_hours,
            }
            self.thinkspace.set_data_context(ctx)

            self._restore_model_state()

        except Exception as e:
            k2_logger.error(f"Failed to load model in stream window: {e}", "STREAM")

    def _load_data_tabs(self, rows, total_count):
        """Feed data into the forecast DataTabsWidget, mirroring middle_pane logic."""
        has_rn = self.current_metadata.get('has_row_number', False)
        if has_rn:
            columns = ['#', 'Date', 'Time', 'Open', 'High', 'Low', 'Close',
                        'Volume', 'VWAP', 'Open_%', 'High_%', 'Low_%',
                        'Close_%', 'Elasticity', 'Close-Open_%']
        else:
            columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close',
                        'Volume', 'VWAP', 'Open_%', 'High_%', 'Low_%',
                        'Close_%', 'Elasticity', 'Close-Open_%']
        df = pd.DataFrame(rows, columns=columns[:len(rows[0])] if rows else columns[:8])

        self.data_tabs.set_model_context(self.table_name)

        if len(df) > 0:
            last_row = df.iloc[-1]
            last_row_number = None
            if '#' in df.columns:
                last_row_number = int(df['#'].iloc[-1])

            timespan_val = self.current_metadata.get('timespan', 'minute').lower()
            is_intraday = timespan_val.startswith('min') or timespan_val.startswith('hour')
            market_hours = self.current_metadata.get('market_hours_only', False)
            if is_intraday:
                market_hours = True

            self.data_tabs.setup_forecast(
                last_date=last_row.get('Date', last_row.iloc[0]),
                last_time=last_row.get('Time', last_row.iloc[1]),
                timespan=self.current_metadata.get('timespan', 'minute'),
                frequency=self.current_metadata.get('frequency', '1'),
                market_hours_only=market_hours,
                last_row_number=last_row_number,
            )

    def _restore_model_state(self):
        try:
            state = saved_models_manager.get_model_state(self.table_name)
            if state and isinstance(state, dict):
                agg = state.get('aggregation')
                if agg:
                    self.chart_widget.change_timeframe(agg)
                raw = state.get('active_strategy') or '[]'
                try:
                    names = set(json.loads(raw))
                except (json.JSONDecodeError, TypeError):
                    names = {raw} if raw else set()
                self.applied_strategies = names
        except Exception as e:
            k2_logger.debug(f"No model state available: {e}", "STREAM")

    # ── Indicator support ─────────────────────────────────────────

    def get_applied_indicators(self) -> set:
        return set(self.applied_indicators.keys())

    def get_applied_strategies(self) -> set:
        return set(self.applied_strategies)

    def apply_indicator(self, indicator_name: str, params: Dict):
        try:
            display_df = self._get_indicator_source_dataframe()
            if display_df is None or display_df.empty:
                return

            if 'Date' in display_df.columns and 'Time' in display_df.columns:
                dt_index = pd.to_datetime(
                    display_df['Date'].astype(str) + ' ' + display_df['Time'].astype(str),
                    errors='coerce')
            elif 'Date' in display_df.columns:
                dt_index = pd.to_datetime(display_df['Date'], errors='coerce')
            elif 'datetime' in display_df.columns:
                dt_index = pd.to_datetime(display_df['datetime'], errors='coerce')
            else:
                return

            if dt_index.isna().all():
                return

            df = display_df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume', 'VWAP': 'vwap',
            }).copy()
            df['datetime'] = dt_index
            df.set_index('datetime', inplace=True)

            base_name = indicator_name.split("(")[0].strip()
            ta_service_name = ta_service.map_display_name_to_service_name(base_name)
            if ta_service_name is None:
                ta_service_name = base_name.upper()

            mapped_params = self._map_indicator_params(ta_service_name, params)
            indicator_data = ta_service.calculate_indicator(df, ta_service_name, mapped_params)

            df_index = df.index
            if isinstance(df_index, pd.DatetimeIndex) and df_index.tz is not None:
                df_index = df_index.tz_localize(None)

            color = '#ffffff'

            if isinstance(indicator_data, dict):
                for line_name, line_series in indicator_data.items():
                    full_name = f"{indicator_name} ({line_name})"
                    if hasattr(line_series.index, 'tz') and line_series.index.tz is not None:
                        line_series.index = line_series.index.tz_localize(None)
                    line_series = line_series.reindex(df_index)
                    line_series.name = full_name
                    if line_series.isna().all():
                        continue
                    self.chart_widget.add_indicator(full_name, line_series, color)
                self.applied_indicators[indicator_name] = params
                return

            if indicator_data is None:
                return
            if isinstance(indicator_data, pd.Series) and (indicator_data.empty or indicator_data.isna().all()):
                return

            if isinstance(indicator_data, np.ndarray):
                indicator_data = pd.Series(indicator_data, index=df_index, name=indicator_name)
            elif isinstance(indicator_data, pd.Series):
                if hasattr(indicator_data.index, 'tz') and indicator_data.index.tz is not None:
                    indicator_data.index = indicator_data.index.tz_localize(None)
                indicator_data = indicator_data.reindex(df_index)
                indicator_data.name = indicator_name

            indicator_config = ta_service.get_indicator_info(ta_service_name)
            pane_type = 'main'
            if indicator_config and hasattr(indicator_config, 'pane'):
                pane_type = indicator_config.pane

            if pane_type == 'separate':
                self.chart_widget.add_indicator_pane(indicator_name, indicator_data, color)
            else:
                self.chart_widget.add_indicator(indicator_name, indicator_data, color)

            self.applied_indicators[indicator_name] = params

        except Exception as e:
            k2_logger.error(f"Stream window indicator error: {e}", "STREAM")

    def remove_indicator(self, indicator_name: str):
        try:
            self.chart_widget.remove_indicator(indicator_name)
            for sub in [f"{indicator_name} (upper)", f"{indicator_name} (middle)",
                        f"{indicator_name} (lower)"]:
                try:
                    self.chart_widget.remove_indicator(sub)
                except Exception:
                    pass
            self.applied_indicators.pop(indicator_name, None)
        except Exception as e:
            k2_logger.error(f"Failed to remove indicator: {e}", "STREAM")

    def _get_indicator_source_dataframe(self) -> Optional[pd.DataFrame]:
        df = getattr(self.chart_widget, "data", None)
        if isinstance(df, pd.DataFrame) and not df.empty:
            cols = [c for c in ['Date', 'Time', 'Open', 'High', 'Low', 'Close',
                                'Volume', 'VWAP', 'datetime'] if c in df.columns]
            return df[cols].copy() if cols else df.copy()
        return None

    def _map_indicator_params(self, ta_service_name: str, params: Dict) -> Dict:
        mapped = params.copy()
        if 'period' in mapped:
            mapped['timeperiod'] = mapped.pop('period')
        if ta_service_name == 'BBANDS' and 'std' in mapped:
            std_value = mapped.pop('std')
            mapped['nbdevup'] = std_value
            mapped['nbdevdn'] = std_value
        if ta_service_name == 'MACD':
            if 'fast' in mapped:
                mapped['fastperiod'] = mapped.pop('fast')
            if 'slow' in mapped:
                mapped['slowperiod'] = mapped.pop('slow')
            if 'signal' in mapped:
                mapped['signalperiod'] = mapped.pop('signal')
        if ta_service_name == 'STOCH':
            if 'k_period' in mapped:
                mapped['slowk_period'] = mapped.pop('k_period')
            if 'd_period' in mapped:
                mapped['slowd_period'] = mapped.pop('d_period')
        return mapped

    @staticmethod
    def extract_default_indicator_params(indicator_name: str) -> Dict:
        params: Dict[str, Any] = {}
        if "(" in indicator_name and ")" in indicator_name:
            match = re.search(r'\((\d+)\)', indicator_name)
            if match:
                params['period'] = int(match.group(1))

        base_name = indicator_name.split("(")[0].strip().upper()
        if base_name == "RSI" and 'period' not in params:
            params['period'] = 14
        elif base_name == "MACD":
            params = {'fast': 12, 'slow': 26, 'signal': 9}
        elif base_name == "BOLLINGER BANDS":
            params = {'period': 20, 'std': 2}
        elif base_name == "STOCHASTIC":
            params = {'k_period': 14, 'd_period': 3}
        elif base_name in ("OBV", "VWAP", "VOLUME"):
            params = {}
        elif base_name in ("SMA", "EMA") and 'period' not in params:
            params['period'] = 20
        return params

    # ── Strategy support ──────────────────────────────────────────

    def apply_strategy(self, strategy_name: str):
        table_name = self.table_name
        code = strategy_service.get_strategy_code(strategy_name)
        if not code:
            k2_logger.warning(f"Strategy code not found: {strategy_name}", "STREAM")
            return

        mkt_hours = self.current_metadata.get('market_hours_only', False)
        rows, _ = stock_service.get_display_data(
            table_name, limit=10**9, market_hours_only=mkt_hours)
        has_rn = self.current_metadata.get('has_row_number', False)
        if has_rn:
            all_columns = ['#', 'Date', 'Time', 'Open', 'High', 'Low', 'Close',
                           'Volume', 'VWAP', 'Open_%', 'High_%', 'Low_%',
                           'Close_%', 'Elasticity', 'Close-Open_%']
        else:
            all_columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close',
                           'Volume', 'VWAP', 'Open_%', 'High_%', 'Low_%',
                           'Close_%', 'Elasticity', 'Close-Open_%']
        df = pd.DataFrame(rows, columns=all_columns[:len(rows[0])] if rows else all_columns[:8])

        df['date_time_market'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Volume': 'volume', 'VWAP': 'vwap',
            'Open_%': 'open_pct', 'High_%': 'high_pct',
            'Low_%': 'low_pct', 'Close_%': 'close_pct',
            'Elasticity': 'elasticity', 'Close-Open_%': 'close_open_pct',
        })
        if 'open_pct' not in df.columns and all(
                c in df.columns for c in ('open', 'high', 'low', 'close')):
            df = df.sort_values('date_time_market', kind='mergesort').reset_index(drop=True)
            for c in ('open', 'high', 'low', 'close'):
                prev = df[c].astype(float).shift(1)
                cur = df[c].astype(float)
                df[f'{c}_pct'] = np.where(
                    (prev != 0) & prev.notna() & cur.notna(),
                    (cur - prev) / prev * 100.0, np.nan)
            lo = df['low'].astype(float)
            o_ = df['open'].astype(float)
            cl = df['close'].astype(float)
            hi = df['high'].astype(float)
            df['elasticity'] = np.where(lo != 0, (hi - lo) / lo * 100.0, np.nan)
            df['close_open_pct'] = np.where(o_ != 0, (cl - o_) / o_ * 100.0, np.nan)

        df = round_dataframe_numeric_columns(df)
        result = dpe_service.execute_strategy(code, df)
        strategy_service.save_run(
            strategy_name=strategy_name, code_snapshot=code,
            result=result, model_table=table_name)
        self.outputs_panel.refresh()
        self.applied_strategies.add(strategy_name)

        if not result.get('success'):
            k2_logger.error(f"Strategy execution failed: {result.get('error')}", "STREAM")
            return

        tab_writes = result.get('_tab_writes', [])
        forecast_writes = [w for w in tab_writes if w.get("type") == "forecast"]
        working_writes = [w for w in tab_writes if w.get("type") == "working"]

        if working_writes:
            self._on_tab_writes(working_writes)
        if forecast_writes:
            for w in forecast_writes:
                w['_strategy'] = strategy_name
            self._on_tab_writes(forecast_writes)
            return

        result_df = result.get('data') if isinstance(result.get('data'), pd.DataFrame) else df
        proj_df = result_df.iloc[len(df):].copy() if len(result_df) > len(df) else pd.DataFrame()
        if not proj_df.empty:
            stock_service.delete_projections(table_name, strategy_name)
            stock_service.insert_projections(table_name, proj_df, strategy_name)
            mkt_hours = self.current_metadata.get('market_hours_only', False)
            rows, total = stock_service.get_display_data(
                table_name, limit=500, market_hours_only=mkt_hours)
            self.current_metadata['total_records'] = total
            self.chart_widget.load_data_from_table(
                table_name=table_name,
                total_records=total,
                metadata=self.current_metadata,
            )

    def remove_strategy(self, strategy_name: str):
        try:
            stock_service.delete_projections(self.table_name, strategy_name)
        except Exception:
            pass

        col_names = list(self.data_tabs._strategy_columns.get(strategy_name, []))
        self.data_tabs.clear_strategy_columns(strategy_name)
        for col_name in col_names:
            self.chart_widget.remove_forecast_line(col_name)

        self.applied_strategies.discard(strategy_name)

    # ── Callback / signal handlers ────────────────────────────────

    def _on_bottom_tab_changed(self, index: int):
        if index == 1:
            self.outputs_panel.refresh()

    def _on_strategy_generated(self, name: str, code: str):
        k2_logger.info(f"Strategy generated in stream window: {name}", "STREAM")
        self.outputs_panel.refresh()

    def _on_strategy_removed(self, name: str):
        name = (name or "").strip()
        if not name:
            return
        self.applied_strategies.discard(name)
        self.remove_strategy(name)
        self.outputs_panel.handle_strategy_deleted(name)

    def _on_ai_data_modified(self):
        if not self.table_name:
            return
        try:
            rows, total = stock_service.get_display_data(self.table_name, limit=500)
            if rows:
                self.current_data = rows
                self.current_metadata['records'] = total
                self.current_metadata['total_records'] = total
                self.chart_widget.load_data_from_table(
                    table_name=self.table_name,
                    total_records=total,
                    metadata=self.current_metadata,
                )
        except Exception as e:
            k2_logger.error(f"Stream window data refresh failed: {e}", "STREAM")

    def _on_tab_writes(self, writes: list):
        if not writes:
            return
        for w in writes:
            try:
                wtype = w.get("type")
                if wtype == "working":
                    scope = w.get("scope", "model")
                    col = w.get("column_name", "result")
                    vals = w.get("values", [])
                    grid_col = w.get("column")
                    sheet = w.get("sheet")
                    self.data_tabs.add_working_column(scope, col, vals,
                                                      column=grid_col, sheet=sheet)
                elif wtype == "delete_working":
                    scope = w.get("scope", "model")
                    col = w.get("column_name", "")
                    sheet = w.get("sheet")
                    if col:
                        self.data_tabs.delete_working_column(scope, col, sheet=sheet)
                elif wtype == "forecast":
                    if "column_name" in w:
                        col_name = w["column_name"]
                        vals = w.get("values", [])
                        anchor = w.get("anchor_price")
                        strategy = w.get("_strategy", "AI")
                        if vals:
                            self.data_tabs.set_forecast_column(strategy, col_name, vals,
                                                               anchor_price=anchor)
                    else:
                        strategy = w.get("_strategy", "AI")
                        set_idx = w.get("set_index", 1)
                        for ohlc, key in [("Open", "open_values"), ("High", "high_values"),
                                          ("Low", "low_values"), ("Close", "close_values")]:
                            vals = w.get(key)
                            if vals:
                                col_name = f"{ohlc}_P{set_idx}"
                                self.data_tabs.set_forecast_column(strategy, col_name, vals)
            except Exception as e:
                k2_logger.error(f"Stream tab write failed: {e}", "STREAM")

    def _on_reference_run_in_chat(self, run_id: int, strategy_name: str):
        run = strategy_service.get_run(run_id)
        if not run:
            return
        summary = (
            f"[Strategy Run Report]\n"
            f"Strategy: {run['strategy_name']}\n"
            f"Model: {run.get('model_table', '--')}\n"
            f"Time: {run['run_timestamp']}\n"
            f"Status: {'SUCCESS' if run.get('success') else 'FAILED'}\n"
        )
        rb = run.get("report_blocks_json")
        if rb:
            try:
                blocks = json.loads(rb)
                if isinstance(blocks, list) and blocks:
                    plain = format_blocks_plain(blocks)
                    if plain:
                        summary += f"\nReport:\n{plain}\n"
            except (json.JSONDecodeError, TypeError):
                pass
        stdout = (run.get('stdout_output') or '').strip()
        if stdout:
            summary += f"\nLog (stdout):\n{stdout}\n"
        error = (run.get('error_output') or '').strip()
        if error:
            summary += f"\nError:\n{error}\n"
        code_snap = (run.get('code_snapshot') or '').strip()
        if code_snap:
            summary += f"\nCode:\n{code_snap}\n"
        self.thinkspace.conversation_history.append({'role': 'user', 'content': summary})
        self.bottom_tabs.setCurrentIndex(0)

    def _on_forecast_column_toggled(self, column_name: str, visible: bool):
        if visible:
            values = self.data_tabs.get_forecast_column_data(column_name)
            anchor = self.data_tabs.get_forecast_column_anchor(column_name)
            if values:
                self.chart_widget.add_forecast_line(column_name, values,
                                                     anchor_price=anchor)
        else:
            self.chart_widget.remove_forecast_line(column_name)

    def _get_workspace_state(self) -> Optional[Dict]:
        result = {}
        for name in self.data_tabs.get_sheet_names():
            ws_df = self.data_tabs.get_working_data('model', sheet=name)
            if ws_df is not None and not ws_df.empty:
                result[name] = ws_df
        return {'sheets': result} if result else None

    def _save_chat(self, table_name: str, html: str, history: list):
        saved_models_manager.save_chat_history(table_name, html, history)

    def _load_chat(self, table_name: str) -> Optional[Dict]:
        return saved_models_manager.get_chat_history(table_name)

    # ── Persistence helpers ───────────────────────────────────────

    def _persist_tab_data(self):
        """Save forecast and working sheet data to DB."""
        try:
            from k2_quant.utilities.data.db_manager import db_manager
            fc = self.data_tabs.serialise_forecast()
            if fc:
                db_manager.save_tab_data(f"forecast:{self.table_name}", fc)
            else:
                db_manager.delete_tab_data(f"forecast:{self.table_name}")

            ws = self.data_tabs.serialise_sheets()
            if ws:
                db_manager.save_tab_data(f"workspace:{self.table_name}", ws)
            else:
                db_manager.delete_tab_data(f"workspace:{self.table_name}")
        except Exception as e:
            k2_logger.error(f"Stream tab data persist failed: {e}", "STREAM")

    def _restore_tab_data(self):
        """Restore forecast and working sheet data from DB."""
        try:
            from k2_quant.utilities.data.db_manager import db_manager
            fc = db_manager.load_tab_data(f"forecast:{self.table_name}")
            if fc:
                self.data_tabs.restore_forecast(fc)

            ws = db_manager.load_tab_data(f"workspace:{self.table_name}")
            self.data_tabs.restore_sheets(ws)
        except Exception as e:
            k2_logger.error(f"Stream tab data restore failed: {e}", "STREAM")

    def persist_state(self):
        """Save current state to DB before closing."""
        try:
            self._persist_tab_data()
            self._save_chat(
                self.table_name,
                json.dumps(self.thinkspace._message_records),
                list(self.thinkspace.conversation_history),
            )
            saved_models_manager.set_model_state(
                self.table_name,
                indicators=None,
                active_strategy=json.dumps(sorted(self.applied_strategies)),
                chart_range=None,
            )
        except Exception as e:
            k2_logger.error(f"Stream window state persistence failed: {e}", "STREAM")

    def cleanup(self):
        """Release resources."""
        try:
            self.persist_state()
            self.thinkspace.cleanup()
            if hasattr(self.chart_widget, 'cleanup'):
                self.chart_widget.cleanup()
        except Exception as e:
            k2_logger.error(f"Stream window cleanup error: {e}", "STREAM")

    def apply_styling(self):
        self.setObjectName("streamWindowContent")
        self.setStyleSheet("""
            #streamWindowContent {
                background-color: #0a0a0a;
                color: #ffffff;
            }
        """)
