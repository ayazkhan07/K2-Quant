"""
Stream Window Component — content widget for each MDI sub-window.

Layout (top-to-bottom):
  60% — Chart (pyqtgraph)
  40% — Tabbed: THINKSPACE | OUTPUTS | FORECAST DATA | ACTUAL DATA

Each window is fully isolated: its own chart, AI session, outputs,
forecast, and live-streaming actual data pipeline.
"""

import json
import re
from datetime import datetime, date as dt_date, time as dt_time
from typing import Dict, Optional, Any, List

import pandas as pd
import numpy as np
import pyqtgraph as pg

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter, QTabWidget, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
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
from k2_quant.utilities.services.strategy_runner import strategy_runner

from k2_quant.pages.analysis.widgets.chart import ChartWidget
from k2_quant.pages.analysis.widgets.data_tabs_widget import DataTabsWidget
from k2_quant.pages.analysis.components.right_pane import RightPaneWidget
from k2_quant.pages.analysis.components.outputs_panel import OutputsPanel
from k2_quant.pages.stream.widgets.actual_data_widget import ActualDataWidget

from k2_quant.utilities.helpers.market_hours import is_market_open, market_status
from k2_quant.utilities.services.polygon_websocket import polygon_ws_manager
from k2_quant.utilities.services.bar_aggregator import BarAggregator
from k2_quant.utilities.services.stream_reconciler import StreamReconciler
from k2_quant.utilities.data.actual_data_manager import actual_data_manager


_INTRADAY_TIMESPANS = {'minute', 'min', 'hour'}


class _PictureItem(pg.GraphicsObject):
    """Lightweight wrapper that renders a pre-painted QPicture in a pyqtgraph scene."""

    def __init__(self, picture):
        super().__init__()
        self._picture = picture
        self._bounding = None

    def paint(self, painter, *_args):
        painter.drawPicture(0, 0, self._picture)

    def boundingRect(self):
        if self._bounding is None:
            from PyQt6.QtCore import QRectF
            self._bounding = QRectF(self._picture.boundingRect())
        return self._bounding


class _NumericTableItem(QTableWidgetItem):
    """QTableWidgetItem that sorts by its stored numeric value."""

    def __init__(self, text: str, sort_value=None):
        super().__init__(text)
        self._sort_value = sort_value

    def __lt__(self, other):
        if isinstance(other, _NumericTableItem):
            if self._sort_value is not None and other._sort_value is not None:
                return self._sort_value < other._sort_value
        return super().__lt__(other)


def _should_filter_market_hours(metadata: dict) -> bool:
    if metadata.get('market_hours_only', False):
        return True
    ts = str(metadata.get('timespan', '')).lower().lstrip('0123456789 ')
    return any(ts.startswith(prefix) for prefix in _INTRADAY_TIMESPANS)


class StreamWindowWidget(QWidget):
    """Self-contained window content for one model inside the Stream MDI area."""

    closed = pyqtSignal(str)
    focused = pyqtSignal(str)
    stream_status_changed = pyqtSignal(str)
    stream_rejected = pyqtSignal()  # title bar should reset its toggle

    def __init__(self, table_name: str, parent=None):
        super().__init__(parent)
        self.table_name = table_name
        self.current_data = None
        self.current_metadata: Dict[str, Any] = {}
        self.applied_indicators: Dict[str, Dict] = {}
        self.applied_strategies: set = set()

        self._is_streaming = False
        self._aggregator: Optional[BarAggregator] = None
        self._reconciler: Optional[StreamReconciler] = None
        self._actual_lines: Dict[str, Any] = {}
        self._last_forming_price: Optional[float] = None

        # Strategy-runner wiring: map job_key -> strategy_name so signal
        # handlers can route back to the right post-processing step. Only
        # signals whose key matches this window's table_name are acted on.
        self._strategy_jobs: Dict[str, str] = {}
        strategy_runner.started.connect(self._on_strategy_started)
        strategy_runner.progress.connect(self._on_strategy_progress)
        strategy_runner.finished.connect(self._on_strategy_finished)
        strategy_runner.failed.connect(self._on_strategy_failed)
        strategy_runner.cancelled.connect(self._on_strategy_cancelled)

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

        # MODEL DATA — standalone table, no reparenting
        self.model_data_table = QTableWidget()
        self.model_data_table.setAlternatingRowColors(True)
        self.model_data_table.horizontalHeader().setStretchLastSection(False)
        self.model_data_table.setSortingEnabled(True)
        self.model_data_table.verticalHeader().setVisible(False)
        self.model_data_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.model_data_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.model_data_table.setSelectionMode(QTableWidget.SelectionMode.ContiguousSelection)
        self.model_data_table.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.model_data_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.model_data_table.setStyleSheet("""
            QTableWidget {
                background-color: #0a0a0a; alternate-background-color: #0f0f0f;
                color: #c8c8c8; gridline-color: #1a1a1a; border: none;
                font-size: 11px; font-family: 'Consolas', 'Courier New', monospace;
            }
            QTableWidget::item { padding: 2px 6px; }
            QTableWidget::item:selected { background-color: #1a2a3a; color: #fff; }
            QHeaderView::section {
                background-color: #111; color: #aaa; border: none;
                border-bottom: 1px solid #222; padding: 4px 6px;
                font-size: 11px; font-weight: 600;
            }
        """)

        # FORECAST DATA — DataTabsWidget with inner bar hidden, forced to forecast tab
        self.data_tabs = DataTabsWidget()
        self.data_tabs.tab_widget.tabBar().setVisible(False)
        self.data_tabs.tab_widget.setCurrentIndex(1)

        self.actual_data_widget = ActualDataWidget()

        self.bottom_tabs.addTab(self.thinkspace, "THINKSPACE")
        self.bottom_tabs.addTab(self.outputs_panel, "OUTPUTS")
        self.bottom_tabs.addTab(self.model_data_table, "MODEL DATA")
        self.bottom_tabs.addTab(self.data_tabs, "FORECAST DATA")
        self.bottom_tabs.addTab(self.actual_data_widget, "ACTUAL DATA")

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
        try:
            vb = self.chart_widget.main_plot.getViewBox()
            vb.sigRangeChanged.connect(self._reposition_price_label)
        except Exception:
            pass

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
        """Feed data into Model Data + Forecast tabs."""
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

        self._populate_model_data_table(df)

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

    def _populate_model_data_table(self, df: pd.DataFrame):
        """Fill the standalone MODEL DATA table with the model's OHLCV rows."""
        if df is None or df.empty:
            return
        table = self.model_data_table
        table.setSortingEnabled(False)
        table.setRowCount(len(df))
        table.setColumnCount(len(df.columns))

        headers = []
        for col in df.columns:
            cs = str(col)
            headers.append(
                cs.capitalize() if cs.lower() in (
                    'open', 'high', 'low', 'close', 'volume', 'vwap', 'date', 'time')
                else cs)
        table.setHorizontalHeaderLabels(headers)

        for r in range(len(df)):
            for c in range(len(df.columns)):
                value = df.iloc[r, c]
                col_lower = str(df.columns[c]).lower()
                if pd.isna(value):
                    text = ""
                elif col_lower in ('date',):
                    text = str(value)
                elif col_lower in ('time',):
                    text = str(value)[:8]
                elif col_lower == '#':
                    text = str(int(value)) if value is not None else ""
                elif col_lower == 'volume':
                    try:
                        text = f"{int(value):,}"
                    except (ValueError, TypeError):
                        text = str(value)
                else:
                    try:
                        text = f"{float(value):.2f}"
                    except (ValueError, TypeError):
                        text = str(value)

                if col_lower not in ('date', 'time'):
                    try:
                        sort_val = float(value)
                    except (ValueError, TypeError):
                        sort_val = None
                    item = _NumericTableItem(text, sort_val)
                else:
                    item = QTableWidgetItem(text)

                if col_lower in ('#', 'date', 'time'):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(r, c, item)

        table.resizeColumnsToContents()
        for i in range(table.columnCount()):
            if table.columnWidth(i) < 80:
                table.setColumnWidth(i, 80)
        table.setSortingEnabled(True)

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

    # ── Strategy apply: async dispatch to StrategyRunner ──────────
    #
    # The data load, DataFrame build, and DPE exec all run on a worker
    # thread so the Qt event loop stays responsive on large minute-bar
    # models (1M+ rows). See k2_quant.utilities.services.strategy_runner.
    # Post-processing (save_run, forecast routing, chart reload) runs on
    # the GUI thread in ``_on_strategy_finished``.

    def apply_strategy(self, strategy_name: str):
        table_name = self.table_name
        code = strategy_service.get_strategy_code(strategy_name)
        if not code:
            k2_logger.warning(f"Strategy code not found: {strategy_name}", "STREAM")
            return

        mkt_hours = bool(self.current_metadata.get('market_hours_only', False))
        has_rn = bool(self.current_metadata.get('has_row_number', False))

        def _loader():
            return self._build_strategy_frame(table_name, mkt_hours, has_rn)

        key = strategy_runner.submit(
            table_name=table_name,
            strategy_name=strategy_name,
            code=code,
            loader=_loader,
        )
        if key is None:
            return
        self._strategy_jobs[key] = strategy_name

    @staticmethod
    def _build_strategy_frame(table_name: str, mkt_hours: bool,
                              has_rn: bool) -> pd.DataFrame:
        """Runs on the strategy worker thread - no Qt calls.

        Logs per-step wall-clock so the terminal shows where the 15-20s
        pre-strategy cost goes on large minute-bar tables.
        """
        import time as _time
        t0 = _time.time()
        rows, _total = stock_service.get_display_data(
            table_name, limit=10**9, market_hours_only=mkt_hours)
        t_fetch = (_time.time() - t0) * 1000.0
        k2_logger.info(
            f"[loader] db_fetch took={t_fetch:,.0f} ms rows={len(rows):,} "
            f"table={table_name} mkt_hours={mkt_hours}",
            "STREAM",
        )

        if has_rn:
            all_columns = ['#', 'Date', 'Time', 'Open', 'High', 'Low', 'Close',
                           'Volume', 'VWAP', 'Open_%', 'High_%', 'Low_%',
                           'Close_%', 'Elasticity', 'Close-Open_%']
        else:
            all_columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close',
                           'Volume', 'VWAP', 'Open_%', 'High_%', 'Low_%',
                           'Close_%', 'Elasticity', 'Close-Open_%']

        t1 = _time.time()
        df = pd.DataFrame(rows, columns=all_columns[:len(rows[0])] if rows else all_columns[:8])
        t_build = (_time.time() - t1) * 1000.0
        k2_logger.info(
            f"[loader] df_build took={t_build:,.0f} ms shape={df.shape}",
            "STREAM",
        )

        t2 = _time.time()
        df['date_time_market'] = pd.to_datetime(
            df['Date'].astype(str) + ' ' + df['Time'].astype(str))
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'Volume': 'volume', 'VWAP': 'vwap',
            'Open_%': 'open_pct', 'High_%': 'high_pct',
            'Low_%': 'low_pct', 'Close_%': 'close_pct',
            'Elasticity': 'elasticity', 'Close-Open_%': 'close_open_pct',
        })
        recomputed = False
        if 'open_pct' not in df.columns and all(
                c in df.columns for c in ('open', 'high', 'low', 'close')):
            recomputed = True
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
        t_derive = (_time.time() - t2) * 1000.0
        k2_logger.info(
            f"[loader] derive_cols took={t_derive:,.0f} ms "
            f"recomputed_pct={recomputed}",
            "STREAM",
        )

        t3 = _time.time()
        out = round_dataframe_numeric_columns(df)
        t_round = (_time.time() - t3) * 1000.0
        k2_logger.info(
            f"[loader] round_numeric took={t_round:,.0f} ms",
            "STREAM",
        )
        k2_logger.info(
            f"[loader] total={(t_fetch + t_build + t_derive + t_round):,.0f} ms",
            "STREAM",
        )
        return out

    # ── Strategy runner signal handlers (GUI thread) ──────────────
    def _own_key(self, key: str) -> bool:
        """True if ``key`` belongs to a job this window submitted."""
        return key in self._strategy_jobs

    def _on_strategy_started(self, key: str):
        if not self._own_key(key):
            return
        name = self._strategy_jobs.get(key, '?')
        k2_logger.info(f"Strategy started: {name}", "STREAM")
        self.stream_status_changed.emit(f"Running strategy: {name}")

    def _on_strategy_progress(self, key: str, message: str):
        if not self._own_key(key):
            return
        name = self._strategy_jobs.get(key, '?')
        self.stream_status_changed.emit(f"{name}: {message}")

    def _on_strategy_failed(self, key: str, error: str):
        if not self._own_key(key):
            return
        name = self._strategy_jobs.pop(key, '?')
        k2_logger.error(f"Strategy '{name}' failed: {error}", "STREAM")
        self.stream_status_changed.emit(f"Strategy '{name}' failed")
        # Mark as applied so the checkbox state matches left-pane; user
        # can un-check to clear and retry.
        self.applied_strategies.add(name)

    def _on_strategy_cancelled(self, key: str):
        if not self._own_key(key):
            return
        name = self._strategy_jobs.pop(key, '?')
        k2_logger.info(f"Strategy '{name}' cancelled", "STREAM")
        self.stream_status_changed.emit(f"Strategy '{name}' cancelled")
        self.applied_strategies.discard(name)

    def _on_strategy_finished(self, key: str, result: object):
        if not self._own_key(key):
            return
        strategy_name = self._strategy_jobs.pop(key, None)
        if strategy_name is None:
            return
        if not isinstance(result, dict):
            k2_logger.error(
                f"Strategy '{strategy_name}' finished with non-dict result", "STREAM")
            return

        table_name = self.table_name
        code = strategy_service.get_strategy_code(strategy_name) or ''
        strategy_service.save_run(
            strategy_name=strategy_name, code_snapshot=code,
            result=result, model_table=table_name)
        self.outputs_panel.refresh()
        self.applied_strategies.add(strategy_name)

        if not result.get('success'):
            k2_logger.error(
                f"Strategy execution failed: {result.get('error')}", "STREAM")
            self.stream_status_changed.emit(f"Strategy '{strategy_name}' failed")
            return

        self.stream_status_changed.emit(f"Strategy '{strategy_name}' complete")

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

        result_df = result.get('data')
        if not isinstance(result_df, pd.DataFrame):
            return
        # The worker produced ``result_df``. We no longer have the input
        # frame in scope, but ``save_run`` already persisted metrics.
        # ``metrics.original_rows`` tells us where projections begin.
        metrics = result.get('metrics') or {}
        original_rows = int(metrics.get('original_rows') or 0)
        proj_df = (result_df.iloc[original_rows:].copy()
                   if original_rows and len(result_df) > original_rows
                   else pd.DataFrame())
        if not proj_df.empty:
            stock_service.delete_projections(table_name, strategy_name)
            stock_service.insert_projections(table_name, proj_df, strategy_name)
            mkt_hours = self.current_metadata.get('market_hours_only', False)
            _, total = stock_service.get_display_data(
                table_name, limit=500, market_hours_only=mkt_hours)
            self.current_metadata['total_records'] = total
            self.chart_widget.load_data_from_table(
                table_name=table_name,
                total_records=total,
                metadata=self.current_metadata,
            )

    def remove_strategy(self, strategy_name: str):
        # If this strategy is still executing, request cancel; the
        # ``cancelled`` signal will unwind applied_strategies.
        key = strategy_runner.make_key(self.table_name, strategy_name)
        if strategy_runner.is_running(key):
            strategy_runner.cancel(key)
            # Fall through to clear any partial projections/columns too.

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

    # ── Live streaming pipeline ──────────────────────────────────

    def toggle_streaming(self, start: bool):
        """Called by the title-bar toggle button."""
        if start:
            self._start_streaming()
        else:
            self._stop_streaming()

    def _start_streaming(self):
        mkt_hours = self.current_metadata.get('market_hours_only', False)
        if mkt_hours and not is_market_open():
            status = market_status()
            QMessageBox.information(
                self, "Market Closed",
                f"{status}\n\nLive streaming is only available during market hours.",
            )
            self.stream_status_changed.emit(status)
            self._is_streaming = False
            self.stream_rejected.emit()
            return

        symbol = self.current_metadata.get('symbol', '').upper()
        if not symbol or symbol == 'UNKNOWN':
            self.stream_status_changed.emit("No symbol")
            self.stream_rejected.emit()
            return

        raw_timespan = self.current_metadata.get('timespan', 'minute').lower().strip()
        freq_str = self.current_metadata.get('frequency', '1')
        frequency = int(''.join(c for c in str(freq_str) if c.isdigit()) or '1')

        ts_match = re.match(r'^(\d+)\s*(min|minute|hour|day|week|month)', raw_timespan)
        if ts_match:
            embedded_freq = int(ts_match.group(1))
            timespan = ts_match.group(2)
            if frequency <= 1:
                frequency = embedded_freq
        else:
            timespan = raw_timespan

        k2_logger.info(
            f"[STREAM START] {symbol} | table={self.table_name} | "
            f"raw_timespan={raw_timespan} | timespan={timespan} | "
            f"raw_freq={freq_str} | frequency={frequency}",
            "STREAM",
        )

        actual_data_manager.ensure_table(self.table_name)

        total_records = self.current_metadata.get('total_records', 0)
        self.actual_data_widget.set_base_index(total_records)

        from k2_quant.utilities.data.db_manager import db_manager as _db
        model_last_bar = _db.get_last_bar(self.table_name)
        k2_logger.info(
            f"[STREAM START] Model last bar: {model_last_bar}",
            "STREAM",
        )

        if model_last_bar:
            pruned = actual_data_manager.prune_up_to(
                self.table_name, model_last_bar[0], model_last_bar[1]
            )
            if pruned:
                k2_logger.info(
                    f"[STREAM START] Pruned {pruned} stale actual bars",
                    "STREAM",
                )

        mkt_hours = self.current_metadata.get('market_hours_only', False)
        if mkt_hours:
            purged_mh = actual_data_manager.purge_outside_market_hours(
                self.table_name
            )
            if purged_mh:
                k2_logger.info(
                    f"[STREAM START] Purged {purged_mh} after-hours actual bars",
                    "STREAM",
                )

        if timespan.startswith("min"):
            window_min = frequency
        elif timespan.startswith("hour"):
            window_min = frequency * 60
        else:
            window_min = 0
        if window_min > 1:
            purged = actual_data_manager.purge_misaligned(
                self.table_name, window_min
            )
            if purged:
                k2_logger.info(
                    f"[STREAM START] Purged {purged} misaligned actual bars "
                    f"(window={window_min}min)",
                    "STREAM",
                )

        # Clean up rows left behind by a previous aggregator that tiled the
        # calendar day into 390-min slices and saved up to three rows per
        # trading date (stamped 00:00 / 06:30 / 13:00 ET). Every legitimate
        # day bar — historical, reconciler-fetched, or live-aggregated —
        # has timestamp_ms at exactly midnight UTC of its trading date.
        if timespan.startswith("day"):
            purged_day = actual_data_manager.purge_non_daily_anchored(
                self.table_name
            )
            if purged_day:
                k2_logger.info(
                    f"[STREAM START] Purged {purged_day} non-daily-anchored "
                    f"actual bars",
                    "STREAM",
                )

        existing_bars = actual_data_manager.get_all_bars(self.table_name)
        k2_logger.info(
            f"[STREAM START] Existing actual bars after prune: {len(existing_bars)}",
            "STREAM",
        )
        if existing_bars:
            self.actual_data_widget.load_bars(existing_bars)

        self._aggregator = BarAggregator(
            symbol=symbol, frequency=frequency, timespan=timespan
        )
        self._aggregator.bar_updated.connect(self._on_bar_updated)
        self._aggregator.bar_completed.connect(self._on_bar_completed)

        self._reconciler = StreamReconciler()
        self._reconciler.reconciliation_progress.connect(self._on_reconcile_progress)
        self._reconciler.reconciliation_complete.connect(self._on_reconcile_complete)
        self._reconciler.reconciliation_error.connect(self._on_reconcile_error)

        last_actual = actual_data_manager.get_last_bar(self.table_name)
        if last_actual:
            anchor_date = last_actual["date"]
            anchor_time = last_actual["time"]
            k2_logger.info(
                f"[STREAM START] Anchor source: ACTUAL DATA | "
                f"date={anchor_date} time={anchor_time}",
                "STREAM",
            )
        elif model_last_bar:
            anchor_date = model_last_bar[0]
            anchor_time = model_last_bar[1]
            k2_logger.info(
                f"[STREAM START] Anchor source: MODEL LAST BAR | "
                f"date={anchor_date} time={anchor_time}",
                "STREAM",
            )
        else:
            anchor_date = dt_date.today()
            anchor_time = dt_time(9, 30)
            k2_logger.info(
                f"[STREAM START] Anchor source: FALLBACK (today 09:30) | "
                f"date={anchor_date} time={anchor_time}",
                "STREAM",
            )

        if isinstance(anchor_date, str):
            anchor_date = datetime.strptime(anchor_date, "%Y-%m-%d").date()
        if isinstance(anchor_time, str):
            anchor_time = datetime.strptime(anchor_time[:8], "%H:%M:%S").time()

        self._is_streaming = True
        self.stream_status_changed.emit("Reconciling …")

        k2_logger.info(
            f"[STREAM START] Sending to reconciler: symbol={symbol} "
            f"anchor={anchor_date} {anchor_time} "
            f"timespan={timespan} freq={frequency}",
            "STREAM",
        )

        self._reconciler.reconcile(
            symbol=symbol,
            timespan=timespan,
            frequency=frequency,
            last_bar_date=anchor_date,
            last_bar_time=anchor_time,
            market_hours_only=self.current_metadata.get('market_hours_only', False),
        )

    def _stop_streaming(self):
        self._is_streaming = False
        symbol = self.current_metadata.get('symbol', '').upper()

        if self._aggregator:
            self._aggregator.flush()
            try:
                self._aggregator.bar_updated.disconnect(self._on_bar_updated)
                self._aggregator.bar_completed.disconnect(self._on_bar_completed)
            except TypeError:
                pass
            self._aggregator = None

        try:
            polygon_ws_manager.bar_received.disconnect(self._on_ws_bar)
        except TypeError:
            pass
        if symbol:
            polygon_ws_manager.unsubscribe(symbol)

        self.stream_status_changed.emit("Stopped")
        k2_logger.info(f"Streaming stopped: {self.table_name}", "STREAM")

    def _on_reconcile_progress(self, done: int, total: int):
        self.stream_status_changed.emit(f"Backfilling {done}/{total} …")

    def _on_reconcile_complete(self, bars: list):
        if bars:
            actual_data_manager.insert_bars(self.table_name, bars)
            all_bars = actual_data_manager.get_all_bars(self.table_name)
            self.actual_data_widget.load_bars(all_bars)
            self._update_chart_actual_series()

        if self._is_streaming:
            symbol = self.current_metadata.get('symbol', '').upper()
            polygon_ws_manager.bar_received.connect(self._on_ws_bar)
            polygon_ws_manager.subscribe(symbol)
            polygon_ws_manager.start()
            self.stream_status_changed.emit("● Live")

        k2_logger.info(
            f"Reconciliation done for {self.table_name}: {len(bars)} bars backfilled",
            "STREAM",
        )

    def _on_reconcile_error(self, msg: str):
        self.stream_status_changed.emit(f"Error: {msg}")
        k2_logger.error(f"Reconciliation error: {msg}", "STREAM")
        if self._is_streaming:
            symbol = self.current_metadata.get('symbol', '').upper()
            polygon_ws_manager.bar_received.connect(self._on_ws_bar)
            polygon_ws_manager.subscribe(symbol)
            polygon_ws_manager.start()
            self.stream_status_changed.emit("● Live (backfill failed)")

    def _on_ws_bar(self, ws_bar: dict):
        """Route raw Polygon WS bar to this window's aggregator."""
        sym = (ws_bar.get("sym") or "").upper()
        expected = self.current_metadata.get('symbol', '').upper()
        if sym != expected:
            return
        if self._aggregator:
            self._aggregator.ingest(ws_bar)

    def _stamp_bar_time(self, bar: dict):
        """Populate ``date`` / ``time`` / ``timestamp_ms`` on an aggregator
        bar dict from its ``start_ts``.

        Mirrors the UTC→ET conversion used by ``db_manager`` for historical
        rows and by ``StreamReconciler`` for backfilled rows, so every
        persisted bar — historical, reconciler, or live-streamed — shares
        one ``(market_date, market_time, timestamp_ms)`` convention.
        """
        import pytz
        et = pytz.timezone("US/Eastern")

        start_ts = bar.get("start_ts", 0)
        if start_ts:
            utc_dt = datetime.utcfromtimestamp(start_ts / 1000)
            market_dt = pytz.utc.localize(utc_dt).astimezone(et).replace(tzinfo=None)
        else:
            market_dt = datetime.now(et).replace(tzinfo=None)
            start_ts = int(market_dt.timestamp() * 1000)

        bar["date"] = market_dt.date()
        bar["time"] = market_dt.time()
        bar["timestamp_ms"] = start_ts

    def _on_bar_updated(self, bar: dict):
        """Partial / forming bar — refine the current window in real time.

        Every incoming 1-minute WS tick upserts the forming bar into
        ``actual_{table}`` using the same ``timestamp_ms`` its eventual
        ``bar_completed`` twin would use. That gives a live-moving row in
        the ACTUAL DATA table and a live dashed close-price line on the
        chart, and it survives a stream stop / restart or a process crash
        because the partial state is already persisted. When the window
        closes, ``_on_bar_completed`` writes the final OHLCV with the same
        key via upsert, cleanly replacing the last forming snapshot.
        """
        self._stamp_bar_time(bar)
        try:
            actual_data_manager.insert_bar(self.table_name, bar, upsert=True)
        except Exception as e:
            k2_logger.error(f"Forming-bar upsert failed: {e}", "STREAM")
        self.actual_data_widget.update_forming_bar(bar)
        self._update_chart_forming_candle(bar)

    def _on_bar_completed(self, bar: dict):
        """Fully aggregated bar — persist final OHLCV and append to table +
        chart. Uses upsert for all timeframes: for daily it replaces the
        last forming snapshot with the finalized bar; for intraday the
        forming path already wrote the same ``timestamp_ms``, so the final
        write is effectively a no-op on value but guarantees idempotency
        under restart-mid-window.
        """
        self._stamp_bar_time(bar)
        actual_data_manager.insert_bar(self.table_name, bar, upsert=True)
        self.actual_data_widget.append_completed_bar(bar)
        self._update_chart_actual_series()

    def _get_actual_base_x(self) -> int:
        """Dynamic base x-position: always uses current chart data length."""
        if self.chart_widget.data is not None:
            return len(self.chart_widget.data)
        return 0

    def _update_chart_actual_series(self):
        """Refresh the actual-data overlay as candlesticks on the chart."""
        import pyqtgraph as pg
        from PyQt6.QtGui import QPicture, QPainter, QColor
        from PyQt6.QtCore import QRectF

        bars = actual_data_manager.get_all_bars(self.table_name)
        if not bars or self.chart_widget.data is None or len(self.chart_widget.data) == 0:
            return

        old = self._actual_lines.pop("actual_candles", None)
        if old and old.scene():
            self.chart_widget.main_plot.removeItem(old)

        base_x = self._get_actual_base_x()
        picture = QPicture()
        painter = QPainter(picture)

        bull_color = QColor("#00e676")
        bear_color = QColor("#ff1744")
        wick_width = 1
        body_width = 0.6

        for i, bar in enumerate(bars):
            x = base_x + i
            o = float(bar[2]) if bar[2] is not None else 0
            h = float(bar[3]) if bar[3] is not None else 0
            lo = float(bar[4]) if bar[4] is not None else 0
            c = float(bar[5]) if bar[5] is not None else 0

            color = bull_color if c >= o else bear_color

            painter.setPen(pg.mkPen(color=color, width=wick_width))
            painter.drawLine(
                pg.QtCore.QPointF(x, lo),
                pg.QtCore.QPointF(x, h),
            )

            painter.setBrush(pg.mkBrush(color))
            body_top = max(o, c)
            body_bot = min(o, c)
            body_h = body_top - body_bot
            if body_h < 0.01:
                body_h = 0.01
            painter.drawRect(QRectF(
                x - body_width / 2, body_bot,
                body_width, body_h,
            ))

        painter.end()

        candle_item = _PictureItem(picture)
        self.chart_widget.main_plot.addItem(candle_item)
        self._actual_lines["actual_candles"] = candle_item

        k2_logger.info(
            f"[CHART] Drew {len(bars)} actual candles starting at x={base_x} "
            f"(chart data len={len(self.chart_widget.data)})",
            "STREAM",
        )

    def _update_chart_forming_candle(self, bar: dict):
        """Update a transient candlestick + live price line for the in-progress bar."""
        import pyqtgraph as pg
        from PyQt6.QtGui import QPicture, QPainter, QColor, QFont
        from PyQt6.QtCore import QRectF

        if self.chart_widget.data is None or len(self.chart_widget.data) == 0:
            return

        actual_item = self._actual_lines.get("actual_candles")
        if actual_item is None or actual_item.scene() is None:
            self._update_chart_actual_series()

        for key in ("_forming_candle", "_price_line", "_price_label"):
            old = self._actual_lines.pop(key, None)
            if old and old.scene():
                self.chart_widget.main_plot.removeItem(old)

        o = bar.get("open")
        h = bar.get("high")
        lo = bar.get("low")
        c = bar.get("close")
        if c is None or o is None:
            return

        completed_bars = actual_data_manager.get_all_bars(self.table_name)
        base_x = self._get_actual_base_x() + len(completed_bars)

        o, h, lo, c = float(o), float(h), float(lo), float(c)
        is_bull = c >= o
        forming_color = QColor("#00e676") if is_bull else QColor("#ff1744")
        forming_color.setAlpha(160)

        picture = QPicture()
        painter = QPainter(picture)

        painter.setPen(pg.mkPen(color=forming_color, width=1))
        painter.drawLine(
            pg.QtCore.QPointF(base_x, lo),
            pg.QtCore.QPointF(base_x, h),
        )

        painter.setBrush(pg.mkBrush(forming_color))
        body_top = max(o, c)
        body_bot = min(o, c)
        body_h = max(body_top - body_bot, 0.01)
        painter.drawRect(QRectF(
            base_x - 0.3, body_bot,
            0.6, body_h,
        ))

        painter.end()

        item = _PictureItem(picture)
        self.chart_widget.main_plot.addItem(item)
        self._actual_lines["_forming_candle"] = item

        line_color = "#00e676" if is_bull else "#ff1744"
        price_line = pg.InfiniteLine(
            pos=c, angle=0, movable=False,
            pen=pg.mkPen(color=line_color, width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
        )
        self.chart_widget.main_plot.addItem(price_line, ignoreBounds=True)
        self._actual_lines["_price_line"] = price_line

        symbol = self.current_metadata.get('symbol', '')
        label_html = (
            f'<div style="background:{line_color}; color:#fff; padding:2px 6px; '
            f'font-size:11px; font-family:Consolas,monospace; font-weight:bold; '
            f'border-radius:2px;">'
            f'{symbol} {c:.2f}</div>'
        )
        price_label = pg.TextItem(html=label_html, anchor=(0, 0.5))
        vb = self.chart_widget.main_plot.getViewBox()
        x_max = vb.viewRange()[0][1]
        price_label.setPos(x_max - 1, c)
        self.chart_widget.main_plot.addItem(price_label, ignoreBounds=True)
        self._actual_lines["_price_label"] = price_label
        self._last_forming_price = c

    def _reposition_price_label(self):
        """Keep the price label anchored to the right edge of the visible area."""
        label = self._actual_lines.get("_price_label")
        if label is None or self._last_forming_price is None:
            return
        try:
            vb = self.chart_widget.main_plot.getViewBox()
            x_max = vb.viewRange()[0][1]
            label.setPos(x_max - 1, self._last_forming_price)
        except Exception:
            pass

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
            if self._is_streaming:
                try:
                    self._stop_streaming()
                except Exception:
                    pass
            if self._reconciler:
                self._reconciler.cancel()

            # Cancel any in-flight strategy runs this window submitted and
            # disconnect from the shared runner so signals don't reach a
            # partially-destroyed widget. Other windows remain subscribed.
            for key in list(self._strategy_jobs.keys()):
                try:
                    strategy_runner.cancel(key)
                except Exception:
                    pass
            self._strategy_jobs.clear()
            for sig in (strategy_runner.started, strategy_runner.progress,
                        strategy_runner.finished, strategy_runner.failed,
                        strategy_runner.cancelled):
                try:
                    sig.disconnect(self)
                except (TypeError, RuntimeError):
                    pass

            for key, item in self._actual_lines.items():
                if item and item.scene():
                    self.chart_widget.main_plot.removeItem(item)
            self._actual_lines.clear()

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
