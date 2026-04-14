"""
Outputs Panel — Strategy Review: code viewer, run history, and reports.

Lives as the second tab in the right pane alongside Thinkspace.

Lists **all** saved strategies so users can review code at any time.
Runs (with reports) appear as children when available.

When strategies are deleted, the analysis page calls ``handle_strategy_deleted``
so the tree matches the DB — see ``strategy_lifecycle_rules``.
"""

import json
import numbers
from typing import Optional, Dict, Any, List

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget,
    QTreeWidgetItem, QTextEdit, QLabel, QPushButton, QStackedWidget,
    QWidget, QHeaderView, QMessageBox, QScrollArea, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QSizePolicy, QApplication,
    QMenu,
)
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtGui import QFont, QColor, QShortcut, QKeySequence

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.services.strategy_service import strategy_service
from k2_quant.utilities.strategy_validator import format_errors, format_warnings
from k2_quant.pages.analysis.widgets.line_number_editor import LineNumberEditor
from k2_quant.utilities.numeric_rounding import (
    format_computation_for_display,
    format_price_for_display,
    report_column_looks_price,
)


class OutputsPanel(QFrame):
    """Strategy run history with report and code views."""

    # Emitted when user wants to reference a run in Thinkspace
    reference_in_chat = pyqtSignal(int, str)  # run_id, strategy_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("outputsPanel")
        self._current_run: Optional[Dict[str, Any]] = None
        self._current_strategy_name: Optional[str] = None
        self._code_dirty = False
        self._code_original = ""
        self._init_ui()
        self._apply_style()

    def showEvent(self, event):
        """Auto-refresh every time the panel becomes visible."""
        super().showEvent(event)
        self.refresh()

    # ── UI ────────────────────────────────────────────────────────

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addSpacing(4)

        # main splitter: tree (left) | detail (right)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(1)
        self.splitter.setStyleSheet("QSplitter::handle { background: #1a1a1a; }")

        # ── left: run tree ───────────────────────────────────────
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["STRATEGIES"])
        self.tree.setIndentation(16)
        self.tree.setRootIsDecorated(True)
        self.tree.header().setStretchLastSection(True)
        self.tree.itemClicked.connect(self._on_tree_click)
        self.splitter.addWidget(self.tree)

        # ── right: detail stack ──────────────────────────────────
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(0)

        # toggle buttons
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(8, 6, 8, 6)

        self.report_btn = QPushButton("Report")
        self.report_btn.setCheckable(True)
        self.report_btn.setChecked(True)
        self.report_btn.setFixedHeight(24)
        self.report_btn.setObjectName("outputToggle")
        self.report_btn.clicked.connect(lambda: self._show_page(0))
        toggle_row.addWidget(self.report_btn)

        self.code_btn = QPushButton("Code")
        self.code_btn.setCheckable(True)
        self.code_btn.setFixedHeight(24)
        self.code_btn.setObjectName("outputToggle")
        self.code_btn.clicked.connect(lambda: self._show_page(1))
        toggle_row.addWidget(self.code_btn)

        toggle_row.addStretch()

        self.ref_btn = QPushButton("Send to Thinkspace")
        self.ref_btn.setFixedHeight(24)
        self.ref_btn.setObjectName("outputBtn")
        self.ref_btn.clicked.connect(self._send_to_chat)
        self.ref_btn.setEnabled(False)
        toggle_row.addWidget(self.ref_btn)

        detail_layout.addLayout(toggle_row)

        # stacked: report | code
        self.stack = QStackedWidget()

        self.report_scroll = QScrollArea()
        self.report_scroll.setObjectName("outputReportScroll")
        self.report_scroll.setWidgetResizable(True)
        self.report_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.report_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.report_inner = QWidget()
        self.report_layout = QVBoxLayout(self.report_inner)
        self.report_layout.setContentsMargins(10, 8, 10, 12)
        self.report_layout.setSpacing(0)
        self.report_scroll.setWidget(self.report_inner)
        self.stack.addWidget(self.report_scroll)

        self.code_view = LineNumberEditor()
        self.code_view.setReadOnly(False)
        self.code_view.setObjectName("codeEditor")
        self.code_view.setFont(QFont("Consolas", 10))
        self.code_view.textChanged.connect(self._on_code_changed)
        self.code_view.installEventFilter(self)
        self.stack.addWidget(self.code_view)

        detail_layout.addWidget(self.stack)
        self.splitter.addWidget(detail)

        self.splitter.setSizes([240, 500])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        root.addWidget(self.splitter)

    # ── report body (structured tables + plain fallback) ─────────

    def _clear_report_body(self):
        while self.report_layout.count():
            item = self.report_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            lay = item.layout()
            if lay is not None:
                self._clear_layout(lay)

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            sub = item.layout()
            if sub is not None:
                OutputsPanel._clear_layout(sub)

    def _set_report_plain_only(self, text: str):
        self._clear_report_body()
        te = QTextEdit()
        te.setReadOnly(True)
        te.setObjectName("outputText")
        te.setFont(QFont("Consolas", 10))
        te.setPlainText(text)
        self.report_layout.addWidget(te)
        self.report_layout.addStretch(1)

    @staticmethod
    def _report_cell_text(v: Any, column_header: str = "") -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, numbers.Integral):
            return f"{int(v):,}"
        if isinstance(v, numbers.Real):
            if report_column_looks_price(column_header):
                return format_price_for_display(v)
            return format_computation_for_display(v)
        return str(v)

    @staticmethod
    def _format_preview_list(preview: Any) -> str:
        if preview is None:
            return ""
        if not isinstance(preview, list):
            return str(preview)
        parts: List[str] = []
        for x in preview:
            if x is None:
                parts.append("")
            elif isinstance(x, numbers.Real):
                parts.append(format_price_for_display(x))
            else:
                parts.append(str(x))
        return "[" + ", ".join(parts) + "]"

    def _append_meta_row(self, label: str, value: str):
        row = QHBoxLayout()
        row.setContentsMargins(8, 0, 0, 2)
        a = QLabel(label)
        a.setStyleSheet("color: #777; font-size: 11px; min-width: 78px;")
        b = QLabel(value)
        b.setStyleSheet(
            "color: #d0d0d0; font-size: 11px; font-family: 'Consolas', monospace;"
        )
        b.setWordWrap(True)
        b.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        row.addWidget(a, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(b, 1)
        wrap = QWidget()
        wrap.setLayout(row)
        self.report_layout.addWidget(wrap)

    def _append_report_title(self, text: str, *, large: bool = False):
        lab = QLabel(text)
        if large:
            lab.setStyleSheet(
                "color: #eaeaea; font-size: 14px; font-weight: 600; "
                "padding: 2px 0 6px 0; background: transparent;"
            )
        else:
            lab.setStyleSheet(
                "color: #4a9eff; font-size: 11px; font-weight: 600; "
                "padding: 8px 0 4px 0; background: transparent;"
            )
        lab.setWordWrap(True)
        lab.setMaximumWidth(720)
        self._append_centered(lab)

    def _append_note(self, text: str, *, color: str = "#888"):
        lab = QLabel(text)
        lab.setStyleSheet(
            f"color: {color}; font-size: 10px; padding: 2px 0 6px 2px;"
        )
        lab.setWordWrap(True)
        lab.setMaximumWidth(720)
        self._append_centered(lab)

    def _append_centered(self, widget: QWidget):
        row = QHBoxLayout()
        row.setContentsMargins(8, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(widget, 0, Qt.AlignmentFlag.AlignTop)
        row.addStretch(1)
        wrap = QWidget()
        wrap.setLayout(row)
        self.report_layout.addWidget(wrap)

    # ── report table copy helpers ──────────────────────────────

    def _copy_table_selection(self, tbl: QTableWidget):
        """Copy selected cells as tab-separated text to clipboard."""
        sel = tbl.selectedIndexes()
        if not sel:
            return
        rows_map: dict = {}
        for idx in sel:
            rows_map.setdefault(idx.row(), {})[idx.column()] = idx.data() or ''
        min_col = min(c for cols in rows_map.values() for c in cols)
        max_col = max(c for cols in rows_map.values() for c in cols)
        lines = []
        for r in sorted(rows_map):
            cells = [str(rows_map[r].get(c, '')) for c in range(min_col, max_col + 1)]
            lines.append('\t'.join(cells))
        QApplication.clipboard().setText('\n'.join(lines))

    def _copy_table_row(self, tbl: QTableWidget, row: int):
        ncols = tbl.columnCount()
        cells = []
        for c in range(ncols):
            item = tbl.item(row, c)
            cells.append(item.text() if item else '')
        QApplication.clipboard().setText('\t'.join(cells))

    def _copy_full_table(self, tbl: QTableWidget):
        """Copy entire table including headers as TSV."""
        ncols = tbl.columnCount()
        nrows = tbl.rowCount()
        hdr_cells = []
        for c in range(ncols):
            hi = tbl.horizontalHeaderItem(c)
            hdr_cells.append(hi.text() if hi else '')
        lines = ['\t'.join(hdr_cells)]
        for r in range(nrows):
            cells = []
            for c in range(ncols):
                item = tbl.item(r, c)
                cells.append(item.text() if item else '')
            lines.append('\t'.join(cells))
        QApplication.clipboard().setText('\n'.join(lines))

    def _report_table_context_menu(self, tbl: QTableWidget, pos):
        menu = QMenu(tbl)
        menu.setStyleSheet("""
            QMenu { background: #1a1a1a; color: #ddd; border: 1px solid #333; }
            QMenu::item:selected { background: #2a3f5f; }
        """)
        idx = tbl.indexAt(pos)
        if idx.isValid():
            cell_val = idx.data() or ''
            act_cell = menu.addAction(f"Copy Cell")
            act_cell.triggered.connect(
                lambda: QApplication.clipboard().setText(str(cell_val)))
            act_row = menu.addAction("Copy Row")
            act_row.triggered.connect(
                lambda: self._copy_table_row(tbl, idx.row()))
        sel = tbl.selectedIndexes()
        if sel and len(sel) > 1:
            act_sel = menu.addAction(f"Copy Selection ({len(sel)} cells)")
            act_sel.triggered.connect(lambda: self._copy_table_selection(tbl))
        menu.addSeparator()
        act_all = menu.addAction("Copy Entire Table")
        act_all.triggered.connect(lambda: self._copy_full_table(tbl))
        menu.exec(tbl.viewport().mapToGlobal(pos))

    @staticmethod
    def _size_table_to_full_content(tbl: QTableWidget) -> None:
        """Expand table to exact content height and hide inner scrollbars."""
        tbl.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tbl.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        hdr = tbl.horizontalHeader()
        hdr_h = max(20, hdr.sizeHint().height())
        rows_h = 0
        for r in range(tbl.rowCount()):
            rows_h += max(18, tbl.rowHeight(r))
        frame_pad = 2
        tbl.setFixedHeight(hdr_h + rows_h + frame_pad)
        tbl.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )

    def _append_data_table(self, title: str, headers: List[str], rows: List[List[Any]]):
        self._append_report_title(title, large=False)
        ncols = len(headers)
        nrows = len(rows)
        tbl = QTableWidget(nrows, ncols)
        tbl.setObjectName("outputReportTable")
        tbl.setFont(QFont("Consolas", 10))
        tbl.setHorizontalHeaderLabels(headers)
        tbl.verticalHeader().setVisible(False)
        tbl.setShowGrid(True)
        tbl.setAlternatingRowColors(True)
        tbl.setSortingEnabled(False)
        tbl.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        tbl.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        tbl.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        tbl.customContextMenuRequested.connect(
            lambda pos, t=tbl: self._report_table_context_menu(t, pos)
        )
        tbl.setWordWrap(False)
        tbl.verticalHeader().setDefaultSectionSize(20)
        tbl.setStyleSheet("""
            QTableWidget {
                background-color: #0d0d0d;
                alternate-background-color: #111111;
                color: #ccc;
                gridline-color: #2a2a2a;
                border: 1px solid #2a2a2a;
                border-radius: 2px;
            }
            QTableWidget::item {
                padding: 1px 4px;
            }
            QTableWidget::item:selected {
                background-color: #1a3a5c;
                color: #fff;
            }
            QHeaderView::section {
                background-color: #141414;
                color: #9a9a9a;
                padding: 2px 4px;
                border: 1px solid #2a2a2a;
                font-size: 10px;
                font-weight: 600;
            }
        """)
        sc = QShortcut(QKeySequence.StandardKey.Copy, tbl)
        sc.setContext(Qt.ShortcutContext.WidgetShortcut)
        sc.activated.connect(lambda t=tbl: self._copy_table_selection(t))
        hdr = tbl.horizontalHeader()
        hdr.setStretchLastSection(False)

        for r, row in enumerate(rows):
            for c in range(ncols):
                raw = row[c] if c < len(row) else None
                col_hdr = headers[c] if c < len(headers) else ""
                txt = self._report_cell_text(raw, col_hdr)
                item = QTableWidgetItem(txt)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if isinstance(raw, bool):
                    pass
                elif isinstance(raw, numbers.Real):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                elif isinstance(raw, numbers.Integral):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                tbl.setItem(r, c, item)

        tbl.resizeRowsToContents()
        for c in range(ncols):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        tbl.resizeColumnsToContents()
        total_w = max(120, sum(tbl.columnWidth(c) for c in range(ncols)) + 4)
        tbl.setFixedWidth(total_w)

        self._size_table_to_full_content(tbl)
        self._append_centered(tbl)

    def _append_report_blocks(self, blocks: List[dict]):
        for b in blocks:
            kind = b.get("kind")
            if kind == "header":
                self._append_report_title(str(b.get("title", "")), large=True)
            elif kind == "config":
                title = str(b.get("title", "Configuration"))
                headers = b.get("headers") or []
                rows = b.get("rows") or []
                self._append_data_table(title, headers, rows)
            elif kind == "table":
                title = str(b.get("title", ""))
                headers = b.get("headers") or []
                rows = b.get("rows") or []
                self._append_data_table(title, headers, rows)
                tr = int(b.get("truncated") or 0)
                if tr > 0:
                    self._append_note(f"{tr} additional rows not shown.", color="#666")

    def _append_log_text(self, heading: str, body: str, *, err: bool = False):
        self._append_report_title(heading, large=False)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(body)
        te.setFont(QFont("Consolas", 9))
        te.setMaximumHeight(160)
        te.setStyleSheet(
            "QTextEdit { background: #0a0a0a; color: %s; border: 1px solid #2a2a2a; }"
            % ("#ff8080" if err else "#999")
        )
        self.report_layout.addWidget(te)

    # ── data loading ─────────────────────────────────────────────

    def handle_strategy_deleted(self, strategy_name: str):
        """Clear UI state for a removed strategy and reload the run tree.

        Invoked from the analysis page when the left pane deletes a strategy.
        See ``k2_quant.utilities.services.strategy_lifecycle_rules.RULE 2``.
        """
        strategy_name = (strategy_name or "").strip()
        self._code_dirty = False
        cur_run_name = (self._current_run.get("strategy_name") or "").strip() if self._current_run else ""
        if cur_run_name == strategy_name:
            self._reset_detail_pane()
        if (self._current_strategy_name or "").strip() == strategy_name:
            self._reset_detail_pane()
        self.refresh()

    def _reset_detail_pane(self):
        """Empty report/code and disable run actions (no save prompts)."""
        self._current_run = None
        self._current_strategy_name = None
        self._code_original = ""
        self.code_view.setPlainText("")
        self._set_report_plain_only(
            "Select a strategy run or strategy to view details."
        )
        self.ref_btn.setEnabled(False)

    def refresh(self):
        """Reload the strategy tree from the database.

        Lists **all** saved strategies (Strategy Review) so users can review
        code for any strategy.  Runs are shown as children when available.
        """
        strategy_service.prune_orphan_strategy_runs()
        self.tree.clear()
        names = strategy_service.get_all_strategy_names()
        for name in names:
            parent = QTreeWidgetItem(self.tree, [name])
            parent.setData(0, Qt.ItemDataRole.UserRole, {"type": "strategy", "name": name})
            parent.setExpanded(False)
            runs = strategy_service.get_runs(strategy_name=name, limit=strategy_service.MAX_RUNS_PER_STRATEGY)
            for run in runs:
                ts = run.get('run_timestamp', '')
                ok = "OK" if run.get('success') else "FAIL"
                ms = run.get('execution_time_ms', 0)
                label = f"{ts}  [{ok}  {ms:.0f}ms]"
                child = QTreeWidgetItem(parent, [label])
                child.setData(0, Qt.ItemDataRole.UserRole, {
                    "type": "run",
                    "id": run['id'],
                    "name": name,
                })
                if not run.get('success'):
                    child.setForeground(0, QColor("#ff5555"))

    def _on_tree_click(self, item: QTreeWidgetItem, column: int):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        if data["type"] == "run":
            run = strategy_service.get_run(data["id"])
            if run:
                self._show_run(run)
        elif data["type"] == "strategy":
            strat = strategy_service.get_strategy(data["name"])
            if strat:
                self._show_strategy_code(strat)

    def _show_run(self, run: Dict[str, Any]):
        self._maybe_save_code()
        self._current_run = run
        self.ref_btn.setEnabled(True)

        self._clear_report_body()
        self._append_meta_row("Strategy", str(run.get("strategy_name", "")))
        self._append_meta_row("Model", str(run.get("model_table", "--")))
        self._append_meta_row("Timestamp", str(run.get("run_timestamp", "")))
        self._append_meta_row(
            "Status", "SUCCESS" if run.get("success") else "FAILED"
        )
        self._append_meta_row(
            "Time", f"{run.get('execution_time_ms', 0):.0f} ms"
        )

        blocks: List[dict] = []
        rb = run.get("report_blocks_json")
        if rb:
            try:
                blocks = json.loads(rb)
                if not isinstance(blocks, list):
                    blocks = []
            except (json.JSONDecodeError, TypeError):
                blocks = []

        if blocks:
            self._append_note(
                "Computation steps and nomenclature (structured report).",
                color="#777",
            )
            self._append_report_blocks(blocks)
        else:
            stdout_fb = (run.get("stdout_output") or "").strip()
            if stdout_fb:
                self._append_note(
                    "Legacy run: report shown as plain text (no structured blocks).",
                    color="#777",
                )
                self._append_log_text("Captured output", stdout_fb)

        stdout = (run.get("stdout_output") or "").strip()
        if stdout and blocks:
            self._append_log_text("Log (stdout)", stdout)

        error = (run.get("error_output") or "").strip()
        if error:
            self._append_log_text("Error", error, err=True)

        tw_json = run.get("tab_writes_json") or "[]"
        try:
            writes = json.loads(tw_json)
        except (json.JSONDecodeError, TypeError):
            writes = []
        if writes:
            wr_rows = []
            for w in writes:
                wtype = w.get("type", "?")
                col = w.get("column_name", w.get("set_index", ""))
                length = w.get("length", "?")
                preview = w.get("preview", [])
                pv = self._format_preview_list(preview) if preview else ""
                wr_rows.append([str(wtype), str(col), str(length), pv])
            self._append_data_table(
                "Forecast / tab writes",
                ["type", "column / set", "length", "preview"],
                wr_rows,
            )

        metrics_json = run.get("metrics_json", "{}")
        try:
            metrics = json.loads(metrics_json)
        except (json.JSONDecodeError, TypeError):
            metrics = {}
        if metrics:
            mrows = [
                [str(k), self._report_cell_text(v)]
                for k, v in metrics.items()
            ]
            self._append_data_table("Metrics", ["name", "value"], mrows)

        self.report_layout.addStretch(1)

        self._load_code_for_run(run)
        self._show_page(0)

    def _show_strategy_code(self, strat: Dict[str, Any]):
        self._maybe_save_code()
        self._current_run = None
        self._current_strategy_name = strat['name']
        self.ref_btn.setEnabled(False)

        self._set_report_plain_only(
            f"Strategy: {strat['name']}\n"
            f"Description: {strat.get('description', '--')}\n"
            f"Category: {strat.get('category', '--')}\n"
            f"Created: {strat.get('created_at', '--')}\n"
            f"Updated: {strat.get('updated_at', '--')}\n"
        )
        code = strat.get('code', '')
        self._code_original = code
        self._code_dirty = False
        self.code_view.setPlainText(code)
        self._show_page(1)

    def _load_code_for_run(self, run: Dict[str, Any]):
        """Load code from a run snapshot (also editable -- saves to strategy)."""
        self._current_strategy_name = run.get('strategy_name')
        code = run.get('code_snapshot', '')
        self._code_original = code
        self._code_dirty = False
        self.code_view.setPlainText(code)

    # -- code editing ----------------------------------------------------

    def _on_code_changed(self):
        if self.code_view.toPlainText() != self._code_original:
            self._code_dirty = True

    def eventFilter(self, obj, event):
        if obj is self.code_view and event.type() == QEvent.Type.FocusOut:
            self._maybe_save_code()
        return super().eventFilter(obj, event)

    def _maybe_save_code(self):
        """If code was edited, validate then prompt the user to save."""
        if not self._code_dirty or not self._current_strategy_name:
            return
        new_code = self.code_view.toPlainText()
        if new_code == self._code_original:
            self._code_dirty = False
            return

        reply = QMessageBox.question(
            self,
            "Save Changes",
            f"Save code changes to strategy '{self._current_strategy_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply == QMessageBox.StandardButton.Yes:
            ok, errors, warnings = strategy_service.save_strategy(
                self._current_strategy_name,
                new_code,
            )
            if not ok:
                QMessageBox.warning(
                    self,
                    "Validation Failed",
                    format_errors(errors),
                )
                return
            if warnings:
                QMessageBox.information(
                    self,
                    "Strategy frame contract",
                    format_warnings(warnings),
                )
            self._code_original = new_code
            k2_logger.info(
                f"Strategy code updated: {self._current_strategy_name}",
                "OUTPUTS",
            )
        else:
            self.code_view.setPlainText(self._code_original)
        self._code_dirty = False

    # -- page switching --------------------------------------------------

    def _show_page(self, index: int):
        if self.stack.currentIndex() == 1 and index != 1:
            self._maybe_save_code()
        self.stack.setCurrentIndex(index)
        self.report_btn.setChecked(index == 0)
        self.code_btn.setChecked(index == 1)

    def _send_to_chat(self):
        if self._current_run:
            self.reference_in_chat.emit(
                self._current_run['id'],
                self._current_run['strategy_name'],
            )

    # ── style ────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet("""
            #outputsPanel {
                background-color: #0a0a0a;
            }
            #sectionTitle {
                color: #4a9eff;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 1px;
                background: transparent;
            }
            QTreeWidget {
                background-color: #0d0d0d;
                color: #ccc;
                border: none;
                font-size: 11px;
            }
            QTreeWidget::item {
                padding: 3px 4px;
            }
            QTreeWidget::item:selected {
                background-color: #1a3a5c;
                color: #fff;
            }
            QTreeWidget::item:hover {
                background-color: #1a1a1a;
            }
            QHeaderView::section {
                background-color: #0a0a0a;
                color: #666;
                border: none;
                border-bottom: 1px solid #1a1a1a;
                padding: 4px;
                font-size: 10px;
            }
            #outputReportScroll {
                background-color: #0a0a0a;
                border: none;
            }
            #outputText {
                background-color: #0d0d0d;
                color: #ccc;
                border: none;
                selection-background-color: #1a3a5c;
            }
            #codeEditor {
                background-color: #0d0d0d;
                color: #a8c7e8;
                border: none;
                selection-background-color: #1a3a5c;
            }
            QPlainTextEdit {
                background-color: #0d0d0d;
                color: #a8c7e8;
                border: none;
                selection-background-color: #1a3a5c;
            }
            #outputBtn {
                background-color: #1a1a1a;
                color: #999;
                border: 1px solid #2a2a2a;
                border-radius: 3px;
                padding: 2px 10px;
                font-size: 11px;
            }
            #outputBtn:hover {
                background-color: #2a2a2a;
                color: #fff;
            }
            #outputToggle {
                background-color: #1a1a1a;
                color: #999;
                border: 1px solid #2a2a2a;
                border-radius: 3px;
                padding: 2px 12px;
                font-size: 11px;
            }
            #outputToggle:checked {
                background-color: #1a3a5c;
                color: #fff;
                border-color: #2a5a8c;
            }
        """)
