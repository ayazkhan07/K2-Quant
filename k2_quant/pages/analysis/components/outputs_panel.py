"""
Outputs Panel — strategy run history, reports, and code viewer.

Lives as the second tab in the right pane alongside Thinkspace.
"""

import json
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QSplitter, QTreeWidget,
    QTreeWidgetItem, QTextEdit, QLabel, QPushButton, QStackedWidget,
    QWidget, QSizePolicy, QHeaderView, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtGui import QFont, QColor

from k2_quant.utilities.logger import k2_logger
from k2_quant.utilities.services.strategy_service import strategy_service
from k2_quant.utilities.strategy_validator import validate_strategy, format_errors
from k2_quant.pages.analysis.widgets.line_number_editor import LineNumberEditor


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
        self.tree.setHeaderLabels(["Strategy / Run"])
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

        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setObjectName("outputText")
        self.report_view.setFont(QFont("Consolas", 10))
        self.stack.addWidget(self.report_view)

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

    # ── data loading ─────────────────────────────────────────────

    def refresh(self):
        """Reload the run tree from the database."""
        self.tree.clear()
        names = strategy_service.get_strategy_names_with_runs()
        for name in names:
            parent = QTreeWidgetItem(self.tree, [name])
            parent.setData(0, Qt.ItemDataRole.UserRole, {"type": "strategy", "name": name})
            parent.setExpanded(False)
            runs = strategy_service.get_runs(strategy_name=name, limit=50)
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

        # report
        lines = []
        lines.append(f"Strategy:  {run['strategy_name']}")
        lines.append(f"Model:     {run.get('model_table', '--')}")
        lines.append(f"Timestamp: {run['run_timestamp']}")
        lines.append(f"Status:    {'SUCCESS' if run.get('success') else 'FAILED'}")
        lines.append(f"Time:      {run.get('execution_time_ms', 0):.0f} ms")
        lines.append("")

        stdout = run.get('stdout_output', '').strip()
        if stdout:
            lines.append("--- stdout ---")
            lines.append(stdout)
            lines.append("")

        error = run.get('error_output', '').strip()
        if error:
            lines.append("--- error ---")
            lines.append(error)
            lines.append("")

        tw_json = run.get('tab_writes_json', '[]')
        try:
            writes = json.loads(tw_json)
        except (json.JSONDecodeError, TypeError):
            writes = []
        if writes:
            lines.append("--- outputs ---")
            for w in writes:
                wtype = w.get('type', '?')
                col = w.get('column_name', w.get('set_index', ''))
                length = w.get('length', '?')
                preview = w.get('preview', [])
                lines.append(f"  {wtype}: {col}  ({length} values)")
                if preview:
                    lines.append(f"    preview: {preview}")
            lines.append("")

        metrics_json = run.get('metrics_json', '{}')
        try:
            metrics = json.loads(metrics_json)
        except (json.JSONDecodeError, TypeError):
            metrics = {}
        if metrics:
            lines.append("--- metrics ---")
            for k, v in metrics.items():
                lines.append(f"  {k}: {v}")

        self.report_view.setPlainText("\n".join(lines))

        self._load_code_for_run(run)
        self._show_page(0)

    def _show_strategy_code(self, strat: Dict[str, Any]):
        self._maybe_save_code()
        self._current_run = None
        self._current_strategy_name = strat['name']
        self.ref_btn.setEnabled(False)

        self.report_view.setPlainText(
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
            passed, errors = validate_strategy(new_code)
            if not passed:
                QMessageBox.warning(
                    self,
                    "Validation Failed",
                    format_errors(errors),
                )
                return

            strategy_service.save_strategy(
                self._current_strategy_name,
                new_code,
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
