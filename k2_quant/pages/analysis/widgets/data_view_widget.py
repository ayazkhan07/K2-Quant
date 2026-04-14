"""
Inline data-view widget for embedding tabular results in the Thinkspace chat.

Renders a list-of-dicts or list-of-lists with headers as a sortable,
optionally editable QTableWidget.  Designed to replace the read-only
HTML pipe-table rendering for structured agent output.
"""

from typing import List, Optional, Any

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QPushButton, QSizePolicy, QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class InlineDataView(QFrame):
    """Sortable, optionally editable table that lives inside the chat scroll area."""

    cell_edited = pyqtSignal(int, int, str)    # row, col, new_value
    send_to_workspace = pyqtSignal(list, list)  # (rows, headers)

    def __init__(
        self,
        rows: List[List[Any]],
        headers: List[str],
        label: str = "",
        editable: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._rows = [list(r) for r in rows]
        self._headers = list(headers)
        self._label = label
        self._editable = editable
        self.setObjectName("inlineDataView")
        self._init_ui()
        self._apply_style()

    # ── build ────────────────────────────────────────────────────

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)

        if self._label:
            title = QLabel(self._label)
            title.setStyleSheet("color:#4a9eff; font-size:11px; background:transparent;")
            toolbar.addWidget(title)

        info = QLabel(f"{len(self._rows)} rows × {len(self._headers)} cols")
        info.setStyleSheet("color:#666; font-size:10px; background:transparent;")
        toolbar.addWidget(info)
        toolbar.addStretch()

        copy_btn = QPushButton("Copy TSV")
        copy_btn.setFixedHeight(20)
        copy_btn.clicked.connect(self._copy_tsv)
        toolbar.addWidget(copy_btn)

        ws_btn = QPushButton(">> Workspace")
        ws_btn.setFixedHeight(20)
        ws_btn.clicked.connect(self._emit_workspace)
        toolbar.addWidget(ws_btn)

        root.addLayout(toolbar)

        n_rows = len(self._rows)
        n_cols = len(self._headers)

        self._table = QTableWidget(n_rows, n_cols)
        self._table.setFont(QFont("Consolas", 10))
        self._table.setHorizontalHeaderLabels(self._headers)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self._table.setSortingEnabled(True)

        for r, row in enumerate(self._rows):
            for c in range(n_cols):
                val = row[c] if c < len(row) else ""
                item = QTableWidgetItem(self._fmt(val))
                item.setData(Qt.ItemDataRole.UserRole, val)
                if self._is_numeric(val):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                else:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    )
                if not self._editable:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(r, c, item)

        visible_rows = min(n_rows, 15)
        row_h = self._table.verticalHeader().defaultSectionSize()
        header_h = 24
        self._table.setFixedHeight(visible_rows * row_h + header_h + 4)

        if self._editable:
            self._table.cellChanged.connect(self._on_cell_changed)

        root.addWidget(self._table)

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _fmt(val) -> str:
        if val is None:
            return ""
        if isinstance(val, float):
            return f"{val:.6g}"
        return str(val)

    @staticmethod
    def _is_numeric(val) -> bool:
        return isinstance(val, (int, float))

    def _on_cell_changed(self, row: int, col: int):
        text = self._table.item(row, col).text().strip()
        self._rows[row][col] = text
        self.cell_edited.emit(row, col, text)

    def _copy_tsv(self):
        from PyQt6.QtWidgets import QApplication
        lines = ["\t".join(self._headers)]
        for row in self._rows:
            lines.append("\t".join(self._fmt(v) for v in row))
        QApplication.clipboard().setText("\n".join(lines))

    def _emit_workspace(self):
        self.send_to_workspace.emit(self._rows, self._headers)

    def get_rows(self) -> List[List]:
        return [list(r) for r in self._rows]

    # ── style ────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet("""
            #inlineDataView {
                background-color: #111;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
            }
            QTableWidget {
                background-color: #0d0d0d;
                color: #ddd;
                gridline-color: #222;
                border: none;
                selection-background-color: #1a3a5c;
            }
            QTableWidget::item {
                padding: 2px 4px;
            }
            QHeaderView::section {
                background-color: #1a1a1a;
                color: #999;
                border: 1px solid #222;
                padding: 2px 4px;
                font-size: 10px;
            }
            QPushButton {
                background-color: #1a1a1a;
                color: #999;
                border: 1px solid #2a2a2a;
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
                color: #fff;
            }
        """)
