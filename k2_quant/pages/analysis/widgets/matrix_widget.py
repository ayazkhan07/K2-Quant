"""
Interactive matrix / vector widget for embedding in the Thinkspace chat.

Renders a 2-D list as an editable QTableWidget with compact dark styling.
Emits ``matrix_edited`` with the full matrix (2-D list) after every cell edit
so the caller can feed changes back into the conversation or workspace.
"""

from typing import List, Optional

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QPushButton, QSizePolicy, QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor


class MatrixWidget(QFrame):
    """Compact editable grid that lives inside the chat scroll area."""

    matrix_edited = pyqtSignal(list)          # full 2-D list after any edit
    send_to_workspace = pyqtSignal(list, str) # (2-D list, label)

    def __init__(
        self,
        data: List[List],
        label: str = "",
        editable: bool = True,
        headers: Optional[List[str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._data = [list(row) for row in data]
        self._label = label
        self._editable = editable
        self._headers = headers
        self.setObjectName("matrixWidget")
        self._init_ui()
        self._apply_style()

    # ── build ────────────────────────────────────────────────────

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(4)

        # toolbar row
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)

        if self._label:
            title = QLabel(self._label)
            title.setStyleSheet("color:#4a9eff; font-size:11px; background:transparent;")
            toolbar.addWidget(title)

        rows = len(self._data)
        cols = len(self._data[0]) if self._data else 0
        dim = QLabel(f"{rows}×{cols}")
        dim.setStyleSheet("color:#666; font-size:10px; background:transparent;")
        toolbar.addWidget(dim)
        toolbar.addStretch()

        copy_btn = QPushButton("Copy")
        copy_btn.setFixedHeight(20)
        copy_btn.clicked.connect(self._copy_tsv)
        toolbar.addWidget(copy_btn)

        ws_btn = QPushButton(">> Workspace")
        ws_btn.setFixedHeight(20)
        ws_btn.clicked.connect(self._emit_workspace)
        toolbar.addWidget(ws_btn)

        root.addLayout(toolbar)

        # table
        self._table = QTableWidget(rows, cols)
        self._table.setFont(QFont("Consolas", 10))
        self._table.verticalHeader().setDefaultSectionSize(22)
        self._table.horizontalHeader().setDefaultSectionSize(72)
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)

        if self._headers:
            self._table.setHorizontalHeaderLabels(self._headers[:cols])
        else:
            self._table.horizontalHeader().setVisible(False)

        self._table.verticalHeader().setVisible(rows > 1)

        for r, row in enumerate(self._data):
            for c, val in enumerate(row):
                item = QTableWidgetItem(self._fmt(val))
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                if not self._editable:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(r, c, item)

        visible_rows = min(rows, 12)
        row_h = self._table.verticalHeader().defaultSectionSize()
        header_h = self._table.horizontalHeader().height() if self._headers else 0
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

    def _on_cell_changed(self, row: int, col: int):
        text = self._table.item(row, col).text().strip()
        try:
            val = float(text)
        except ValueError:
            val = text if text else None
        self._data[row][col] = val
        self.matrix_edited.emit(self._data)

    def _copy_tsv(self):
        from PyQt6.QtWidgets import QApplication
        lines = ["\t".join(self._fmt(v) for v in row) for row in self._data]
        QApplication.clipboard().setText("\n".join(lines))

    def _emit_workspace(self):
        self.send_to_workspace.emit(self._data, self._label)

    def get_data(self) -> List[List]:
        return [list(row) for row in self._data]

    # ── style ────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet("""
            #matrixWidget {
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
