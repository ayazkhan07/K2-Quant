"""
Indicator Widget for K2 Quant Analysis

Displays technical indicators with basic parameter editing and emits
signals to apply/remove indicators.
"""

from typing import List, Dict, Any

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QCheckBox, QLabel,
                             QScrollArea, QFrame, QHBoxLayout, QFormLayout,
                             QSpinBox, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal

from k2_quant.utilities.services.technical_analysis_service import ta_service
from k2_quant.utilities.logger import k2_logger


class IndicatorWidget(QWidget):
    """Widget for displaying and toggling technical indicators"""

    # Signals
    indicator_toggled = pyqtSignal(str, bool)  # legacy: indicator_name, enabled
    indicator_applied = pyqtSignal(str, dict)        # name, params
    indicator_removed = pyqtSignal(str)              # name
    indicator_params_changed = pyqtSignal(str, dict) # name, params
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.indicator_checkboxes = {}
        self.param_inputs = {}
        self.current_name = None
        self.init_ui()
        self.populate_indicators()
    
    def init_ui(self):
        """Initialize the UI"""
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(5)
        self.setLayout(self.layout)

        # Parameter editor area
        self.param_form = QFormLayout()
        self.apply_btn = QPushButton("Apply")
        self.remove_btn = QPushButton("Remove")
        self.apply_btn.clicked.connect(self._apply)
        self.remove_btn.clicked.connect(self._remove)
        self.layout.addSpacing(6)
        self.layout.addWidget(QLabel("Parameters"))
        self.layout.addLayout(self.param_form)
        action_row = QHBoxLayout()
        action_row.addWidget(self.apply_btn)
        action_row.addWidget(self.remove_btn)
        self.layout.addLayout(action_row)
    
    def populate_indicators(self):
        """Populate widget with all available indicators"""
        # Get all indicators from service
        indicators = ta_service.get_all_indicators()
        
        if not indicators:
            placeholder = QLabel("No indicators available")
            placeholder.setStyleSheet("color: #666; font-size: 12px;")
            self.layout.addWidget(placeholder)
            return
        
        # Group indicators by first letter for better organization
        grouped = {}
        for indicator in indicators:
            first_letter = indicator[0]
            if first_letter not in grouped:
                grouped[first_letter] = []
            grouped[first_letter].append(indicator)
        
        # Create checkboxes for each indicator
        for letter in sorted(grouped.keys()):
            # Add letter separator
            separator = QLabel(letter)
            separator.setStyleSheet("""
                QLabel {
                    color: #666;
                    font-size: 10px;
                    font-weight: bold;
                    padding: 5px 0px 2px 0px;
                    border-bottom: 1px solid #2a2a2a;
                    margin-top: 5px;
                }
            """)
            self.layout.addWidget(separator)
            
            # Add indicators for this letter
            for indicator_name in grouped[letter]:
                checkbox = self.create_indicator_checkbox(indicator_name)
                self.layout.addWidget(checkbox)
                self.indicator_checkboxes[indicator_name] = checkbox
        
        # Add stretch at the end
        self.layout.addStretch()
        
        k2_logger.info(f"Populated {len(indicators)} indicators", "INDICATOR_WIDGET")
    
    def create_indicator_checkbox(self, indicator_name: str) -> QWidget:
        """Create checkbox for an indicator"""
        # Get indicator info
        info = ta_service.get_indicator_info(indicator_name)
        
        # Create container
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 2, 0, 2)
        container.setLayout(layout)
        
        # Create checkbox
        checkbox = QCheckBox(indicator_name)
        checkbox.setStyleSheet("""
            QCheckBox {
                color: #ccc;
                font-size: 12px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                background-color: #1a1a1a;
                border: 1px solid #3a3a3a;
                border-radius: 2px;
            }
            QCheckBox::indicator:checked {
                background-color: #4a4;
                border-color: #4a4;
            }
            QCheckBox:hover {
                color: #fff;
            }
        """)
        
        # Connect signal
        checkbox.stateChanged.connect(
            lambda state, name=indicator_name: self.on_indicator_toggled(name, state)
        )
        checkbox.clicked.connect(lambda _=None, name=indicator_name: self._select_for_params(name))
        
        layout.addWidget(checkbox)
        
        # Add pane indicator
        if info and info.pane == 'separate':
            pane_label = QLabel("◧")  # Separate pane indicator
            pane_label.setStyleSheet("""
                QLabel {
                    color: #666;
                    font-size: 10px;
                    padding: 0px 5px;
                }
            """)
            pane_label.setToolTip("Displays in separate pane")
            layout.addWidget(pane_label)
        
        layout.addStretch()
        
        # Set tooltip with description
        if info:
            checkbox.setToolTip(f"{info.full_name}\n{info.description}")
        
        return container
    
    def on_indicator_toggled(self, indicator_name: str, state: int):
        """Handle indicator toggle"""
        enabled = state == 2  # Qt.CheckState.Checked
        self.indicator_toggled.emit(indicator_name, enabled)
        
        # Log the action
        k2_logger.info(
            f"Indicator {'enabled' if enabled else 'disabled'}: {indicator_name}",
            "INDICATOR_WIDGET"
        )

    def _select_for_params(self, indicator_name: str):
        self.current_name = indicator_name
        self.param_inputs.clear()
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        info = ta_service.get_indicator_info(indicator_name)
        if not info:
            return
        for param, default in (info.parameters or {}).items():
            spin = QSpinBox()
            spin.setRange(1, 1000)
            spin.setValue(int(default))
            spin.valueChanged.connect(lambda _=None, n=indicator_name: self._emit_params_changed(n))
            self.param_inputs[param] = spin
            self.param_form.addRow(QLabel(param), spin)

    def _collect_params(self) -> dict:
        return {k: w.value() for k, w in self.param_inputs.items()}

    def _apply(self):
        if self.current_name:
            self.indicator_applied.emit(self.current_name, self._collect_params())

    def _remove(self):
        if self.current_name:
            self.indicator_removed.emit(self.current_name)

    def _emit_params_changed(self, name: str):
        self.indicator_params_changed.emit(name, self._collect_params())
    
    def set_indicator_enabled(self, indicator_name: str, enabled: bool):
        """Programmatically set indicator state"""
        if indicator_name in self.indicator_checkboxes:
            checkbox = self.indicator_checkboxes[indicator_name].findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(enabled)
    
    def get_enabled_indicators(self) -> List[str]:
        """Get list of enabled indicators"""
        enabled = []
        for name, container in self.indicator_checkboxes.items():
            checkbox = container.findChild(QCheckBox)
            if checkbox and checkbox.isChecked():
                enabled.append(name)
        return enabled
    
    def disable_all(self):
        """Disable all indicators"""
        for container in self.indicator_checkboxes.values():
            checkbox = container.findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(False)
    
    def filter_indicators(self, filter_text: str):
        """Filter displayed indicators"""
        filter_lower = filter_text.lower()
        
        for name, container in self.indicator_checkboxes.items():
            # Show/hide based on filter
            if filter_lower in name.lower():
                container.setVisible(True)
            else:
                container.setVisible(False)