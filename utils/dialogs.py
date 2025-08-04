"""
Custom Dialog Utilities

Provides icon-free message box functions for the application.
"""

from PyQt6.QtWidgets import QMessageBox


def show_message(parent, title, text, buttons=QMessageBox.StandardButton.Ok):
    """Show a message box without any icon"""
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(buttons)
    return msg.exec()


def show_warning(parent, title, text, buttons=QMessageBox.StandardButton.Ok):
    """Show a warning message without icon"""
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(buttons)
    return msg.exec()


def show_error(parent, title, text, buttons=QMessageBox.StandardButton.Ok):
    """Show an error message without icon"""
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(buttons)
    return msg.exec()


def show_info(parent, title, text, buttons=QMessageBox.StandardButton.Ok):
    """Show an information message without icon"""
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(buttons)
    return msg.exec()


def show_question(parent, title, text, buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No):
    """Show a question dialog without icon"""
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(buttons)
    return msg.exec()


def show_confirm_delete(parent, title, text):
    """Show a confirmation dialog for delete operations without icon"""
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(
        QMessageBox.StandardButton.Yes | 
        QMessageBox.StandardButton.No
    )
    # Make No the default button for safety
    msg.setDefaultButton(QMessageBox.StandardButton.No)
    return msg.exec()


def show_data_clear_options(parent):
    """Show a dialog with clear data options without icon"""
    msg = QMessageBox(parent)
    msg.setWindowTitle("Clear Data")
    msg.setText(f"Delete table '{parent.current_table}' from database?")
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(
        QMessageBox.StandardButton.Yes | 
        QMessageBox.StandardButton.No | 
        QMessageBox.StandardButton.Cancel
    )
    msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
    return msg.exec()