from PyQt6.QtWidgets import QMessageBox


def show_message(parent, title, text, buttons=QMessageBox.StandardButton.Ok):
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(buttons)
    return msg.exec()


def show_warning(parent, title, text, buttons=QMessageBox.StandardButton.Ok):
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(buttons)
    return msg.exec()


def show_error(parent, title, text, buttons=QMessageBox.StandardButton.Ok):
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(buttons)
    return msg.exec()


def show_info(parent, title, text, buttons=QMessageBox.StandardButton.Ok):
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(buttons)
    return msg.exec()


def show_question(parent, title, text, buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No):
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(buttons)
    return msg.exec()


def show_confirm_delete(parent, title, text):
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )
    msg.setDefaultButton(QMessageBox.StandardButton.No)
    return msg.exec()


def show_data_clear_options(parent):
    msg = QMessageBox(parent)
    msg.setWindowTitle("Clear Data")
    current = getattr(parent, 'current_table', '')
    msg.setText(f"Delete table '{current}' from database?")
    msg.setIcon(QMessageBox.Icon.NoIcon)
    msg.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
    )
    msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
    return msg.exec()


