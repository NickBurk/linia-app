"""Qt background task with cooperative cancellation and queued result delivery."""

import warnings
from threading import Event

from PySide6.QtCore import QThread, Signal

from lineament_analyzer.core.windowed import CancelledError, check_cancel


class Task(QThread):
    result = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(int)
    notices = Signal(str)

    def __init__(self, operation, parent=None):
        super().__init__(parent)
        self.operation = operation
        self.stop_event = Event()

    def cancel(self):
        self.stop_event.set()

    def run(self):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                check_cancel(self.stop_event.is_set)
                result = self.operation(self.progress.emit, self.stop_event.is_set)
                check_cancel(self.stop_event.is_set)
            messages = list(
                dict.fromkeys(
                    str(item.message) for item in caught if issubclass(item.category, UserWarning)
                )
            )
            if messages:
                self.notices.emit("\n".join(messages))
            self.result.emit(result)
        except CancelledError:
            self.cancelled.emit()
        except Exception as error:  # noqa: BLE001 - report worker failures to the GUI
            self.failed.emit(f"{type(error).__name__}: {error}")
