"""Desktop application entry point."""

import sys

from PySide6.QtWidgets import QApplication

from lineament_analyzer.gui.window import MainWindow


def main() -> int:
    """Start the Qt event loop."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
