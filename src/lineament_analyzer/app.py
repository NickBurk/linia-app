"""Desktop application entry point."""

import os
import sys
import traceback
from pathlib import Path

from PySide6.QtWidgets import QApplication

from lineament_analyzer.gui.window import MainWindow


def main() -> int:
    """Start the Qt event loop."""
    app = QApplication(sys.argv)
    if "--self-test" in sys.argv or "--self-test-restore" in sys.argv:
        from lineament_analyzer.selftest import restore_in_new_process, run

        try:
            if "--self-test-restore" in sys.argv:
                return restore_in_new_process(Path(sys.argv[sys.argv.index("--self-test-restore") + 1]))
            return run()
        except Exception:
            if log := os.environ.get("LINEAMENT_SELFTEST_LOG"):
                with Path(log).open("a", encoding="utf-8") as stream:
                    stream.write(traceback.format_exc())
            raise
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
