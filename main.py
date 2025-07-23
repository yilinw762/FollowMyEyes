import sys
from PyQt5.QtWidgets import QApplication
from UI.dashboard import EyeTrackerApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EyeTrackerApp()
    window.show()
    sys.exit(app.exec_())
