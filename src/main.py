import sys
from PyQt6.QtWidgets import QApplication
from src.views.main_window import PalmOracleApp

def main():
    app = QApplication(sys.argv)
    window = PalmOracleApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
