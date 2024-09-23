import sys
from PySide6.QtWidgets import QApplication, QMainWindow

def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Test Window")
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
