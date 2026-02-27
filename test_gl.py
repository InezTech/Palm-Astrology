import sys
from PyQt6.QtWidgets import QApplication
import pyqtgraph.opengl as gl

app = QApplication(sys.argv)
w = gl.GLViewWidget()
w.show()
sys.exit(0)
