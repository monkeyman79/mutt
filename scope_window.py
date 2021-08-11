import sys
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QPushButton

from scope_window_ui import Ui_ScopeWindow
from audio_source import PyAudioSource

class ScopeWindow(QMainWindow):
    def __init__(self, audio):
        super().__init__()
        self.ui = Ui_ScopeWindow()
        self.ui.setupUi(self)
        if audio is not None:
            self.ui.scopeWidget.connectAudio(audio)

if __name__ == "__main__":
    audio_source = PyAudioSource()
    app = QApplication(sys.argv)
    window = ScopeWindow(audio_source)
    window.show()
    audio_source.start()
    sys.exit(app.exec())
