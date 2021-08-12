import sys
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QPushButton

from scope_window_ui import Ui_ScopeWindow
from audio_source import PyAudioSource

class ScopeWindow(QMainWindow):
    def __init__(self, audio=None):
        super().__init__()
        self.initUi()
        if audio is not None:
            self.ui.scopeWidget.connectAudio(audio)
    
    def triggerLevelDialChanged(self):
        value = self.ui.triggerLevelDial.value()
        self.ui.triggerLevelEdit.setText(str(value))
        self.ui.scopeWidget.setTriggerLevel(value * 32767 / 100)

    def listenButtonClicked(self):
        if self.ui.listenButton.isChecked():
            result = self.ui.scopeWidget.audio_source.start_listening()
            if not result:
                self.ui.listenButton.setChecked(False)
        else:
            self.ui.scopeWidget.audio_source.stop_listening()

    def initUi(self):
        self.ui = Ui_ScopeWindow()
        self.ui.setupUi(self)
        self.ui.triggerLevelDial.valueChanged.connect(
                self.triggerLevelDialChanged)
        self.ui.listenButton.clicked.connect(self.listenButtonClicked)

if __name__ == "__main__":
    audio_source = PyAudioSource(enable_listening=True)
    app = QApplication(sys.argv)
    window = ScopeWindow(audio_source)
    window.show()
    audio_source.start()
    result = app.exec()
    audio_source.stop_listening()
    audio_source.stop()
    audio_source.audio.terminate()
