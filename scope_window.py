import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

from scope_widget import TriggerMode
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
        if value > 0:
            level = int((value / 100.) ** 2 * 32767)
        elif value < 0:
            level = int(-((-value / 100.) ** 2 * 32768))
        else:
            level = 0
        self.ui.scopeWidget.triggerLevel = level
        self.ui.triggerLevelEdit.setText(str(level))

    def listenButtonClicked(self):
        if self.ui.listenButton.isChecked():
            result = self.ui.scopeWidget.audio_source.start_listening()
            if not result:
                self.ui.listenButton.setChecked(False)
        else:
            self.ui.scopeWidget.audio_source.stop_listening()

    def triggerModeChanged(self, mode: TriggerMode, checked, setmode=False):
        newModeButton = self.triggerModeButtons[mode]
        if checked:
            if self.activeTriggerModeButton != None:
                self.activeTriggerModeButton.setChecked(False)
            self.activeTriggerModeButton = newModeButton
            if setmode:
                self.ui.scopeWidget.triggerMode = mode
        # Don't allow unchecking
        newModeButton.setChecked(True)

    def updateTriggerMode(self):
        mode = self.ui.scopeWidget.triggerMode
        self.triggerModeChanged(mode, checked=True, setmode=False)

    def forceButtonClicked(self):
        self.ui.scopeWidget.forceTrigger()

    def initUi(self):
        self.ui = Ui_ScopeWindow()
        self.ui.setupUi(self)
        self.ui.triggerLevelDial.valueChanged.connect(
                self.triggerLevelDialChanged)
        self.ui.listenButton.clicked.connect(self.listenButtonClicked)
        self.triggerModeButtons = {
            TriggerMode.Off: self.ui.offToolButton,
            TriggerMode.Stop: self.ui.stopToolButton,
            TriggerMode.Normal: self.ui.normalToolButton,
            TriggerMode.Auto: self.ui.autoToolButton,
            TriggerMode.Single: self.ui.singleToolButton }
        for mode, button in self.triggerModeButtons.items():
            action = lambda checked, mode=mode: self.triggerModeChanged(
                    mode=mode, checked=checked, setmode=True)
            button.clicked.connect(action)
        self.ui.forceToolButton.clicked.connect(self.forceButtonClicked)
        self.ui.scopeWidget.triggerModeChanged.connect(self.updateTriggerMode)
        self.activeTriggerModeButton = None
        self.updateTriggerMode()

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
