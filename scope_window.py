import sys
import os

from typing import Optional
from PyQt5.QtWidgets import QApplication, QMainWindow, QToolButton

from scope_widget import TriggerEdge, TriggerMode
from scope_window_ui import Ui_ScopeWindow
from audio_source import PyAudioSource


class ScopeWindow(QMainWindow):
    def __init__(self, audio=None):
        super().__init__()
        self.activeTriggerModeButton: Optional[QToolButton]
        self.activeTriggerEdgeButton: Optional[QToolButton]
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

    def deadbandDialChanged(self):
        value = self.ui.deadbandDial.value()
        deadband = int((value / 100.) ** 2 * 65535)
        self.ui.scopeWidget.deadband = deadband
        self.ui.deadbandLineEdit.setText(str(deadband))

    def listenButtonClicked(self):
        if self.ui.listenButton.isChecked():
            result = self.ui.scopeWidget.audio_source.start_listening()
            if not result:
                self.ui.listenButton.setChecked(False)
        else:
            self.ui.scopeWidget.audio_source.stop_listening()

    def triggerEdgeChanged(self, edge: TriggerEdge, checked, setedge):
        newEdgeButton = self.triggerEdgeButtons[edge]
        if checked:
            if self.activeTriggerEdgeButton is not None:
                self.activeTriggerEdgeButton.setChecked(False)
            self.activeTriggerEdgeButton = newEdgeButton
            if setedge:
                self.ui.scopeWidget.triggerEdge = edge
        newEdgeButton.setChecked(True)

    def updateTriggerEdge(self):
        edge = self.ui.scopeWidget.triggerEdge
        self.triggerEdgeChanged(edge, True, False)

    def triggerModeChanged(self, mode: TriggerMode, checked, setmode=False):
        newModeButton = self.triggerModeButtons[mode]
        if checked:
            if self.activeTriggerModeButton is not None:
                self.activeTriggerModeButton.setChecked(False)
            self.activeTriggerModeButton = newModeButton
            if setmode:
                self.ui.scopeWidget.triggerMode = mode
        # Don't allow unchecking
        newModeButton.setChecked(True)

    def updateTriggerMode(self):
        mode = self.ui.scopeWidget.triggerMode
        self.triggerModeChanged(mode, checked=True, setmode=False)

    def audioOverflowSlot(self):
        self.ui.statusbar.showMessage("Audio buffer overflow", 1000)

    def forceButtonClicked(self):
        self.ui.scopeWidget.forceTrigger()

    def initUi(self):
        self.ui = Ui_ScopeWindow()
        # pyuic generated code loads icons from relative paths
        cwd = os.getcwd()
        modulepath = os.path.dirname(os.path.abspath(__file__))
        os.chdir(modulepath)
        self.ui.setupUi(self)
        os.chdir(cwd)
        self.ui.triggerLevelDial.valueChanged.connect(
                self.triggerLevelDialChanged)
        self.ui.deadbandDial.valueChanged.connect(
                self.deadbandDialChanged)
        self.ui.listenButton.clicked.connect(self.listenButtonClicked)

        self.triggerModeButtons = {
            TriggerMode.Off: self.ui.offToolButton,
            TriggerMode.Stop: self.ui.stopToolButton,
            TriggerMode.Normal: self.ui.normalToolButton,
            TriggerMode.Auto: self.ui.autoToolButton,
            TriggerMode.Single: self.ui.singleToolButton}
        for mode, button in self.triggerModeButtons.items():
            button.clicked.connect(
                lambda checked, mode=mode: self.triggerModeChanged(
                    mode=mode, checked=checked, setmode=True))
        self.ui.forceToolButton.clicked.connect(self.forceButtonClicked)
        self.ui.scopeWidget.triggerModeChanged.connect(self.updateTriggerMode)
        self.activeTriggerModeButton = None
        self.updateTriggerMode()

        self.triggerEdgeButtons = {
            TriggerEdge.Positive: self.ui.raisingToolButton,
            TriggerEdge.Negative: self.ui.fallingToolButton}
        self.ui.raisingToolButton.clicked.connect(
            lambda checked:
            self.triggerEdgeChanged(TriggerEdge.Positive, checked, True))
        self.ui.fallingToolButton.clicked.connect(
            lambda checked:
            self.triggerEdgeChanged(TriggerEdge.Negative, checked, True))
        self.activeTriggerEdgeButton = None
        self.updateTriggerEdge()

        self.ui.scopeWidget.audioOverflowSignal.connect(self.audioOverflowSlot)


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
