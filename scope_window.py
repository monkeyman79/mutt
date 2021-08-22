#! /bin/env python3

import sys
import os
import math
from typing import Optional

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QToolButton

from scope_widget import ScopeScene, ScopeWidget, TriggerEdge, TriggerMode
from scope_window_ui import Ui_ScopeWindow
from audio_source import AudioSource, PyAudioSource


class ScopeWindow(QMainWindow):
    LOG_DIAL_VALS = [1, 2, 5]
    TRIGGER_LEVEL_CHANGE = 50

    def __init__(self, audio=None):
        super().__init__()
        self.activeTriggerModeButton: Optional[QToolButton]
        self.activeTriggerEdgeButton: Optional[QToolButton]
        self.prev_trigger_dial = 0
        self.prev_deadband_dial = 0
        self.prev_horiz_dial = 0
        self.initUi()
        if audio is not None:
            self.ui.scopeWidget.connectAudio(audio)

    @staticmethod
    def dialDiff(value: int, prev: int):
        rel_diff = value - prev
        diff = rel_diff % 100
        if diff >= 50:
            diff = diff - 100
        # Dial value jumps from 0 to 98 and from 99 to 1
        if rel_diff > 0 and diff < 0:
            diff += 1
        elif rel_diff < 0 and diff > 0:
            diff -= 1
        return diff

    def showTriggerLevel(self):
        self.ui.triggerLevelEdit.setText(str(self.ui.scopeWidget.triggerLevel))

    def triggerLevelDialChanged(self):
        value = self.ui.triggerLevelDial.value()
        diff = self.dialDiff(value, self.prev_trigger_dial)
        level = self.ui.scopeWidget.triggerLevel
        modifiers = QApplication.keyboardModifiers()
        if (modifiers & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier:
            tick = 1
        else:
            tick = self.TRIGGER_LEVEL_CHANGE
        if ((modifiers & QtCore.Qt.ControlModifier)
                == QtCore.Qt.ControlModifier):
            tick = 5 * tick
        level += diff * tick
        if level > 32767:
            level = 32767
        elif level < -32768:
            level = -32768
        self.ui.scopeWidget.triggerLevel = level
        self.prev_trigger_dial = value
        self.showTriggerLevel()

    def showDeadband(self):
        self.ui.deadbandLineEdit.setText(str(self.ui.scopeWidget.deadband))

    def deadbandDialChanged(self):
        value = self.ui.deadbandDial.value()
        diff = self.dialDiff(value, self.prev_deadband_dial)
        deadband = self.ui.scopeWidget.deadband
        modifiers = QApplication.keyboardModifiers()
        if (modifiers & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier:
            tick = 1
        else:
            tick = self.TRIGGER_LEVEL_CHANGE
        if ((modifiers & QtCore.Qt.ControlModifier)
                == QtCore.Qt.ControlModifier):
            tick = 5 * tick
        deadband += diff * tick
        if deadband > 65535:
            deadband = 65535
        elif deadband < 0:
            deadband = 0
        self.ui.scopeWidget.deadband = deadband
        self.prev_deadband_dial = value
        self.showDeadband()

    def showHorizScale(self):
        if self.ui.scopeWidget.audio_source is not None:
            freq = self.ui.scopeWidget.audio_source.FREQ
        else:
            freq = AudioSource.FREQ
        sample_count = self.ui.scopeWidget.displayCount
        time_per_div = (1000000 * sample_count // freq + 9) // 10
        if time_per_div >= 1000:
            time_text = "{} ms".format(time_per_div // 1000)
        else:
            time_text = "{} us".format(time_per_div)
        self.ui.horizScaleLineEdit.setText(str(time_text))

    def horizScaleChanged(self):
        value = self.ui.horizScaleDial.value()
        divisor = self.LOG_DIAL_VALS[value % 3] * 10 ** (value // 3)
        sample_count = int(round(ScopeScene.MAX_VERTEX_COUNT / divisor))
        self.ui.scopeWidget.displayCount = sample_count
        self.showHorizScale()

    def showHorizPosition(self):
        if self.ui.scopeWidget.audio_source is not None:
            freq = self.ui.scopeWidget.audio_source.FREQ
        else:
            freq = AudioSource.FREQ
        position = self.ui.scopeWidget.displayPosition
        # Convert to milliseconds
        t_offset = 1000 * position / freq
        if t_offset == 0:
            display_str = "0 us"
        elif abs(t_offset) < 1:
            display_str = "{:+d} us".format(int(t_offset * 1000))
        else:
            display_str = "{:+.2f} ms".format(t_offset)
        self.ui.horizPositionLineEdit.setText(display_str)

    def horizPositionChanged(self):
        value = self.ui.horizPositionDial.value()
        diff = self.dialDiff(value, self.prev_horiz_dial)
        # Samples per horizontal division
        div_count = (self.ui.scopeWidget.displayCount
                     / ScopeScene.HGRID_DIV / 10)
        modifiers = QApplication.keyboardModifiers()
        if (modifiers & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier:
            # Change to samples per tick
            div_count /= 5
        # Round up
        div_count = int(math.ceil(div_count))
        # Update position
        position = self.ui.scopeWidget.displayPosition + diff * div_count
        self.ui.scopeWidget.displayPosition = position
        self.prev_horiz_dial = value
        self.showHorizPosition()

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

    def bindBoolPropertyButton(self, button: QToolButton, obj, prop: property):
        button.setChecked(prop.__get__(obj))
        button.clicked.connect(lambda checked:  # type: ignore
                               prop.__set__(obj, checked))

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
        self.ui.horizScaleDial.valueChanged.connect(
                self.horizScaleChanged)
        self.ui.horizScaleLineEdit.setText("10 ms")
        self.ui.horizPositionDial.valueChanged.connect(
                self.horizPositionChanged)
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

        for prop, button in {
                ScopeWidget.displaySignal: self.ui.displaySignalButton,
                ScopeWidget.displayRelay: self.ui.displayRelayButton,
                ScopeWidget.displayFFTImage: self.ui.displayFFTImageButton,
                ScopeWidget.displayFFTGraph: self.ui.displayFFTGraphButton,
                ScopeWidget.displayTriggerLines: self.ui.displayTriggerButton,
                ScopeWidget.displayGrid: self.ui.displayGridButton}.items():
            self.bindBoolPropertyButton(button, self.ui.scopeWidget, prop)

        self.ui.scopeWidget.audioOverflowSignal.connect(self.audioOverflowSlot)

        self.showTriggerLevel()
        self.showDeadband()
        self.showHorizScale()
        self.showHorizPosition()


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
