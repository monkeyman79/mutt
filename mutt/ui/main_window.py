#! /bin/env python3

import sys
import math
from typing import Optional

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QToolButton

from .scope_scene import ScopeScene, TriggerEdge, TriggerMode
from .main_window_ui import Ui_ScopeWindow
from ..audio import AudioSource, PyAudioSource


class MUTTMainWindow(QMainWindow):
    LOG_DIAL_VALUES = [1, 2, 5]
    TRIGGER_LEVEL_CHANGE = 50

    def __init__(self, audio=None):
        super().__init__()
        self.activeTriggerModeButton: Optional[QToolButton]
        self.activeTriggerEdgeButton: Optional[QToolButton]
        self.prev_high_level_dial = 0
        self.prev_low_level_dial = 0
        self.prev_horiz_dial = 0
        self.initUi()
        if audio is not None:
            self.scene.connect_audio(audio)

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

    def showLevels(self):
        self.ui.highLevelEdit.setText(str(self.scene.highLevel))
        self.ui.lowLevelEdit.setText(str(self.scene.lowLevel))

    def highLevelDialChanged(self):
        value = self.ui.highLevelDial.value()
        diff = self.dialDiff(value, self.prev_high_level_dial)
        level = self.scene.highLevel
        modifiers = QApplication.keyboardModifiers()
        if (modifiers & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier:
            tick = 1
        else:
            tick = self.TRIGGER_LEVEL_CHANGE
        if ((modifiers & QtCore.Qt.ControlModifier)
                == QtCore.Qt.ControlModifier):
            tick = 5 * tick
        level += diff * tick
        self.scene.highLevel = level
        self.prev_high_level_dial = value
        self.showLevels()

    def lowLevelDialChanged(self):
        value = self.ui.lowLevelDial.value()
        diff = self.dialDiff(value, self.prev_low_level_dial)
        level = self.scene.lowLevel
        modifiers = QApplication.keyboardModifiers()
        if (modifiers & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier:
            tick = 1
        else:
            tick = self.TRIGGER_LEVEL_CHANGE
        if ((modifiers & QtCore.Qt.ControlModifier)
                == QtCore.Qt.ControlModifier):
            tick = 5 * tick
        level += diff * tick
        self.scene.lowLevel = level
        self.prev_low_level_dial = value
        self.showLevels()

    def showHorizScale(self):
        if self.scene.audio_source is not None:
            freq = self.scene.audio_source.FREQ
        else:
            freq = AudioSource.FREQ
        sample_count = self.scene.displayCount
        time_per_div = (1000000 * sample_count // freq + 9) // 10
        if time_per_div >= 1000:
            time_text = "{} ms".format(time_per_div // 1000)
        else:
            time_text = "{} us".format(time_per_div)
        self.ui.horizScaleLineEdit.setText(str(time_text))

    def horizScaleChanged(self):
        value = self.ui.horizScaleDial.value()
        divisor = self.LOG_DIAL_VALUES[value % 3] * 10 ** (value // 3)
        sample_count = int(round(ScopeScene.MAX_VERTEX_COUNT / divisor))
        self.scene.displayCount = sample_count
        self.showHorizScale()

    def showPulseScale(self):
        if self.scene.audio_source is not None:
            freq = self.scene.audio_source.FREQ
        else:
            freq = AudioSource.FREQ
        scale = self.scene.pulseImageScale
        screen_time = (1000000 * self.scene.PULSE_TEX_WIDTH /
                       (freq * scale * ScopeScene.HGRID_DIV))
        self.ui.pulseScaleEdit.setText(str(screen_time) + " us")

    def pulseScaleDialChanged(self):
        value = self.ui.pulseScaleDial.value()
        scale = 1 << value
        self.scene.pulseImageScale = scale
        self.showPulseScale()

    def showHorizPosition(self):
        if self.scene.audio_source is not None:
            freq = self.scene.audio_source.FREQ
        else:
            freq = AudioSource.FREQ
        position = self.scene.displayPosition
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
        div_count = (self.scene.displayCount
                     / ScopeScene.HGRID_DIV / 10)
        modifiers = QApplication.keyboardModifiers()
        if (modifiers & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier:
            # Change to samples per tick
            div_count /= 5
        # Round up
        div_count = int(math.ceil(div_count))
        # Update position
        position = self.scene.displayPosition + diff * div_count
        self.scene.displayPosition = position
        self.prev_horiz_dial = value
        self.showHorizPosition()

    def listenButtonClicked(self):
        if self.ui.listenButton.isChecked():
            result = self.scene.audio_source.start_listening()
            if not result:
                self.ui.listenButton.setChecked(False)
        else:
            self.scene.audio_source.stop_listening()

    def triggerEdgeChanged(self, edge: TriggerEdge, checked, setedge):
        newEdgeButton = self.triggerEdgeButtons[edge]
        if checked:
            if self.activeTriggerEdgeButton is not None:
                self.activeTriggerEdgeButton.setChecked(False)
            self.activeTriggerEdgeButton = newEdgeButton
            if setedge:
                self.scene.triggerEdge = edge
        newEdgeButton.setChecked(True)

    def updateTriggerEdge(self):
        edge = self.scene.triggerEdge
        self.triggerEdgeChanged(edge, True, False)

    def triggerModeChanged(self, mode: TriggerMode, checked, setmode=False):
        newModeButton = self.triggerModeButtons[mode]
        if checked:
            if self.activeTriggerModeButton is not None:
                self.activeTriggerModeButton.setChecked(False)
            self.activeTriggerModeButton = newModeButton
            if setmode:
                self.scene.triggerMode = mode
        # Don't allow unchecking
        newModeButton.setChecked(True)

    def updateTriggerMode(self):
        mode = self.scene.triggerMode
        self.triggerModeChanged(mode, checked=True, setmode=False)

    def audioOverflowSlot(self):
        self.ui.statusbar.showMessage("Audio buffer overflow", 1000)

    def forceButtonClicked(self):
        self.scene.forceTrigger()

    def bindBoolPropertyButton(self, button: QToolButton, obj, prop: property):
        button.setChecked(prop.__get__(obj))
        button.clicked.connect(lambda checked:  # type: ignore
                               prop.__set__(obj, checked))

    def initUi(self):
        self.ui = Ui_ScopeWindow()
        self.ui.setupUi(self)
        self.scene = self.ui.scopeWidget.scene
        self.ui.highLevelDial.valueChanged.connect(
                self.highLevelDialChanged)
        self.ui.lowLevelDial.valueChanged.connect(
                self.lowLevelDialChanged)
        self.ui.horizScaleDial.valueChanged.connect(
                self.horizScaleChanged)
        self.ui.horizPositionDial.valueChanged.connect(
                self.horizPositionChanged)
        self.ui.pulseScaleDial.valueChanged.connect(
                self.pulseScaleDialChanged)
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
                ScopeScene.displaySignal: self.ui.displaySignalButton,
                ScopeScene.displayRelay: self.ui.displayRelayButton,
                ScopeScene.displayFFTImage: self.ui.displayFFTImageButton,
                ScopeScene.displayFFTGraph: self.ui.displayFFTGraphButton,
                ScopeScene.displayTriggerLines: self.ui.displayTriggerButton,
                ScopeScene.displayGrid: self.ui.displayGridButton,
                ScopeScene.displayPulseImage: self.ui.displayPulseButton,
                ScopeScene.displayPulseGraph: self.ui.displayPulseGraphButton
                }.items():
            self.bindBoolPropertyButton(button, self.scene, prop)

        self.ui.scopeWidget.audioOverflowSignal.connect(self.audioOverflowSlot)

        self.showLevels()
        self.showHorizScale()
        self.showHorizPosition()
        self.showPulseScale()

        self.ui.horizScaleDial.setValue(6)
        self.ui.pulseScaleDial.setValue(5)


def Run():
    audio_source = PyAudioSource(enable_listening=True)
    app = QApplication(sys.argv)
    window = MUTTMainWindow(audio_source)
    window.show()
    audio_source.start()
    app.exec()
    audio_source.stop_listening()
    audio_source.stop()
    audio_source.audio.terminate()


if __name__ == "__main__":
    Run()
