import sys

import moderngl
from PyQt5 import QtCore, QtOpenGL, QtWidgets

from .scope_scene import ScopeSceneSignal


class ScopeWidget(QtOpenGL.QGLWidget):
    triggerModeChanged = QtCore.pyqtSignal()
    audioOverflowSignal = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSampleBuffers(True)
        self.ctx: moderngl.Context = None
        self.scene = ScopeSceneSignal()
        super(ScopeWidget, self).__init__(fmt, parent,
                                          size=QtCore.QSize(512, 512))

        self.scene.connect_update(self.update_handle)
        self.scene.connect_trigger_mode_changed(self.trigger_change_handle)
        self.scene.connect_input_overflow(self.input_overflow_handle)

    def update_handle(self, scene):
        self.update()

    def trigger_change_handle(self, scene):
        self.triggerModeChanged.emit()

    def input_overflow_handle(self, scene):
        self.audioOverflowSignal.emit()

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.scene.initialize(self.ctx)

    def paintGL(self):
        self.scene.paint()

    def resizeGL(self, w, h):
        self.ctx.viewport = (0, 0, w, h)

    def closeEvent(self, evt):
        worker = self.scene.audio_source
        if worker is not None:
            worker.connect_data_ready(None)
        return super().closeEvent(evt)


if __name__ == "__main__":
    from ..audio import PyAudioSource
    app = QtWidgets.QApplication(sys.argv)
    widget = ScopeWidget()
    audio_source = PyAudioSource()
    widget.show()
    widget.scene.connect_audio(audio_source)
    audio_source.start()
    sys.exit(app.exec_())
