import numpy as np
import sys

import moderngl
from PyQt5 import QtCore, QtOpenGL, QtWidgets

from audio_source import AudioSource

class ScopeScene:
    VERTEX_COUNT = 4096
    VERTEX_SIZE = 2
    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform vec2 Pan;
                uniform int Count;

                in int in_vert;

                void main() {
                    gl_Position = vec4((2. * gl_VertexID / Count - 1.),
                                       in_vert / 32767., 0.0, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec4 Color;

                out vec4 f_color;

                void main() {
                    f_color = Color;
                }
            ''',
        )

        self.vbo = ctx.buffer(reserve=self.VERTEX_COUNT * self.VERTEX_SIZE)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, 'i2', 'in_vert')])
        self.prog['Color'] = (0, 1, 1, 1)
        self.prog['Count'] = self.VERTEX_COUNT

    def pan(self, pos):
        pass
        # self.prog['Pan'].value = pos

    def clear(self, color=(0, 0, 0, 0)):
        self.ctx.clear(*color)

    def plot(self, points, type='line'):
        self.vbo.orphan()
        self.vbo.write(points)
        if type == 'line':
            self.ctx.line_width = 5.0
            self.vao.render(moderngl.LINE_STRIP, self.VERTEX_COUNT)
        if type == 'points':
            self.ctx.point_size = 3.0
            self.vao.render(moderngl.POINTS, self.VERTEX_COUNT)


class PanTool:
    def __init__(self):
        self.total_x = 0.0
        self.total_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.drag = False

    def start_drag(self, x, y):
        self.start_x = x
        self.start_y = y
        self.drag = True

    def dragging(self, x, y):
        if self.drag:
            self.delta_x = (x - self.start_x) * 2.0
            self.delta_y = (y - self.start_y) * 2.0

    def stop_drag(self, x, y):
        if self.drag:
            self.dragging(x, y)
            self.total_x -= self.delta_x
            self.total_y += self.delta_y
            self.delta_x = 0.0
            self.delta_y = 0.0
            self.drag = False

    @property
    def value(self):
        return (self.total_x - self.delta_x, self.total_y + self.delta_y)


class ScopeWidget(QtOpenGL.QGLWidget):
    def __init__(self, parent=None):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSampleBuffers(True)
        self.scene = None
        self.ctx = None
        self._pan_tool = PanTool()
        self.audio_data = np.zeros(ScopeScene.VERTEX_COUNT, np.int16)

        super(ScopeWidget, self).__init__(fmt, parent,
                                          size=QtCore.QSize(512, 512))

    def connectAudio(self, audio_source: AudioSource):
        self.audio_source = audio_source
        self.audio_source.connectDataReady(self.updateAudioData)

    def updateAudioData(self):
        chunk_size = self.audio_source.CHUNK_SIZE
        self.audio_data[0:-chunk_size] = self.audio_data[chunk_size:]
        self.audio_data[-chunk_size:] = self.audio_source.getBuffer()
        self.update()

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.framebuffer = self.ctx.detect_framebuffer()
        self.scene = ScopeScene(self.ctx)

    def paintGL(self):
        self.framebuffer.use()
        self.scene.clear()
        self.scene.plot(self.audio_data)

    def resizeGL(self, w, h):
        self.ctx.viewport = (0, 0, w, h)

    def closeEvent(self, evt):
        worker = self.audio_source
        if worker is not None:
            worker.data_ready.disconnect(self.updateAudioData)
        return super().closeEvent(evt)

    def mousePressEvent(self, evt):
        self._pan_tool.start_drag(evt.x() / self.width(),
                                  evt.y() / self.height())
        self.scene.pan(self._pan_tool.value)
        self.update()

    def mouseMoveEvent(self, evt):
        self._pan_tool.dragging(evt.x() / self.width(),
                                evt.y() / self.height())
        self.scene.pan(self._pan_tool.value)
        self.update()

    def mouseReleaseEvent(self, evt):
        self._pan_tool.stop_drag(evt.x() / self.width(),
                                 evt.y() / self.height())
        self.scene.pan(self._pan_tool.value)
        self.update()

if __name__ == "__main__":
    from audio_source import PyAudioSource
    app = QtWidgets.QApplication(sys.argv)
    widget = ScopeWidget()
    audio_source = PyAudioSource()
    widget.show()
    widget.connectAudio(audio_source)
    audio_source.start()
    sys.exit(app.exec_())
