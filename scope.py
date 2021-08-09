import numpy as np
import sys

import pyaudio
import moderngl
from PyQt5 import QtCore, QtOpenGL, QtWidgets


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


class AudioInputWorker(QtCore.QObject):
    FX = np.linspace(-1.0, 1.0, 882)
    CHUNK_SIZE = 882
    data_ready = QtCore.pyqtSignal()

    def __init__(self, audio):
        super().__init__()
        self.next_buffer = 0
        self.buffers = np.zeros((2, self.CHUNK_SIZE), np.int16)
        self.stream = audio.open(44100, 1, pyaudio.paInt16, input=True,
                                 frames_per_buffer=self.CHUNK_SIZE,
                                 stream_callback=self.stream_callback)

    def stream_callback(self, in_data, frame_count, time_info, status):
        self.buffers[self.next_buffer][:] = \
            np.frombuffer(in_data, dtype=np.int16)
        self.next_buffer = 1 - self.next_buffer
        self.data_ready.emit()
        return (None, pyaudio.paContinue)

    def start(self):
        self.stream.start_stream()

    def stop(self):
        self.stream.stop_stream()

    def get_buffer(self) -> np.array:
        return self.buffers[1 - self.next_buffer]


class ScopeWidget(QtOpenGL.QGLWidget):
    def __init__(self, audio, parent=None):
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
        self.audio_worker = AudioInputWorker(audio)
        self.audio_worker.data_ready.connect(self.updateAudioData)

    def updateAudioData(self):
        chunk_size = self.audio_worker.CHUNK_SIZE
        self.audio_data[0:-chunk_size] = self.audio_data[chunk_size:]
        self.audio_data[-chunk_size:] = self.audio_worker.get_buffer()
        self.update()

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.framebuffer = self.ctx.detect_framebuffer()
        self.scene = ScopeScene(self.ctx)
        self.audio_worker.start()

    def paintGL(self):
        self.framebuffer.use()
        self.scene.clear()
        self.scene.plot(self.audio_data)

    def resizeGL(self, w, h):
        self.ctx.viewport = (0, 0, w, h)

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


audio = pyaudio.PyAudio()
app = QtWidgets.QApplication(sys.argv)
widget = ScopeWidget(audio=audio)
widget.show()
sys.exit(app.exec_())
