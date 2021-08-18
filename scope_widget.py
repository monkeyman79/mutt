import numpy as np
import sys

from enum import IntEnum

import moderngl
from PyQt5 import QtCore, QtOpenGL, QtWidgets

from audio_source import AudioSource

_SIGNAL_VERTEX_SHADER = '''
    #version 330

    uniform int Count;
    uniform vec2 MainRect;

    in int in_vert;

    void main() {
        float x = (2. * gl_VertexID / Count - 1.);
        float y = (in_vert / 32768.);
        gl_Position = vec4(x * MainRect[0],
                            y * MainRect[1],
                            0.0, 1.0);
    }
'''

_BASIC_FRAGMENT_SHADER = '''
    #version 330

    uniform vec4 Color;

    out vec4 f_color;

    void main() {
        f_color = Color;
    }
'''

_BASIC_VERTEX_SHADER = '''
    #version 330
    out int inst;
    void main() {
        inst = gl_InstanceID;
    }
'''

_VLINE_GEOMETRY_SHADER = '''
    #version 330
    layout (points) in;
    layout (line_strip, max_vertices = 2) out;
    uniform int Y;
    uniform int Segments;
    uniform vec2 MainRect;
    void main() {
        float y = (Y / 32768.);
        float x1 = float(4 * gl_PrimitiveIDIn) / Segments - 1;
        float x2 = float(4 * gl_PrimitiveIDIn + 2) / Segments - 1;
        gl_Position = vec4(MainRect[0] * x1, MainRect[1] * y, 0, 1);
        EmitVertex();
        gl_Position = vec4(MainRect[0] * x2, MainRect[1] * y, 0, 1);
        EmitVertex();
        EndPrimitive();
    }
'''

_GRID_GEOMETRY_SHADER = '''
    #version 330
    layout (points) in;
    layout (line_strip, max_vertices = 2) out;
    in int inst[1];
    uniform vec2 MainRect;
    uniform int HCount;
    uniform int VCount;
    uniform int Segments;
    void vline(float n, float m) {
        float x = 0.;
        if (HCount != 0)
            x = 2 * n / HCount - 1;
        float y1 = 4 * m / Segments - 1;
        float y2 = (4 * m + 2) / Segments - 1;
        gl_Position = vec4(MainRect[0] * x, MainRect[1] * y1, 0, 1);
        EmitVertex();
        gl_Position = vec4(MainRect[0] * x, MainRect[1] * y2, 0, 1);
        EmitVertex();
        EndPrimitive();
    }
    void hline(float n, float m) {
        float y = 0;
        if (VCount != 0)
            y = 2 * n / VCount - 1;
        float x1 = 4 * m / Segments - 1;
        float x2 = (4 * m + 2) / Segments - 1;
        gl_Position = vec4(MainRect[0] * x1, MainRect[1] * y, 0, 1);
        EmitVertex();
        gl_Position = vec4(MainRect[0] * x2, MainRect[1] * y, 0, 1);
        EmitVertex();
        EndPrimitive();
    }
    void main() {
        if (inst[0] <= HCount)
            vline(inst[0], gl_PrimitiveIDIn);
        else
            hline(inst[0] - HCount - 1, gl_PrimitiveIDIn);
    }
'''

_TEXTURE_VERTEX_SHADER = '''
    #version 330

    uniform vec2 MainRect;
    uniform int TexOffset;

    in vec2 in_vert;
    in vec2 in_texcoord;

    out vec2 v_texcoord;

    void main() {
        v_texcoord = in_texcoord - vec2(0, TexOffset / 1024.);
        gl_Position = vec4(in_vert[0] * MainRect[0],
                           in_vert[1] * MainRect[1],
                           0.0, 1.0);
    }
'''

_TEXTURE_FRAGMENT_SHADER = '''
    #version 330

    uniform sampler2D Texture;

    in vec2 v_texcoord;
    out vec4 f_color;

    vec4 palette(float v) {
        return vec4(v*v, v*v*v, v*(1-v)+v*v*v, 1.);
    }

    void main() {
        f_color = palette(texture(Texture, v_texcoord)[0]);
    }
'''


class ScopeScene:
    MAX_VERTEX_COUNT = 4410
    VERTEX_WIDTH = 2
    DEFAULT_SIGNAL_COLOR = (0.0, 1.0, 1.0, 1.0)
    DEFAULT_TRIGGER_COLOR = (0.6, 0.3, 0.3, 0.5)
    GRID_COLOR = (0.4, 0.4, 0.4, 1.0)
    GRID_COLOR2 = (0.6, 0.6, 0.6, 1.0)
    MARGIN = 0.02
    HGRID_DIV = 10
    VGRID_DIV = 8
    GRID_SEGMENTS = 301
    TRIGGER_SEGMENTS = 65
    MAIN_RECT = (1-MARGIN, 1-MARGIN)
    FFT_WIDTH = 1024
    FFT_HEIGHT = 1024

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.signal_prog = self.ctx.program(
            vertex_shader=_SIGNAL_VERTEX_SHADER,
            fragment_shader=_BASIC_FRAGMENT_SHADER)
        self.vbo = ctx.buffer(
                reserve=self.MAX_VERTEX_COUNT * self.VERTEX_WIDTH)
        self.vao = ctx.vertex_array(
                self.signal_prog, [(self.vbo, 'i2', 'in_vert')])
        self.signal_prog['MainRect'] = self.MAIN_RECT

        self.grid_prog = self.ctx.program(
            vertex_shader=_BASIC_VERTEX_SHADER,
            geometry_shader=_GRID_GEOMETRY_SHADER,
            fragment_shader=_BASIC_FRAGMENT_SHADER)
        self.grid_vao = ctx.vertex_array(self.grid_prog, [])
        self.grid_prog['MainRect'] = self.MAIN_RECT

        self.vline_prog = self.ctx.program(
            vertex_shader=_BASIC_VERTEX_SHADER,
            geometry_shader=_VLINE_GEOMETRY_SHADER,
            fragment_shader=_BASIC_FRAGMENT_SHADER)
        self.vline_vao = ctx.vertex_array(self.vline_prog, [])
        self.vline_prog['MainRect'] = self.MAIN_RECT

        self.fft_prog = self.ctx.program(
            vertex_shader=_TEXTURE_VERTEX_SHADER,
            fragment_shader=_TEXTURE_FRAGMENT_SHADER)
        self.fft_vbo = ctx.buffer(np.array([
            -1.0, -1.0,  0, 0,  # lower left
            -1.0,  1.0,  0, 1,  # upper left
             1.0, -1.0,  1, 0,  # lower right
             1.0,  1.0,  1, 1,  # upper right
        ], dtype=np.float32))
        self.fft_vao = ctx.vertex_array(self.fft_prog, 
               [(self.fft_vbo, '2f4 2f4', 'in_vert', 'in_texcoord')])
        self.fft_prog['MainRect'] = self.MAIN_RECT
        self.fft_texture = self.ctx.texture(
                (self.FFT_WIDTH, self.FFT_HEIGHT), 1, dtype='f4')
        # self.fft_texture.filter = moderngl.NEAREST, moderngl.NEAREST
        self.fft_texture.swizzle = 'R001'
        self.fft_buffer = ctx.buffer(
                reserve=self.FFT_WIDTH * self.FFT_HEIGHT * 4)
        self.fft_pos = 0

    def clear(self, color=(0, 0, 0, 0)):
        self.ctx.clear(*color)

    def plot_fft(self, data):
        self.fft_buffer.write(data, offset=(
                self.FFT_HEIGHT - self.fft_pos - 1) * self.FFT_WIDTH * 4)
        self.fft_texture.write(self.fft_buffer)
        self.fft_texture.use(location=0)
        self.fft_prog['TexOffset'] = self.fft_pos
        self.fft_vao.render(moderngl.TRIANGLE_STRIP)
        self.fft_pos = (self.fft_pos + 1) % self.FFT_HEIGHT

    def plot_signal(self, points, count, color = DEFAULT_SIGNAL_COLOR):
        self.vbo.orphan()
        self.vbo.write(points)
        self.ctx.line_width = 1.0
        self.signal_prog['Color'] = color
        self.signal_prog['Count'] = count
        self.vao.render(moderngl.LINE_STRIP, count)

    def _plot_grid_frag(self, hcount, vcount, segments, color, ticks=False):
        self.grid_prog['HCount'] = hcount
        self.grid_prog['VCount'] = vcount
        self.grid_prog['Color'] = color
        self.grid_prog['Segments'] = segments
        segs = segments // 2 + 1 if not ticks else 1
        self.grid_vao.render(moderngl.POINTS, vertices=segs,
                             instances=hcount + vcount + 2)

    def plot_grid(self):
        self._plot_grid_frag(self.HGRID_DIV, self.VGRID_DIV,
                             self.GRID_SEGMENTS, self.GRID_COLOR)
        self._plot_grid_frag(0, 0, self.GRID_SEGMENTS, self.GRID_COLOR2)
        self._plot_grid_frag(self.HGRID_DIV*5, self.VGRID_DIV*5,
                             200, self.GRID_COLOR2, ticks=1)

    def plot_frame(self):
        self._plot_grid_frag(1, 1, 1, self.GRID_COLOR2)

    def draw_vline(self, pos: int, segments = TRIGGER_SEGMENTS,
                   color = DEFAULT_TRIGGER_COLOR):
        self.vline_prog['Y'] = pos
        self.vline_prog['Color'] = color
        self.vline_prog['Segments'] = segments
        self.vline_vao.render(moderngl.POINTS, vertices=segments // 2+1)

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


class TriggerMode(IntEnum):
    Off = 0
    Stop = 1
    Normal = 2
    Auto = 3
    Single = 4

class ScopeWidget(QtOpenGL.QGLWidget):
    AUDIO_BUFFER_SIZE = 8820
    AUTO_TRIGGER = 10
    triggerModeChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        fmt = QtOpenGL.QGLFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QtOpenGL.QGLFormat.CoreProfile)
        fmt.setSampleBuffers(True)
        self.scene: ScopeScene = None
        self.framebuffer: moderngl.Framebuffer = None
        self.ctx: moderngl.Context = None
        self.audio_source: AudioSource = None
        self._pan_tool = PanTool()
        self._trigger_level = 0
        self._auto_count = 0
        self.display_data = np.zeros(ScopeScene.MAX_VERTEX_COUNT, np.int16)
        self.display_count = ScopeScene.MAX_VERTEX_COUNT
        self.audio_data = np.zeros(self.AUDIO_BUFFER_SIZE, np.int16)
        self._trigger_mode = TriggerMode.Off
        self._force_trigger = False

        super(ScopeWidget, self).__init__(fmt, parent,
                                          size=QtCore.QSize(512, 512))

    def connectAudio(self, audio_source: AudioSource):
        self.audio_source = audio_source
        self.audio_source.connectDataReady(self.updateAudioData)

    def detectTriggerEdge(self, initial, arrange, first: bool = False) -> int:
        cond = lambda x: x > self._trigger_level

        cinitial = cond(initial)

        if first:
            crange = cond(arrange)
            sidx = 0
            if cinitial:
                # was 'above' edge initially - find first element 'below'
                sidx = np.argmin(crange)
                # sidx will be 0 if either first element is below or none
                if sidx == 0 and crange[0]:
                    return -1

            # find first element 'above'
            idx = np.argmax(crange[sidx:])
            # idx will be 0 if either first element is above or no elements
            if idx == 0 and not crange[sidx]:
                return -1
            return sidx + idx
        else:
            # Reverse the array while checking condition
            crrange = cond(arrange[::-1])
            sidx = 0
            if not crrange[0]:
                # last element is 'below', find last 'above'
                sidx = np.argmax(crrange)
                if sidx == 0:
                    # all elements 'below'
                    return -1
            
            # now find last element 'below'
            idx = np.argmin(crrange[sidx:])
            if idx == 0:
                # no 'raising' edge in array and first element is above
                if not cinitial:
                    return 0
                return -1
            # idx+sidx is index of last element before raising edge in the
            # reversed array
            return len(arrange) - (idx + sidx) + 1

    def processTriggering(self):

        chunk_size = self.audio_source.CHUNK_SIZE
        max_screen = ScopeScene.MAX_VERTEX_COUNT
        half_max_screen = max_screen // 2
        audio_data = self.audio_data

        # Detect trigger with delay to display half of screen after
        scan_start = (self.AUDIO_BUFFER_SIZE
                      - half_max_screen - chunk_size - 1)

        if self._trigger_mode == TriggerMode.Off or self._force_trigger:
            self._force_trigger = False
            trigger_idx = chunk_size
        elif self._trigger_mode == TriggerMode.Stop:
            trigger_idx = -1
        else:
            first = (self._trigger_mode == TriggerMode.Single)
            trigger_idx = self.detectTriggerEdge(
                    audio_data[scan_start-1], 
                    audio_data[scan_start:scan_start+chunk_size], first)
            if self._trigger_mode == TriggerMode.Auto:
                if trigger_idx == -1:
                    self._auto_count += 1
                    if self._auto_count >= self.AUTO_TRIGGER:
                        self._auto_count = 0
                        trigger_idx = chunk_size
                else:
                    self._auto_count = 0

        if trigger_idx != -1:
            if self._trigger_mode == TriggerMode.Single:
                self._trigger_mode = TriggerMode.Stop
                self.triggerModeChanged.emit()
            copy_start = scan_start + trigger_idx - half_max_screen
            self.display_data[:] = audio_data[
                    copy_start:copy_start + max_screen]

    def updateAudioData(self):
        chunk_size = self.audio_source.CHUNK_SIZE
        audio_data = self.audio_data

        data_in = self.audio_source.getBuffer()
        # Move data in audio buffer and append new chunk
        audio_data[0:-chunk_size] = audio_data[chunk_size:]
        audio_data[-chunk_size:] = data_in
        self.processTriggering()
        self.update()

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.framebuffer = self.ctx.detect_framebuffer()
        self.scene = ScopeScene(self.ctx)

    def paintGL(self):
        self.framebuffer.use()
        self.scene.clear()
        fft = np.maximum((np.log(np.abs(np.fft.rfft(self.audio_data[-4095:])) + 1.).astype('f4') - 11) / 4, 0.)
        self.scene.plot_fft(fft[:1023])
        self.scene.plot_grid()
        self.scene.draw_vline(self._trigger_level)
        plot_start = (ScopeScene.MAX_VERTEX_COUNT - self.display_count) // 2
        self.scene.plot_signal(self.display_data[plot_start:plot_start+self.display_count], 
                               self.display_count)
        self.scene.plot_frame()

    def resizeGL(self, w, h):
        self.ctx.viewport = (0, 0, w, h)

    def closeEvent(self, evt):
        worker = self.audio_source
        if worker is not None:
            worker.data_ready.disconnect(self.updateAudioData)
        return super().closeEvent(evt)

    def forceTrigger(self):
        self._force_trigger = True

    @property
    def triggerMode(self):
        return self._trigger_mode

    @triggerMode.setter
    def triggerMode(self, mode: TriggerMode):
        self._auto_count = 0
        self._trigger_mode = mode

    @property
    def triggerLevel(self):
        return self._trigger_level

    @triggerLevel.setter
    def triggerLevel(self, level):
        self._trigger_level = level

    def mousePressEvent(self, evt):
        self._pan_tool.start_drag(evt.x() / self.width(),
                                  evt.y() / self.height())
        # self.scene.pan(self._pan_tool.value)
        self.update()

    def mouseMoveEvent(self, evt):
        self._pan_tool.dragging(evt.x() / self.width(),
                                evt.y() / self.height())
        # self.scene.pan(self._pan_tool.value)
        self.update()

    def mouseReleaseEvent(self, evt):
        self._pan_tool.stop_drag(evt.x() / self.width(),
                                 evt.y() / self.height())
        # self.scene.pan(self._pan_tool.value)
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
