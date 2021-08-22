import numpy as np
import sys

from enum import IntEnum
from typing import Callable, Tuple, cast

import moderngl
from PyQt5 import QtCore, QtOpenGL, QtWidgets

from audio_source import AudioSource

_SIGNAL_VERTEX_SHADER = '''
    #version 330

    uniform int Start;
    uniform int Scale;
    uniform vec2 MainRect;

    in int in_vert;

    void main() {
        float x = 2. * (Start + gl_VertexID) / (Scale - 1) - 1.;
        float y = in_vert / 32768.;
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

_HLINE_GEOMETRY_SHADER = '''
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

_VLINE_GEOMETRY_SHADER = '''
    #version 330
    layout (points) in;
    layout (line_strip, max_vertices = 2) out;
    uniform float X;
    uniform int Segments;
    uniform vec2 MainRect;
    void main() {
        float y1 = float(4 * gl_PrimitiveIDIn) / Segments - 1;
        float y2 = float(4 * gl_PrimitiveIDIn + 2) / Segments - 1;
        gl_Position = vec4(MainRect[0] * X, MainRect[1] * y1, 0, 1);
        EmitVertex();
        gl_Position = vec4(MainRect[0] * X, MainRect[1] * y2, 0, 1);
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
    MAX_VERTEX_COUNT = 6250
    VERTEX_WIDTH = 2
    GRID_COLOR = (0.4, 0.4, 0.4, 1.0)
    GRID_COLOR2 = (0.6, 0.6, 0.6, 1.0)
    MARGIN = 0.02
    HGRID_DIV = 10
    VGRID_DIV = 8
    GRID_SEGMENTS = 301
    TRIGGER_SEGMENTS = 33
    MAIN_RECT = (1-MARGIN, 1-MARGIN)
    FFT_WIDTH = 512
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

        self.hline_prog = self.ctx.program(
            vertex_shader=_BASIC_VERTEX_SHADER,
            geometry_shader=_HLINE_GEOMETRY_SHADER,
            fragment_shader=_BASIC_FRAGMENT_SHADER)
        self.hline_vao = ctx.vertex_array(self.hline_prog, [])
        self.hline_prog['MainRect'] = self.MAIN_RECT

        self.vline_prog = self.ctx.program(
            vertex_shader=_BASIC_VERTEX_SHADER,
            geometry_shader=_VLINE_GEOMETRY_SHADER,
            fragment_shader=_BASIC_FRAGMENT_SHADER)
        self.vline_vao = ctx.vertex_array(self.vline_prog, [])
        self.vline_prog['MainRect'] = self.MAIN_RECT

        self.fft_prog = self.ctx.program(
            vertex_shader=_TEXTURE_VERTEX_SHADER,
            fragment_shader=_TEXTURE_FRAGMENT_SHADER)
        self.fft_vbo = ctx.buffer(
            np.array([
                -1.0, -1.0,  0, 0,  # lower left
                -1.0,  1.0,  0, 1,  # upper left
                1.0, -1.0,  1, 0,  # lower right
                1.0,  1.0,  1, 1,  # upper right
            ], dtype=np.float32))
        self.fft_vao = ctx.vertex_array(
                self.fft_prog,
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

    def clear_fft(self):
        self.fft_buffer.clear()

    def plot_signal(self, points, count, scale, start, color):
        self.vbo.orphan()
        self.vbo.write(points)
        self.ctx.line_width = 1.0
        self.signal_prog['Color'] = color
        self.signal_prog['Scale'] = scale
        self.signal_prog['Start'] = start
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

    def draw_hline(self, pos: int, color: Tuple[int, int, int, int],
                   segments=TRIGGER_SEGMENTS):
        self.hline_prog['Y'] = pos
        self.hline_prog['Color'] = color
        self.hline_prog['Segments'] = segments
        self.hline_vao.render(moderngl.POINTS, vertices=segments // 2 + 1)

    def draw_vline(self, pos: float, color: Tuple[int, int, int, int],
                   segments=TRIGGER_SEGMENTS):
        self.vline_prog['X'] = pos
        self.vline_prog['Color'] = color
        self.vline_prog['Segments'] = segments
        self.vline_vao.render(moderngl.POINTS, vertices=segments // 2 + 1)


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


class TriggerEdge(IntEnum):
    Positive = 0
    Negative = 1


class ScopeWidget(QtOpenGL.QGLWidget):
    AUDIO_BUFFER_SIZE = ScopeScene.MAX_VERTEX_COUNT * 3
    SNAPSHOT_SIZE = ScopeScene.MAX_VERTEX_COUNT * 2
    AUTO_TRIGGER = 10
    FFT_CALC_SIZE = 4096
    TRIGGER_COLOR = (0.9, 0.5, 0.5, 0.8)
    TRIGGER_ARM_COLOR = (0.9, 0.5, 0.5, 0.8)
    SIGNAL_COLOR = (0.0, 1.0, 1.0, 1.0)
    FFT_GRAPH_COLOR = (0.8, 0.0, 0.3, 1.0)
    RELAY_GRAPH_COLOR = (0.8, 0.8, 0.0, 1.0)

    triggerModeChanged = QtCore.pyqtSignal()
    audioOverflowSignal = QtCore.pyqtSignal()

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
        self._deadband = 0
        self._deadband_level = 0
        self._auto_count = 0
        self.snapshot_data = np.zeros(self.SNAPSHOT_SIZE, np.int16)
        self.relay_snapshot_data = np.zeros(self.SNAPSHOT_SIZE, np.int8)
        self._display_sample_count = ScopeScene.MAX_VERTEX_COUNT
        self._display_position = 0
        self.audio_data = np.zeros(self.AUDIO_BUFFER_SIZE, np.int16)
        self.relay_data = -1 * np.ones(self.AUDIO_BUFFER_SIZE, np.int8)
        self._trigger_mode = TriggerMode.Normal
        self._trigger_edge = TriggerEdge.Positive
        self._force_trigger = False
        self._triggered = False
        self._display_fft_image = False
        self.fft_tex_log = True
        self._display_fft_graph = False
        self.fft_graph_log = False
        self._display_signal = True
        self._display_relay = False
        self._display_trigger_lines = True
        self._display_grid = True
        self.deadband = 500

        super(ScopeWidget, self).__init__(fmt, parent,
                                          size=QtCore.QSize(512, 512))

    def connectAudio(self, audio_source: AudioSource):
        self.audio_source = audio_source
        self.chunk_range = np.arange(audio_source.CHUNK_SIZE + 1)
        self.audio_source.connectDataReady(self.updateAudioData)

    def detectTriggerEdge(self, last_value: int, arrange: np.ndarray,
                          first: bool = False) -> int:

        if first:
            sidx: int = 0
            if last_value != -1:
                # was not armed initially - find first element 'below'
                sidx = cast(int, np.argmin(arrange))
                # sidx will be 0 if either first element is below or none
                if sidx == 0 and arrange[0] != -1:
                    return -1

            # find first element 'above'
            idx = np.argmax(arrange[sidx:])
            # idx will be 0 if either first element is above or no elements
            if arrange[sidx + idx] != 1:
                return -1
            return cast(int, sidx + idx)

        else:
            # Reverse the array while checking condition
            crrange = arrange[::-1]
            sidx = 0

            # Never armed
            if crrange[0] == 0:
                return -1

            if crrange[0] == -1:
                # last element is 'below', find last 'above'
                sidx = cast(int, np.argmax(crrange))
                if sidx == 0:
                    # all elements 'below'
                    return -1

            # now find last element 'below'
            idx = np.argmin(crrange[sidx:])
            if idx == 0:
                # no 'raising' edge in array and first element is above
                if last_value == -1:
                    return 0
                return -1

            # idx+sidx is index of last element before raising edge in the
            # reversed array
            return cast(int, len(arrange) - (idx + sidx) + 1)

    def processTriggering(self):

        chunk_size = self.audio_source.CHUNK_SIZE
        max_screen = self.SNAPSHOT_SIZE
        half_max_screen = max_screen // 2
        relay_data = self.relay_data

        # Detect trigger with delay to display half of screen after
        scan_start = (self.AUDIO_BUFFER_SIZE
                      - half_max_screen - chunk_size - 1)

        if self._trigger_mode == TriggerMode.Off or self._force_trigger:
            self._force_trigger = False
            self._triggered = False
            trigger_idx = chunk_size
        elif self._trigger_mode == TriggerMode.Stop:
            trigger_idx = -1
        else:
            first = (self._trigger_mode == TriggerMode.Single)
            trigger_idx = self.detectTriggerEdge(
                    relay_data[scan_start-1],
                    relay_data[scan_start:scan_start+chunk_size], first)
            if trigger_idx != -1:
                self._triggered = True
            if self._trigger_mode == TriggerMode.Auto:
                if trigger_idx == -1:
                    self._auto_count += 1
                    if self._auto_count >= self.AUTO_TRIGGER:
                        self._auto_count = 0
                        trigger_idx = chunk_size
                        self._triggered = False
                else:
                    self._auto_count = 0

        if trigger_idx != -1:
            if self._trigger_mode == TriggerMode.Single:
                self._trigger_mode = TriggerMode.Stop
                self.triggerModeChanged.emit()
            copy_start = scan_start + trigger_idx - half_max_screen
            self.snapshot_data[:] = self.audio_data[
                    copy_start:copy_start + max_screen]
            self.relay_snapshot_data[:] = self.relay_data[
                    copy_start:copy_start + max_screen]

    def updateDeadbandLevel(self):
        if self._trigger_edge == TriggerEdge.Positive:
            deadband_level = self._trigger_level - self._deadband
            if deadband_level < -32768:
                deadband_level = 32768
        else:
            deadband_level = self._trigger_level + self.deadband
            if deadband_level > 32767:
                deadband_level = 32767
        self._deadband_level = deadband_level

    def updateRelayData(self, data_in: np.ndarray):
        chunk_size = self.audio_source.CHUNK_SIZE
        relay_data = self.relay_data
        one = np.int8(1)

        # Put last relay condition at the beginning of buffer to process
        relay_chunk = np.zeros(chunk_size + 1, dtype=np.int8)
        relay_chunk[0] = relay_data[-1]

        if self._trigger_edge == TriggerEdge.Positive:
            trigger_op: Callable = np.greater
            arm_op: Callable = np.less
        else:
            trigger_op = np.less
            arm_op = np.greater

        # Put relay state for each sample in the buffer
        np.subtract(one * trigger_op(data_in, self._trigger_level),
                    one * arm_op(data_in, self._deadband_level),
                    out=relay_chunk[1:], dtype=np.int8)

        # https://stackoverflow.com/questions/68869535/numpy-accumulate-greater-operation
        # Process hysteresis on the chunk
        masked_indexes = np.where((relay_chunk != 0), self.chunk_range, 0)
        hyst_indexes = np.maximum.accumulate(masked_indexes)
        result = relay_chunk[hyst_indexes][1:]

        # Move data in buffer append new chunk
        relay_data[0:-chunk_size] = relay_data[chunk_size:]
        relay_data[-chunk_size:] = result
        return result

    def updateAudioData(self):
        chunk_size = self.audio_source.CHUNK_SIZE
        audio_data = self.audio_data

        data_in, overflow = self.audio_source.getBuffer()
        # Move data in audio buffer and append new chunk
        audio_data[0:-chunk_size] = audio_data[chunk_size:]
        audio_data[-chunk_size:] = data_in
        self.updateRelayData(data_in)
        self.audio_source.writeOutput(data_in)
        self.processTriggering()
        self.update()
        if overflow:
            self.audioOverflowSignal.emit()

    def initializeGL(self):
        self.ctx = moderngl.create_context()
        self.framebuffer = self.ctx.detect_framebuffer()
        self.scene = ScopeScene(self.ctx)

    def plotSignal(self):
        offset = self._display_position
        plot_start = ((self.SNAPSHOT_SIZE - self._display_sample_count) // 2
                      + offset)
        count = self._display_sample_count
        scale = self._display_sample_count
        start = 0

        if plot_start < 0:
            if plot_start + scale <= 0:
                return
            start = -plot_start
            count = scale - start
            plot_start = 0
        elif plot_start + scale > self.SNAPSHOT_SIZE:
            if plot_start >= self.SNAPSHOT_SIZE:
                return
            count = self.SNAPSHOT_SIZE - plot_start

        if self._display_relay:
            if self._trigger_edge == TriggerEdge.Positive:
                multiply = np.int16(8192)
            else:
                multiply = np.int16(-8192)
            plot_data = self.relay_snapshot_data[
                plot_start:plot_start+count].astype('i2') * multiply
            self.scene.plot_signal(plot_data, count, scale, start,
                                   color=self.RELAY_GRAPH_COLOR)
        if self._display_signal:
            plot_data = self.snapshot_data[plot_start:plot_start+count]
            self.scene.plot_signal(plot_data, count, scale, start,
                                   color=self.SIGNAL_COLOR)

    def paintGL(self):
        self.framebuffer.use()
        self.scene.clear()
        if self._display_fft_image or self._display_fft_graph:
            fft_data = np.abs((np.fft.rfft(
                self.audio_data[-self.FFT_CALC_SIZE:])
                    [:ScopeScene.FFT_WIDTH - 1]))
            if (self._display_fft_image and self.fft_tex_log
                    or self._display_fft_graph and self.fft_graph_log):
                fft_data2 = np.maximum((
                    np.log((fft_data + 1.)) - 11.) / 4., 0.).astype('f4')
                if self.fft_tex_log:
                    fft_tex_data = fft_data2
                if self.fft_graph_log:
                    fft_graph_data = fft_data2
            if (self._display_fft_image and not self.fft_tex_log
                    or self._display_fft_graph and not self.fft_graph_log):
                fft_data2 = (fft_data / (32768. * 512.)).astype('f4')
                if not self.fft_tex_log:
                    fft_tex_data = fft_data2
                if not self.fft_graph_log:
                    fft_graph_data = fft_data2
        if self._display_fft_image:
            self.scene.plot_fft(fft_tex_data)
        if self._display_grid:
            self.scene.plot_grid()
        if self._display_trigger_lines:
            self.scene.draw_hline(self._deadband_level, self.TRIGGER_ARM_COLOR)
            self.scene.draw_hline(self._trigger_level, self.TRIGGER_COLOR)
            if self._triggered:
                trig_pos = -(2 * self._display_position
                             / self._display_sample_count)
                if trig_pos >= -1 and trig_pos <= 1:
                    self.scene.draw_vline(trig_pos, self.TRIGGER_COLOR)
        if self._display_fft_graph:
            self.scene.plot_signal(
                (fft_graph_data * 32768 - 32768).astype('i2'),
                ScopeScene.FFT_WIDTH - 1, ScopeScene.FFT_WIDTH - 1, 0,
                color=self.FFT_GRAPH_COLOR)
        self.plotSignal()
        if self._display_grid:
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
    def triggerLevel(self) -> int:
        return self._trigger_level

    @triggerLevel.setter
    def triggerLevel(self, level: int):
        self._trigger_level = level
        self.updateDeadbandLevel()

    @property
    def triggerEdge(self) -> TriggerEdge:
        return self._trigger_edge

    @triggerEdge.setter
    def triggerEdge(self, edge: TriggerEdge):
        self._trigger_edge = edge
        self.updateDeadbandLevel()

    @property
    def deadband(self):
        return self._deadband

    @deadband.setter
    def deadband(self, value):
        self._deadband = value
        self.updateDeadbandLevel()

    @property
    def displayCount(self):
        return self._display_sample_count

    @displayCount.setter
    def displayCount(self, value):
        int_value = int(value)
        if int_value > self.scene.MAX_VERTEX_COUNT:
            int_value = self.scene.MAX_VERTEX_COUNT
        self._display_sample_count = int_value

    @property
    def displayPosition(self):
        return self._display_position

    @displayPosition.setter
    def displayPosition(self, value):
        int_value = int(value)
        if int_value > self.AUDIO_BUFFER_SIZE // 2:
            int_value = self.AUDIO_BUFFER_SIZE // 2
        elif int_value < -self.AUDIO_BUFFER_SIZE // 2:
            int_value = -self.AUDIO_BUFFER_SIZE // 2
        self._display_position = int_value

    @property
    def displaySignal(self) -> bool:
        return self._display_signal

    @displaySignal.setter
    def displaySignal(self, value: bool):
        self._display_signal = value

    @property
    def displayRelay(self) -> bool:
        return self._display_relay

    @displayRelay.setter
    def displayRelay(self, value: bool):
        self._display_relay = value

    @property
    def displayFFTGraph(self) -> bool:
        return self._display_fft_graph

    @displayFFTGraph.setter
    def displayFFTGraph(self, value: bool):
        self._display_fft_graph = value

    @property
    def displayFFTImage(self) -> bool:
        return self._display_fft_image

    @displayFFTImage.setter
    def displayFFTImage(self, value: bool):
        # Clear old FFT data before re-enabling
        if value:
            self.scene.clear_fft()
        self._display_fft_image = value

    @property
    def displayTriggerLines(self) -> bool:
        return self._display_trigger_lines

    @displayTriggerLines.setter
    def displayTriggerLines(self, value: bool):
        self._display_trigger_lines = value

    @property
    def displayGrid(self) -> bool:
        return self._display_grid

    @displayGrid.setter
    def displayGrid(self, value: bool):
        self._display_grid = value

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
