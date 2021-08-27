from enum import IntEnum
from typing import Callable, cast, Optional

import numpy as np
import moderngl

from ..audio import AudioSource
from .shaders import SignalVA, GridVA, LineVA, FFTTextureVA


class TriggerEdge(IntEnum):
    Positive = 0
    Negative = 1


class TriggerMode(IntEnum):
    Off = 0
    Stop = 1
    Normal = 2
    Auto = 3
    Single = 4


class ScopeScene:
    # Maximum number of signal vertices on screen
    MAX_VERTEX_COUNT = 6250
    # Margin for main scope rectangle
    MARGIN = 0.02
    # Number of horizontal grid divisons
    HGRID_DIV = 10
    # Number of vertical grid divisons
    VGRID_DIV = 8
    # Horizontal resolution for FFT texture
    FFT_WIDTH = 512
    # Vertical resolution for FFT texture
    FFT_HEIGHT = 1024
    # Color of signal graph
    SIGNAL_COLOR = (0.0, 1.0, 1.0, 1.0)
    # Color of relay graph
    RELAY_GRAPH_COLOR = (0.8, 0.8, 0.0, 1.0)
    # Color of fft graph
    FFT_GRAPH_COLOR = (0.8, 0.0, 0.3, 1.0)
    # Color of trigger level line
    TRIGGER_COLOR = (0.9, 0.5, 0.5, 0.8)
    # Color of trigger deadband line
    TRIGGER_ARM_COLOR = (0.9, 0.5, 0.5, 0.8)
    # Number of trigger line segments
    TRIGGER_SEGMENTS = 33
    # Size of snapshort data
    SNAPSHOT_SIZE = MAX_VERTEX_COUNT * 2

    @staticmethod
    def scale_matrix(x: float, y: float):
        result = np.eye(4)
        result[0, 0] = x
        result[1, 1] = y
        return result

    def __init__(self, initialize: bool = False, ctx: moderngl.Context = None):
        # Number of displayed samples (corresponds to H Scale)
        self._display_sample_count = ScopeScene.MAX_VERTEX_COUNT
        # Horizontal sample offset (corresponds to H Position)
        self._display_position = 0
        # Enable signal graph
        self._display_signal = True
        # Enable relay graph
        self._display_relay = False
        # Enable FFT texture
        self._display_fft_image = False
        # Enable FFT graph
        self._display_fft_graph = False
        # Enable trigger lines
        self._display_trigger_lines = True
        # Enable grid and frame
        self._display_grid = True
        self._triggered = False

        # FFT settings
        self.fft_tex_log = True
        self.fft_graph_log = False

        # Relay settings
        self._trigger_edge = TriggerEdge.Positive
        self._trigger_level = 0
        self._deadband = 0
        self._deadband_level = 0

        # Snapshot data
        self.snapshot_data = np.zeros(self.SNAPSHOT_SIZE, np.int16)
        self.relay_snapshot_data = np.zeros(self.SNAPSHOT_SIZE, np.int8)

        # FFT data
        self.fft_tex_data = np.zeros(self.FFT_WIDTH, np.float32)
        self.fft_graph_data = np.zeros(self.FFT_WIDTH, np.int16)

        self.framebuffer: moderngl.Framebuffer = None
        self.ctx: moderngl.Context = None

        self.signal_vao: SignalVA = None    # type: ignore
        self.grid_vao: GridVA = None        # type: ignore
        self.line_vao: LineVA = None        # type: ignore
        self.fft_vao: FFTTextureVA = None   # type: ignore
        if initialize:
            self.initialize(ctx)

    def initialize(self, ctx: moderngl.Context):
        if ctx is not None:
            self.ctx = ctx
        else:
            self.ctx = moderngl.create_context()
        self.framebuffer = self.ctx.detect_framebuffer()

        main_rect = self.scale_matrix(1 - self.MARGIN, 1 - self.MARGIN)

        self.signal_vao = SignalVA(self.ctx, main_rect, self.MAX_VERTEX_COUNT)
        self.grid_vao = GridVA(self.ctx, main_rect, self.HGRID_DIV,
                               self.VGRID_DIV)
        self.line_vao = LineVA(self.ctx, main_rect)
        self.fft_vao = FFTTextureVA(self.ctx, main_rect, self.FFT_WIDTH,
                                    self.FFT_HEIGHT)

    def clear(self, color=(0, 0, 0, 0)):
        self.ctx.clear(*color)

    def paint_signal_and_relay(self):
        # Offset from center of snapshot data
        offset = self._display_position
        # Starting index in snapshot data to display
        plot_start = ((self.SNAPSHOT_SIZE - self._display_sample_count) // 2
                      + offset)
        # Actual number of samples to draw
        count = self._display_sample_count
        # Scale (number of samples per screen width)
        scale = self._display_sample_count
        # Drawing starting position from left screen edge in samples
        start = 0

        # Compensate if starting index is negative
        if plot_start < 0:
            # Drawing completely outside of the screen
            if plot_start + scale <= 0:
                return
            # Crop samples outside left edge
            start = -plot_start
            count = scale - start
            plot_start = 0
        # Compensate if ending index is outside of right edge
        elif plot_start + scale > self.SNAPSHOT_SIZE:
            # Drawing completely outside of the screen
            if plot_start >= self.SNAPSHOT_SIZE:
                return
            # Crop samples outside right edge
            count = self.SNAPSHOT_SIZE - plot_start

        if self._display_relay:
            if self._trigger_edge == TriggerEdge.Positive:
                multiply = np.int16(8192)
            else:
                multiply = np.int16(-8192)
            plot_data = self.relay_snapshot_data[
                plot_start:plot_start+count].astype('i2') * multiply
            self.signal_vao.render_data(plot_data, count, scale, start,
                                        color=self.RELAY_GRAPH_COLOR)
        if self._display_signal:
            plot_data = self.snapshot_data[plot_start:plot_start+count]
            self.signal_vao.render_data(plot_data, count, scale, start,
                                        color=self.SIGNAL_COLOR)

    def paint_fft_texture(self):
        self.fft_vao.render_data(self.fft_tex_data)

    def paint_grid(self):
        self.grid_vao.render_grid()

    def paint_frame(self):
        self.grid_vao.render_frame()

    def paint_trigger_lines(self):
        self.line_vao.render_hline(
            self._deadband_level / 32768, self.TRIGGER_ARM_COLOR,
            self.TRIGGER_SEGMENTS)
        self.line_vao.render_hline(
            self._trigger_level / 32768, self.TRIGGER_COLOR,
            self.TRIGGER_SEGMENTS)
        if self._triggered:
            trig_pos = -(2 * self._display_position
                         / self._display_sample_count)
            if trig_pos >= -1 and trig_pos <= 1:
                self.line_vao.render_vline(
                    trig_pos, self.TRIGGER_COLOR, self.TRIGGER_SEGMENTS)

    def paint_fft_graph(self):
        self.signal_vao.render_data(
            self.fft_graph_data,
            ScopeScene.FFT_WIDTH - 1, ScopeScene.FFT_WIDTH - 1, 0,
            color=self.FFT_GRAPH_COLOR)

    def paint(self):
        self.framebuffer.use()
        self.clear()
        if self._display_fft_image:
            self.paint_fft_texture()
        if self._display_grid:
            self.paint_grid()
        if self._display_trigger_lines:
            self.paint_trigger_lines()
        if self._display_fft_graph:
            self.paint_fft_graph()
        self.paint_signal_and_relay()
        if self._display_grid:
            self.paint_frame()

    def update_deadband_level(self):
        if self._trigger_edge == TriggerEdge.Positive:
            deadband_level = self._trigger_level - self._deadband
            if deadband_level < -32768:
                deadband_level = 32768
        else:
            deadband_level = self._trigger_level + self._deadband
            if deadband_level > 32767:
                deadband_level = 32767
        self._deadband_level = deadband_level

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
        self.update_deadband_level()

    @property
    def triggerEdge(self) -> TriggerEdge:
        return self._trigger_edge

    @triggerEdge.setter
    def triggerEdge(self, edge: TriggerEdge):
        self._trigger_edge = edge
        self.update_deadband_level()

    @property
    def deadband(self):
        return self._deadband

    @deadband.setter
    def deadband(self, value):
        self._deadband = value
        self.update_deadband_level()

    @property
    def displayCount(self):
        return self._display_sample_count

    @displayCount.setter
    def displayCount(self, value):
        int_value = int(value)
        if int_value > self.MAX_VERTEX_COUNT:
            int_value = self.MAX_VERTEX_COUNT
        self._display_sample_count = int_value

    @property
    def displayPosition(self):
        return self._display_position

    @displayPosition.setter
    def displayPosition(self, value):
        int_value = int(value)
        if int_value > self.MAX_VERTEX_COUNT:
            int_value = self.MAX_VERTEX_COUNT
        elif int_value < -self.MAX_VERTEX_COUNT:
            int_value = -self.MAX_VERTEX_COUNT
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
            self.fft_vao.clear()
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


class ScopeSceneSignal(ScopeScene):
    AUDIO_BUFFER_SIZE = ScopeScene.MAX_VERTEX_COUNT * 3
    AUTO_TRIGGER = 10
    FFT_CALC_SIZE = 4096

    def __init__(self, initialize: bool = False, ctx: moderngl.Context = None):
        super().__init__(initialize, ctx)
        self.audio_source: Optional[AudioSource] = None
        self._auto_count = 0
        self.audio_data = np.zeros(self.AUDIO_BUFFER_SIZE, np.int16)
        self.relay_data = -1 * np.ones(self.AUDIO_BUFFER_SIZE, np.int8)
        self._trigger_mode = TriggerMode.Normal
        self._force_trigger = False
        self.deadband = 500

        self.trigger_mode_changed_callback: Optional[Callable] = None
        self.update_callback: Optional[Callable] = None
        self.input_overflow_callback: Optional[Callable] = None

    def connect_audio(self, audio_source: AudioSource):
        self.audio_source = audio_source
        self.chunk_range = np.arange(audio_source.CHUNK_SIZE + 1)
        self.audio_source.connect_data_ready(self.update_audio_data)

    def connect_trigger_mode_changed(self, callback: Callable):
        self.trigger_mode_changed_callback = callback

    def connect_update(self, callback: Callable):
        self.update_callback = callback

    def connect_input_overflow(self, callback: Callable):
        self.input_overflow_callback = callback

    def update_fft_data(self):
        fft_data = np.abs((np.fft.rfft(
            self.audio_data[-self.FFT_CALC_SIZE:])
            [:ScopeScene.FFT_WIDTH - 1]))
        if (self._display_fft_image and self.fft_tex_log
                or self._display_fft_graph and self.fft_graph_log):
            fft_data2 = np.maximum((
                np.log((fft_data + 1.)) - 11.) / 4., 0.).astype(np.float32)
            if self.fft_tex_log:
                self.fft_tex_data = fft_data2
            if self.fft_graph_log:
                self.fft_graph_data = (
                    (fft_data2 - 1) * 32768.).astype(np.int16)
        if (self._display_fft_image and not self.fft_tex_log
                or self._display_fft_graph and not self.fft_graph_log):
            fft_data2 = (fft_data / (32768. * 512.)).astype('f4')
            if not self.fft_tex_log:
                self.fft_tex_data = fft_data2
            if not self.fft_graph_log:
                self.fft_graph_data = (
                    (fft_data2 - 1) * 32768.).astype(np.int16)

    def detect_trigger_edge(self, last_value: int, arrange: np.ndarray,
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

    def process_triggering(self):

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
            trigger_idx = self.detect_trigger_edge(
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
                callback = self.trigger_mode_changed_callback
                if callback is not None:
                    callback(self)
            copy_start = scan_start + trigger_idx - half_max_screen
            self.snapshot_data[:] = self.audio_data[
                    copy_start:copy_start + max_screen]
            self.relay_snapshot_data[:] = self.relay_data[
                    copy_start:copy_start + max_screen]

    def update_relay_data(self, audio_source: AudioSource,
                          data_in: np.ndarray):
        chunk_size = audio_source.CHUNK_SIZE
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

    def update_audio_data(self, audio_source: AudioSource):
        chunk_size = audio_source.CHUNK_SIZE
        audio_data = self.audio_data

        data_in, overflow = audio_source.get_buffer()
        # Move data in audio buffer and append new chunk
        audio_data[0:-chunk_size] = audio_data[chunk_size:]
        audio_data[-chunk_size:] = data_in
        self.update_relay_data(audio_source, data_in)
        audio_source.write_output(data_in)
        self.process_triggering()
        if self._display_fft_image or self._display_fft_graph:
            self.update_fft_data()
        update_callback = self.update_callback
        if update_callback is not None:
            update_callback(self)
        if overflow:
            overflow_callback = self.input_overflow_callback
            if overflow_callback is not None:
                overflow_callback(self)

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
