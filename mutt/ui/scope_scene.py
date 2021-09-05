from enum import IntEnum
from typing import Callable, cast, Optional

import numpy as np
import moderngl

from ..audio import AudioSource
from .shaders import SignalVA, GridVA, LineVA, TextureVA

# TODO temporary
from ..tape.loader import Loader


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
    # Number of horizontal grid divisions
    HGRID_DIV = 10
    # Number of vertical grid divisions
    VGRID_DIV = 8
    # Horizontal resolution for FFT texture
    FFT_WIDTH = 512
    # Vertical resolution for FFT texture
    FFT_HEIGHT = 1024
    # Horizontal resulution for pulse texture
    PULSE_TEX_WIDTH = 1024
    # Vertical resolution for pulse texture
    PULSE_TEX_HEIGHT = 1024
    # Color of signal graph
    SIGNAL_COLOR = (0.0, 1.0, 1.0, 1.0)
    # Color of relay graph
    RELAY_GRAPH_COLOR = (0.8, 0.8, 0.0, 1.0)
    # Color of fft graph
    FFT_GRAPH_COLOR = (0.8, 0.0, 0.3, 1.0)
    # Color of pulse graph
    PULSE_GRAPH_COLOR = (0.8, 0.8, 0.0, 1.0)
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
        self._display_sample_count = ScopeScene.MAX_VERTEX_COUNT // 100
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
        # Enable pulse width image
        self._display_pulse_image = True
        # Enable pulse width graph
        self._display_pulse_graph = False
        self._triggered = False

        # FFT settings
        self.fft_tex_log = True
        self.fft_graph_log = False

        # Relay settings
        self._trigger_edge = TriggerEdge.Positive
        self._high_level = 0
        self._low_level = 0

        # Snapshot data
        self.snapshot_data = np.zeros(self.SNAPSHOT_SIZE, np.int16)
        self.relay_snapshot_data = np.zeros(self.SNAPSHOT_SIZE, np.int8)

        # FFT data
        self.fft_tex_data = np.zeros(self.FFT_WIDTH, np.float32)
        self.fft_graph_data = np.zeros(self.FFT_WIDTH, np.int16)

        self.pulse_tex_data = np.zeros(self.PULSE_TEX_WIDTH, np.int16)
        self._pulse_tex_scale = 32.

        self.framebuffer: moderngl.Framebuffer = None
        self.ctx: moderngl.Context = None

        self.signal_vao: SignalVA = None    # type: ignore
        self.grid_vao: GridVA = None        # type: ignore
        self.line_vao: LineVA = None        # type: ignore
        self.fft_vao: TextureVA = None      # type: ignore
        self.pulse_vao: TextureVA = None    # type: ignore
        self.format_vao: TextureVA = None   # type: ignore
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
        self.line_vao = LineVA(self.ctx, main_rect)
        self.grid_vao = GridVA(self.ctx, main_rect, self.HGRID_DIV,
                               self.VGRID_DIV)
        self.fft_vao = TextureVA(self.ctx, main_rect, self.FFT_WIDTH,
                                 self.FFT_HEIGHT)
        self.pulse_vao = TextureVA(self.ctx, main_rect, self.PULSE_TEX_WIDTH,
                                   self.PULSE_TEX_HEIGHT, 1)
        self.pulse_vao.set_scale(self._pulse_tex_scale)
        self.format_vao = TextureVA(self.ctx, main_rect, self.PULSE_TEX_WIDTH,
                                    1, 2)
        self.format_vao.set_scale(self._pulse_tex_scale)

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
            multiply = np.int16(8192)
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

    def paint_pulse_texture(self):
        self.format_vao.render()
        self.pulse_vao.render_data((self.pulse_tex_data / 16384)
                                   .astype(np.float32))

    def paint_grid(self):
        self.grid_vao.render_grid()

    def paint_frame(self):
        self.grid_vao.render_frame()

    def paint_trigger_lines(self):
        self.line_vao.render_hline(
            self._low_level / 32768, self.TRIGGER_ARM_COLOR,
            self.TRIGGER_SEGMENTS)
        self.line_vao.render_hline(
            self._high_level / 32768, self.TRIGGER_COLOR,
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

    def paint_pulse_graph(self):
        width = int(ScopeScene.PULSE_TEX_WIDTH / self._pulse_tex_scale)
        self.signal_vao.render_data(
            (self.pulse_tex_data[:width] - 32768).astype(np.int16),
            width, width, 0, color=self.PULSE_GRAPH_COLOR)

    def paint(self):
        self.framebuffer.use()
        self.clear()
        if self._display_fft_image:
            self.paint_fft_texture()
        if self._display_pulse_image:
            self.paint_pulse_texture()
        if self._display_grid:
            self.paint_grid()
        if self._display_trigger_lines:
            self.paint_trigger_lines()
        if self._display_fft_graph:
            self.paint_fft_graph()
        if self._display_pulse_graph:
            self.paint_pulse_graph()
        self.paint_signal_and_relay()
        if self._display_grid:
            self.paint_frame()

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
    def highLevel(self) -> int:
        return self._high_level

    @highLevel.setter
    def highLevel(self, level: int):
        if level > 32767:
            level = 32767
        elif level < -32768:
            level = -32768
        self._high_level = level
        if self._low_level > self._high_level:
            self._low_level = self._high_level

    @property
    def lowLevel(self):
        return self._low_level

    @lowLevel.setter
    def lowLevel(self, level):
        if level > 32767:
            level = 32767
        elif level < -32768:
            level = -32768
        self._low_level = level
        if self._high_level < self._low_level:
            self._high_level = self._low_level

    @property
    def triggerEdge(self) -> TriggerEdge:
        return self._trigger_edge

    @triggerEdge.setter
    def triggerEdge(self, edge: TriggerEdge):
        self._trigger_edge = edge

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
    def displayPulseImage(self) -> bool:
        return self._display_pulse_image

    @displayPulseImage.setter
    def displayPulseImage(self, value: bool):
        # Clear old Pulse image data before re-enabling
        if value:
            self.pulse_vao.clear()
        self._display_pulse_image = value

    @property
    def displayPulseGraph(self) -> bool:
        return self._display_pulse_graph

    @displayPulseGraph.setter
    def displayPulseGraph(self, value: bool):
        self._display_pulse_graph = value

    @property
    def pulseImageScale(self) -> float:
        return self._pulse_tex_scale

    @pulseImageScale.setter
    def pulseImageScale(self, value: float):
        if self.pulse_vao is not None:
            self.pulse_vao.set_scale(value)
        if self.format_vao is not None:
            self.format_vao.set_scale(value)
        self._pulse_tex_scale = value

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
        super().__init__(False, ctx)
        self.audio_source: Optional[AudioSource] = None
        self._auto_count = 0
        self.audio_data = np.zeros(self.AUDIO_BUFFER_SIZE, np.int16)
        self.relay_data = -1 * np.ones(self.AUDIO_BUFFER_SIZE, np.int8)
        self._trigger_mode = TriggerMode.Normal
        self._force_trigger = False
        self._high_level = 250
        self._low_level = -250
        self._relay_count = 0
        self._pulse_count = 0
        self._diff_input = False
        self._play_relay = False
        self._speed_correction = 0
        self._start_level = 0
        self._integral = 0
        self._pulse_data: np.ndarray = cast(np.ndarray, None)
        self._tape_loader: Optional[Loader] = None

        self.trigger_mode_changed_callback: Optional[Callable] = None
        self.update_callback: Optional[Callable] = None
        self.input_overflow_callback: Optional[Callable] = None

        if initialize:
            self.initialize(ctx)

    def initialize(self, ctx: moderngl.Context):
        super().initialize(ctx)
        if self._tape_loader is not None:
            self.update_tape_format()

    def update_tape_format(self):
        if self.format_vao is None or self.audio_source is None:
            return
        data = np.zeros(self.PULSE_TEX_WIDTH, dtype=np.float32)
        freq = self.audio_source.FREQ
        if self._tape_loader is not None:
            for pulse_width in self._tape_loader.pulse_widths:
                samples_count = int(np.rint(pulse_width * freq / 1000000))
                if samples_count < self.PULSE_TEX_WIDTH:
                    data[samples_count] = 1.
        self.format_vao.write_data(data)

    def connect_audio(self, audio_source: AudioSource):
        self.audio_source = audio_source
        self.chunk_range = np.arange(audio_source.CHUNK_SIZE + 1)
        self.prev_data_in = np.zeros(audio_source.CHUNK_SIZE, np.int16)
        self.raw_relay_data = np.zeros(audio_source.CHUNK_SIZE, np.int16)
        self.audio_source.connect_data_ready(self.process_audio_data)
        self.update_tape_format()

    def connect_trigger_mode_changed(self, callback: Callable):
        self.trigger_mode_changed_callback = callback

    def connect_update(self, callback: Callable):
        self.update_callback = callback

    def connect_input_overflow(self, callback: Callable):
        self.input_overflow_callback = callback

    def update_fft_data(self):
        fft_data = np.abs((np.fft.rfft(
            self.audio_data[-self.FFT_CALC_SIZE:])
            [:ScopeScene.FFT_WIDTH]))
        if (self._display_fft_image and self.fft_tex_log
                or self._display_fft_graph and self.fft_graph_log):
            fft_data2 = np.maximum((
                np.log((fft_data + 1.)) - 11.) / 4., 0.).astype(np.float32)
            if self.fft_tex_log:
                self.fft_tex_data[:] = fft_data2
            if self.fft_graph_log:
                self.fft_graph_data[:] = (
                    (fft_data2 - 1) * 32768.).astype(np.int16)
        if (self._display_fft_image and not self.fft_tex_log
                or self._display_fft_graph and not self.fft_graph_log):
            fft_data2 = (fft_data / (32768. * 512.)).astype('f4')
            if not self.fft_tex_log:
                self.fft_tex_data[:] = fft_data2
            if not self.fft_graph_log:
                self.fft_graph_data[:] = (
                    (fft_data2 - 1) * 32768.).astype(np.int16)

    def detect_trigger_edge(self, last_value: int, arrange: np.ndarray,
                            first: bool = False) -> int:

        if self._trigger_edge == TriggerEdge.Positive:
            trigger = 1
            fmin = np.argmin
            fmax = np.argmax
        else:
            trigger = -1
            fmin = np.argmax
            fmax = np.argmin

        if first:
            sidx: int = 0
            if last_value != -trigger:
                # was not armed initially - find first element 'below'
                sidx = cast(int, fmin(arrange))
                # sidx will be 0 if either first element is below or none
                if sidx == 0 and arrange[0] != -trigger:
                    return -1

            # find first element 'above'
            idx = fmax(arrange[sidx:])
            # idx will be 0 if either first element is above or no elements
            if arrange[sidx + idx] != trigger:
                return -1
            return cast(int, sidx + idx)

        else:
            # Reverse the array while checking condition
            crrange = arrange[::-1]
            sidx = 0

            # Never armed
            if crrange[0] == 0:
                return -1

            if crrange[0] == -trigger:
                # last element is 'below', find last 'above'
                sidx = cast(int, fmax(crrange))
                if sidx == 0:
                    # all elements 'below'
                    return -1

            # now find last element 'below'
            idx = fmin(crrange[sidx:])
            if idx == 0:
                # no 'raising' edge in array and first element is above
                if last_value == -trigger:
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

    def process_pulse_data_us(self, start_level: int,
                              pulse_widths_us: np.ndarray):
        pass

    def process_cycle_data_us(self, cycle_width_us: np.ndarray):
        if self._tape_loader is not None:
            self._tape_loader.process_input(cycle_width_us)

    def process_pulse_data(self, start_level: int, pulse_widths: np.ndarray):
        if len(pulse_widths) == 0:
            self.pulse_tex_data[:] = 0
            return
        if self._trigger_edge == TriggerEdge.Positive:
            trigger = 1
        else:
            trigger = -1

        assert self.audio_source is not None
        multiplier = 1000000 / self.audio_source.FREQ

        if start_level == trigger:
            count = len(pulse_widths) // 2
            relay_widths = (pulse_widths[:2 * count:2]
                            + pulse_widths[1:2 * count:2])
        else:
            count = (len(pulse_widths) - 1) // 2 + 1
            relay_widths = np.ndarray(count, dtype=int)
            relay_widths[0] = self._pulse_count + pulse_widths[0]
            relay_widths[1:] = (pulse_widths[1:(2 * count - 1):2]
                                + pulse_widths[2:(2 * count - 1):2])
        self._pulse_count = pulse_widths[-1]
        if self._display_pulse_image or self._display_pulse_graph:
            # self.pulse_tex_data[:] = self.pulse_tex_data[:] >> 1
            tex_data = np.bincount(np.minimum(relay_widths,
                                   self.PULSE_TEX_WIDTH + 1))
            copy_length = min(len(tex_data), self.PULSE_TEX_WIDTH)
            self.pulse_tex_data[:copy_length] = np.minimum(
                    tex_data[:copy_length], 1) * 16384
            self.pulse_tex_data[copy_length:] = 0
        self.process_pulse_data_us(start_level, pulse_widths * multiplier)
        self.process_cycle_data_us(relay_widths * multiplier)

    def update_relay_data(self, audio_source: AudioSource,
                          data_in: np.ndarray):
        chunk_size = audio_source.CHUNK_SIZE
        relay_data = self.relay_data

        # Put last relay condition at the beginning of buffer to process
        relay_chunk = np.zeros(chunk_size + 1, dtype=np.int8)
        relay_chunk[0] = relay_data[-1]

        # Put relay state for each sample in the buffer
        np.subtract(np.greater(data_in, self._high_level).astype(np.int8),
                    np.less(data_in, self._low_level).astype(np.int8),
                    out=relay_chunk[1:])

        # https://stackoverflow.com/questions/68869535/numpy-accumulate-greater-operation
        # Process hysteresis on the chunk
        masked_indexes = np.where((relay_chunk != 0), self.chunk_range, 0)
        hyst_indexes = np.maximum.accumulate(masked_indexes)
        relay_chunk_hyst = relay_chunk[hyst_indexes]

        # Get indexes where relay value changes
        indexes = np.nonzero(np.diff(relay_chunk_hyst))[0]
        if len(indexes) > 0:
            indexes2 = np.insert(indexes, 0, 0)
            widths = np.diff(indexes2)
            widths[0] += self._relay_count
            if self._speed_correction != 0:
                widths = (widths *
                          (1. + self._speed_correction / 100)).astype(int)
            self._relay_count = chunk_size - indexes[-1]
            # self.trigger_indexes = np.stack((result[indexes],
            #                                  indexes)).transpose()
            self._start_level = -relay_chunk_hyst[indexes[0]]
            self._pulse_data = widths
        else:
            # self.trigger_indexes = np.ndarray((0, 2))
            self._pulse_data = np.ndarray(0)
            self._pulse_count += chunk_size
        self.process_pulse_data(self._start_level, self._pulse_data)

        # Move data in buffer append new chunk
        relay_data[0:-chunk_size] = relay_data[chunk_size:]
        relay_data[-chunk_size:] = relay_chunk_hyst[1:]
        self.raw_relay_data[:] = relay_chunk_hyst[1:] * 8192

    def process_audio_data(self, audio_source: AudioSource):
        chunk_size = audio_source.CHUNK_SIZE
        audio_data = self.audio_data

        raw_data_in, overflow = audio_source.get_buffer()
        if self._diff_input:
            data_in = np.diff(np.concatenate(
                    (self.prev_data_in[-1:], raw_data_in)))
        else:
            data_in = raw_data_in
        self.prev_data_in[:] = raw_data_in

        # Move data in audio buffer and append new chunk
        audio_data[0:-chunk_size] = audio_data[chunk_size:]
        audio_data[-chunk_size:] = data_in
        self.fft_graph_data[:] = self.fft_graph_data[:] / 2
        self.fft_tex_data[:] = self.fft_tex_data[:] / 2
        self.update_relay_data(audio_source, data_in)
        if self._play_relay:
            audio_source.write_output(self.raw_relay_data)
        else:
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

    @property
    def diffInput(self) -> bool:
        return self._diff_input

    @diffInput.setter
    def diffInput(self, value: bool):
        self._diff_input = value

    @property
    def playRelay(self) -> bool:
        return self._play_relay

    @playRelay.setter
    def playRelay(self, value: bool):
        self._play_relay = value

    @property
    def speedCorrection(self) -> int:
        return self._speed_correction

    @speedCorrection.setter
    def speedCorrection(self, value: int):
        self._speed_correction = value

    @property
    def tapeDecoder(self):
        return self._tape_loader

    @tapeDecoder.setter
    def tapeDecoder(self, decoder):
        self._tape_loader = decoder
        self.update_tape_format()
