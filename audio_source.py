import numpy as np
import pyaudio
from PyQt5 import QtCore


class AudioSource(QtCore.QObject):
    # FX = np.linspace(-1.0, 1.0, 882)
    CHUNK_SIZE = 882
    data_ready = QtCore.pyqtSignal()

    def connectDataReady(self, func):
        self.data_ready.connect(func)

    def start(self):
        pass

    def stop(self):
        pass

    def getBuffer(self) -> np.array:
        pass


class PyAudioSource(AudioSource):
    def __init__(self, audio=None):
        super().__init__()
        if audio is None:
            self.audio = pyaudio.PyAudio()
        else:
            self.audio = audio
        self.next_buffer = 0
        self.buffers = np.zeros((2, self.CHUNK_SIZE), np.int16)
        self.stream = self.audio.open(44100, 1, pyaudio.paInt16, input=True,
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

    def getBuffer(self) -> np.array:
        return self.buffers[1 - self.next_buffer]
