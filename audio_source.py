from typing import Optional, Mapping

import numpy as np
import pyaudio
from PyQt5 import QtCore


class AudioSource(QtCore.QObject):
    CHUNK_SIZE = 882
    data_ready = QtCore.pyqtSignal()

    def connectDataReady(self, func):
        self.data_ready.connect(func)

    def start(self):
        pass

    def stop(self):
        pass

    def start_listening(self) -> bool:
        return False

    def stop_listening(self):
        pass

    def getBuffer(self) -> np.array:
        return np.zeros(self.CHUNK_SIZE, np.int16)


class PyAudioSource(AudioSource):
    CHUNK_SIZE = 882

    def __init__(self, audio: pyaudio.PyAudio = None, enable_listening = False):
        super().__init__()
        if audio is None:
            self.audio = pyaudio.PyAudio()
        else:
            self.audio = audio
        self._buffer = np.zeros(self.CHUNK_SIZE, np.int16)
        self._have_chunk = False
        self._listening = False
        self.input_stream = self.audio.open(
                44100, 1, pyaudio.paInt16, input=True,
                frames_per_buffer=self.CHUNK_SIZE,
                stream_callback=self.input_callback)
        self.output_stream = None
        if enable_listening:
            self.output_stream = self.audio.open(
                    44100, 1, pyaudio.paInt16, output=True,
                    frames_per_buffer=self.CHUNK_SIZE,
                    stream_callback=self.output_callback)
            self.output_stream.stop_stream()

    def input_callback(self, in_data: Optional[bytes], frame_count: int,
                        time_info: Mapping[str, float], status: int):
        self._buffer[:] = np.frombuffer(in_data, dtype=np.int16)
        self.data_ready.emit()
        self._have_chunk = True
        return (None, pyaudio.paContinue)

    def output_callback(self, in_data: Optional[bytes], frame_count: int,
                        time_info: Mapping[str, float], status: int):
        if not self._listening:
            return (None, pyaudio.paComplete)
        if self._have_chunk:
            self._have_chunk = False
            return (bytes(self._buffer), pyaudio.paContinue)
        return (np.zeros(self.CHUNK_SIZE, np.int16), pyaudio.paContinue)

    def start(self):
        self.input_stream.start_stream()

    def stop(self):
        self.input_stream.stop_stream()
        self.stop_listening()

    def start_listening(self) -> bool:
        if self.output_stream is None:
            return False
        self._listening = True
        self.output_stream.start_stream()
        return True

    def stop_listening(self):
        if self._listening:
            self.output_stream.stop_stream()
            self._listening = False

    def getBuffer(self) -> np.array:
        return self._buffer
