from typing import Tuple
import numpy as np
import os
import os.path

class Format:
    @property
    def pulse_widths(self) -> Tuple[int, ...]:
        pass

    def process_input(self, cycles_us: np.ndarray):
        pass


class TurboTapeFormat(Format):
    WIDTHS = (208, 320)
    THRESHOLD = 263

    SYNCING_BITS = 0
    SYNCING_SEQ = 1
    SYNCED = 2
    HEADER = 3
    DATA = 4

    def __init__(self):
        self.byte = 0
        self.sync_seq = 9
        self.bitcnt = 0
        self._state = self.SYNCING_BITS
        self.header_type = 0
        self.header = bytearray(192)
        self.start_addr = 0
        self.end_addr = 0
        self.file_name = b''
        self.unicode_file_name = ""
        self.data_size = 0
        self.data = bytearray(0)
        self.pointer = 0

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def pulse_widths(self) -> Tuple[int, ...]:
        return self.WIDTHS

    def error(self, text: str):
        self.print_progress()
        print("")
        print("Error: %s" % text)
        self.data_size = 0
        self.state = self.SYNCING_BITS

    def success(self):
        self.print_progress()
        print("")
        print("OK")
        illegal='/?<>\\:*|^"\''
        file_name = "".join([char
            for char in self.unicode_file_name
            if ord(char) >= 32 and char not in illegal]).strip()
        output = "output"
        if not os.path.isdir(output):
            os.mkdir(output)
        file_path = output + os.sep + file_name
        with open(file_path + ".prg", "wb") as file:
            file.write(self.start_addr.to_bytes(2, 'little'))
            file.write(self.data)
        self.state = self.SYNCING_BITS

    def process_header(self):
        self.start_addr = int.from_bytes(self.header[0:2], 'little')
        self.end_addr = int.from_bytes(self.header[2:4], 'little')
        self.file_name = self.header[5:21].rstrip(b'\0')
        self.unicode_file_name = self.file_name.decode(
                'utf-8', errors='replace')
        print("{:16s} {:04X} {:04X}".format(
            self.unicode_file_name, self.start_addr, self.end_addr))
        self.data_size = self.end_addr - self.start_addr

    def print_progress(self):
        if self.data_size != 0:
            print("\r{:16s} {:04X} {:04X} {:04X}".format(
                self.unicode_file_name, self.start_addr, self.end_addr,
                self.start_addr + self.pointer), end='')

    def process_file(self, checksum):
        sum = np.bitwise_xor.reduce(self.data)
        if sum != checksum:
            self.error("LOAD ERROR")
            # print("checksum {:x}!={:x}".format(sum, checksum))
        else:
            self.success()
        self.data_size = 0

    def process_bit(self, bit: int):
        self.byte = ((self.byte << 1) & 255) | bit

        state = self.state
        if state == self.SYNCING_BITS:
            if self.byte == 2:
                self.state = self.SYNCING_SEQ
                self.bitcnt = 0
                self.sync_seq = 9
            return

        self.bitcnt += 1
        if self.bitcnt != 8:
            return

        self.bitcnt = 0
        if state == self.SYNCING_SEQ:
            if self.byte == 2 and self.sync_seq != 2:
                return
            if self.byte != self.sync_seq:
                self.state = self.SYNCING_BITS
                return
            self.sync_seq -= 1
            if self.sync_seq == 0:
                self.state = self.SYNCED
            return

        if state == self.SYNCED:
            if self.byte == 1 or self.byte == 2:
                self.header_type = self.byte
                self.state = self.HEADER
                self.pointer = 0
            elif self.byte == 0:
                if self.data_size == 0:
                    self.error("Data without header")
                else:
                    self.data = bytearray(self.data_size)
                    self.state = self.DATA
                    print("LOADING")
                    self.pointer = 0
            else:
                self.error("Bad block type %d" % self.byte)
            return

        if state == self.HEADER:
            self.header[self.pointer] = self.byte
            self.pointer += 1
            if self.pointer == 192:
                self.process_header()
                self.state = self.SYNCING_BITS
            return

        if self.pointer == self.data_size:
            self.process_file(self.byte)
            self.state = self.SYNCING_BITS
            return

        self.data[self.pointer] = self.byte
        self.pointer += 1
        if self.pointer % 16 == 0:
            self.print_progress()

    def process_input(self, cycles_us: np.ndarray):
        for cycle in cycles_us:
            if self.state >= self.SYNCED and (cycle < 160 or cycle > 416):
                self.error("Sync lost %d us" % cycle)
            else:
                if cycle > self.THRESHOLD:
                    bit = 1
                else:
                    bit = 0
                self.process_bit(bit)
