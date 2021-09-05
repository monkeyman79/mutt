from typing import Callable, Tuple
import numpy as np
import os
import os.path


def save_file(name: str, start_addr: int, data: bytes):
    illegal = '/?<>\\:*|^"\''
    file_data = bytearray(2 + len(data))
    file_data[:2] = start_addr.to_bytes(2, 'little')
    file_data[2:] = data
    file_name = "".join(
        [char
            for char in name
            if ord(char) >= 32 and char not in illegal]).strip()
    output = "output"
    if not os.path.isdir(output):
        os.mkdir(output)
    file_path = output + os.sep + file_name + ".prg"
    if os.path.exists(file_path):
        index = 2
        while True:
            if os.path.getsize(file_path) == len(file_data):
                with open(file_path, "rb") as prev_file:
                    prev_data = prev_file.read()
                if prev_data == file_data:
                    print("Identical file %s already exists" % file_path)
                    return
            file_path = "{}{}{} ({}).prg".format(
                    output, os.sep, file_name, index)
            if not os.path.exists(file_path):
                break
            index += 1
    with open(file_path, "wb") as file:
        file.write(file_data)


class Loader:
    WIDTHS: Tuple[int, ...] = ()

    NOT_SYNCED = 0
    SYNCING_SEQ = 2
    SYNCED = 3
    HEADER = 4
    DATA = 5

    def _print(self, text, newline: bool = True):
        handler = self.print_handler
        if handler is not None and text != "":
            handler(text)
        print(text, end=('\n' if newline else ''))

    def __init__(self):
        self.start_addr = 0
        self.end_addr = 0
        self.file_name = b''
        self.unicode_file_name = ""
        self.data_size = 0
        self.pointer = 0
        self.data = bytes()
        self.print_handler: Callable[[str], None] = None

    def reset(self):
        self.data_size = 0

    def print_progress(self):
        if self.data_size != 0:
            msg = "\r{:16s} {:04X} {:04X} {:04X}".format(
                self.unicode_file_name, self.start_addr, self.end_addr,
                self.start_addr + self.pointer)
            self._print(msg, False)

    def error(self, text: str):
        self.print_progress()
        self._print("")
        self._print("Error: %s" % text)
        self.reset()

    def success(self):
        self.print_progress()
        self._print("")
        self._print("OK")
        save_file(self.unicode_file_name, self.start_addr, self.data)
        self.reset()

    def process_pulse(self, pulse_us: int):
        pass

    def process_input(self, pulses_us: np.ndarray):
        for pulse in pulses_us:
            self.process_pulse(pulse)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def pulse_widths(self) -> Tuple[int, ...]:
        return self.WIDTHS

    @property
    def name(self) -> str:
        return "Null"


class KernalLoader(Loader):
    WIDTHS = (352, 512, 672)
    SHORT_THRESHOLD = 432
    LONG_THRESHOLD = 592

    LEADER_DETECT = 3000

    PULSE_NONE = 0
    PULSE_SHORT = 1
    PULSE_MEDIUM = 2
    PULSE_LONG = 3

    PAIR_0 = 0
    PAIR_1 = 1
    PAIR_START = 2
    PAIR_EOF = 3
    PAIR_LEAD = 4
    PAIR_BAD = 5

    def __init__(self):
        super().__init__()
        self.sync_count = 0
        self.byte = 0
        self.sync_seq = 9
        self.bitcnt = 0
        self.first = self.PULSE_NONE
        self._state = self.NOT_SYNCED
        self.header_type = 0
        self.header = bytearray(192)
        self.data = bytearray(0)
        self.repeated_block = False
        self.header_repeated = False

    def process_header(self, checksum):
        sum = np.bitwise_xor.reduce(self.header)
        if sum != checksum:
            self.error("Header checksum error")
            return
        self.start_addr = int.from_bytes(self.header[1:3], 'little')
        self.end_addr = int.from_bytes(self.header[3:5], 'little')
        self.file_name = self.header[5:21].rstrip(b'\0')
        self.unicode_file_name = self.file_name.decode(
                'utf-8', errors='replace')
        self._print("{:16s} {:04X} {:04X}".format(
            self.unicode_file_name, self.start_addr, self.end_addr))
        self.data_size = self.end_addr - self.start_addr

    def process_file(self, checksum):
        sum = np.bitwise_xor.reduce(self.data)
        if sum != checksum:
            self.error("File checksum error")
            return
            # print("checksum {:x}!={:x}".format(sum, checksum))
        self.success()

    def process_byte(self):
        if self.state == self.SYNCING_SEQ:
            if self.byte != self.sync_seq and self.byte != self.sync_seq + 128:
                self.state = self.NOT_SYNCED
                return
            self.sync_seq -= 1
            if self.sync_seq == 0:
                if self.byte >= 128:
                    self.repeated_block = False
                else:
                    self.repeated_block = True
                self.state = self.SYNCED
            return

        if self.state == self.SYNCED:
            if (self.data_size == 0
                    or not self.header_repeated):
                self.state = self.HEADER
                self.header_type = self.byte
                self.header_repeated = self.repeated_block
                self.pointer = 0
                if self.header_type == 5:
                    self._print("Ignoring END OF TAPE")
                    self.reset()
                    return
                elif self.header_type == 2 or self.header_type == 4:
                    self._print("Ignoring SEQ file")
                    self.reset()
                    return
                elif self.header_type != 1 and self.header_type != 3:
                    self.error("Invalid header type %d" % self.header_type)
                    self.reset()
                    return
            else:
                self.data = bytearray(self.data_size)
                self.state = self.DATA
                self._print("LOADING")
                self.pointer = 0
                # process first data byte

        if self.state == self.HEADER:
            if self.pointer < 192:
                self.header[self.pointer] = self.byte
                self.pointer += 1
            else:
                self.process_header(self.byte)
                self.state = self.NOT_SYNCED
                self.sync_count = 0
            return

        if self.pointer == self.data_size:
            self.process_file(self.byte)
            self.state = self.NOT_SYNCED
            return

        self.data[self.pointer] = self.byte
        self.pointer += 1
        if self.pointer % 16 == 0:
            self.print_progress()

    def parity(self, byte: int):
        byte = (byte ^ (byte >> 4)) & 0xF
        return (0x6996 >> byte) & 1

    def sync_lost(self, message):
        if self.state > self.SYNCING_SEQ:
            self.error(message)
            return
        if self.sync_count >= self.LEADER_DETECT:
            self._print("Lead signal lost")
        self.state = self.NOT_SYNCED
        self.sync_count = 0

    def process_pair(self, pair: int):
        if self.state == self.NOT_SYNCED:
            if pair != self.PAIR_START:
                self.sync_lost("Framing error")
                return
            self.state = self.SYNCING_SEQ
            self.bitcnt = 0
            self.sync_seq = 9

        if pair == self.PAIR_BAD or pair == self.PAIR_LEAD:
            self.sync_lost("Framing error")
            return

        if pair == self.PAIR_START or pair == self.PAIR_EOF:
            if self.bitcnt != 0:
                self.sync_lost("Framing error")
                return
            if pair == self.PAIR_EOF and self.state != self.NOT_SYNCED:
                self.sync_lost("Unexpected EOF marker")
                return
            self.bitcnt = 1
            return

        if self.bitcnt == 0:
            self.sync_lost("Framing error")
            return

        if self.bitcnt < 9:
            self.byte = ((self.byte >> 1) & 255) | (pair << 7)
            self.bitcnt += 1
            return

        if pair ^ self.parity(self.byte) == 0:
            self.sync_lost("Parity error")
            return

        self.process_byte()
        self.bitcnt = 0

    def process_pulse(self, pulse_us: int):
        if pulse_us < 272 or pulse_us > 752:
            self.sync_lost("Sync lost %d us" % pulse_us)
            return

        if pulse_us <= self.SHORT_THRESHOLD:
            length = self.PULSE_SHORT
        elif pulse_us > self.LONG_THRESHOLD:
            length = self.PULSE_LONG
        else:
            length = self.PULSE_MEDIUM

        if (self.state < self.SYNCING_SEQ and self.first != self.PULSE_LONG
                and length == self.PULSE_SHORT):
            self.first = self.PULSE_NONE
            self.sync_count += 1
            if self.sync_count == self.LEADER_DETECT:
                self._print("Lead signal detected")
            return

        if self.first == self.PULSE_NONE:
            self.first = length
            return

        pair = self.PAIR_BAD
        if self.first == self.PULSE_SHORT:
            if length == self.PULSE_SHORT:
                pair = self.PAIR_LEAD
            elif length == self.PULSE_MEDIUM:
                pair = self.PAIR_0
        elif self.first == self.PULSE_MEDIUM:
            if length == self.PULSE_SHORT:
                pair = self.PAIR_1
        elif self.first == self.PULSE_LONG:
            if length == self.PULSE_MEDIUM:
                pair = self.PAIR_START
            elif length == self.PULSE_SHORT:
                pair = self.PAIR_EOF
        self.process_pair(pair)
        self.first = self.PULSE_NONE

    def reset(self):
        self.sync_count = 0
        if (self.state == self.HEADER and self.repeated_block
                and self.data_size != 0):
            self.state = self.NOT_SYNCED
            self.header_repeated = True
            return
        if self.state == self.DATA and not self.repeated_block:
            self.state = self.NOT_SYNCED
            return
        super().reset()
        self.state = self.NOT_SYNCED
        self.header_repeated = False

    @property
    def name(self) -> str:
        return "CBM"


class TurboTapeLoader(Loader):
    WIDTHS = (208, 320)
    THRESHOLD = 263

    def __init__(self):
        super().__init__()
        self.byte = 0
        self.sync_seq = 9
        self.bitcnt = 0
        self._state = self.NOT_SYNCED
        self.header_type = 0
        self.header = bytearray(192)
        self.data = bytearray(0)

    def reset(self):
        super().reset()
        self.state = self.NOT_SYNCED

    def process_header(self):
        self.start_addr = int.from_bytes(self.header[0:2], 'little')
        self.end_addr = int.from_bytes(self.header[2:4], 'little')
        self.file_name = self.header[5:21].rstrip(b'\0')
        self.unicode_file_name = self.file_name.decode(
                'utf-8', errors='replace')
        self._print("{:16s} {:04X} {:04X}".format(
            self.unicode_file_name, self.start_addr, self.end_addr))
        self.data_size = self.end_addr - self.start_addr

    def process_file(self, checksum):
        sum = np.bitwise_xor.reduce(self.data)
        if sum != checksum:
            self.error("LOAD ERROR")
            # print("checksum {:x}!={:x}".format(sum, checksum))
        else:
            self.success()

    def process_byte(self):
        state = self.state
        if state == self.SYNCING_SEQ:
            if self.byte == 2 and self.sync_seq != 2:
                return
            if self.byte != self.sync_seq:
                self.state = self.NOT_SYNCED
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
                    self._print("LOADING")
                    self.pointer = 0
            else:
                self.error("Bad block type %d" % self.byte)
            return

        if state == self.HEADER:
            self.header[self.pointer] = self.byte
            self.pointer += 1
            if self.pointer == 192:
                self.process_header()
                self.state = self.NOT_SYNCED
            return

        if self.pointer == self.data_size:
            self.process_file(self.byte)
            self.state = self.NOT_SYNCED
            return

        self.data[self.pointer] = self.byte
        self.pointer += 1
        if self.pointer % 16 == 0:
            self.print_progress()

    def process_bit(self, bit: int):
        self.byte = ((self.byte << 1) & 255) | bit

        if self.state == self.NOT_SYNCED:
            if self.byte == 2:
                self.state = self.SYNCING_SEQ
                self.bitcnt = 0
                self.sync_seq = 9
            return

        self.bitcnt += 1
        if self.bitcnt != 8:
            return

        self.bitcnt = 0
        self.process_byte()

    def process_pulse(self, pulse_us: int):
        if self.state >= self.SYNCED and (pulse_us < 160 or pulse_us > 416):
            self.error("Sync lost %d us" % pulse_us)
        else:
            if pulse_us > self.THRESHOLD:
                bit = 1
            else:
                bit = 0
            self.process_bit(bit)

    @property
    def name(self) -> str:
        return "TurboTape64"
