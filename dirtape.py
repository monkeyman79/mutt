import sys
import struct
from typing import List


class T64Entry:

    def __init__(self, entry_bytes: bytes):
        used: int
        self.FileType: int
        self.StartAddr: int
        self.EndAddr: int
        self.TapePos: int
        name_bytes: bytes
        (used, self.FileType, self.StartAddr, self.EndAddr, self.TapePos,
            name_bytes) = struct.unpack("<BBHHxxIxxxx16s", entry_bytes)
        self.IsUsed = (used != 0)
        self.Name = name_bytes.decode('ascii', errors='replace')


class T64File:

    def __init__(self, file_name: str):
        with open(file_name, "rb") as file:
            format_descr = bytearray(32)
            format_descr_string = b'C64 tape image file'
            format_descr[:len(format_descr_string)] = format_descr_string
            self.HeaderBytes = bytearray(64)
            self.HeaderBytes[:] = file.read(64)
            tape_descr_bytes: bytes
            name_bytes: bytes
            self.Version: int
            self.MaxFiles: int
            self.CurrFiles: int
            (tape_descr_bytes, self.Version, self.MaxFiles, self.CurrFiles,
                name_bytes) = struct.unpack(
                    "<32sHHHxx24s", self.HeaderBytes)
            if tape_descr_bytes != format_descr:
                raise ValueError("This it not T64 file")
            self.Name = name_bytes.decode('ascii', errors='replace')
            self.IndexBytes = bytearray(32 * self.MaxFiles)
            self.IndexBytes[:] = file.read(len(self.IndexBytes))
            self.Entries: List[T64Entry] = []
            for index in range(self.MaxFiles):
                offset = index * 32
                entry = T64Entry(self.IndexBytes[offset:offset+32])
                self.Entries.append(entry)


def dirtape(file_name: str):
    tape = T64File(file_name)
    print(tape.Name)
    used_count = 0
    for entry in tape.Entries:
        if entry.IsUsed:
            used_count += 1
            print("{:16s} {:04X} {:04X} {:}".format(
                  entry.Name, entry.StartAddr, entry.EndAddr, entry.FileType))
    print("{:} Entries".format(used_count))


dirtape(sys.argv[1])
