import os
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np


MAGIC = b"LAEB"
VERSION = 1
HEADER_SIZE = 32


@dataclass
class BulkHeader:
    version: int
    expert_count: int
    expert_size_bytes: int


def encode_header(expert_count: int, expert_size_bytes: int, version: int = VERSION) -> bytes:
    reserved = bytes(HEADER_SIZE - 16)
    return MAGIC + struct.pack("<B3xII", version, expert_count, expert_size_bytes) + reserved


def read_header(path: str) -> Optional[BulkHeader]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
    if len(header) < HEADER_SIZE or header[:4] != MAGIC:
        return None
    version, expert_count, expert_size_bytes = struct.unpack("<B3xII", header[4:16])
    return BulkHeader(version=version, expert_count=expert_count, expert_size_bytes=expert_size_bytes)


def create_bulk_file(path: str, expert_count: int, expert_size_bytes: int, seed: int = 0) -> BulkHeader:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rng = np.random.default_rng(seed)
    header = encode_header(expert_count, expert_size_bytes)
    with open(path, "wb") as f:
        f.write(header)
        for _ in range(expert_count):
            data = rng.integers(0, 256, size=expert_size_bytes, dtype=np.uint8)
            f.write(data.tobytes())
    return BulkHeader(version=VERSION, expert_count=expert_count, expert_size_bytes=expert_size_bytes)


def write_packed_experts(path: str, packed_experts: np.ndarray) -> BulkHeader:
    expert_count, expert_size_bytes = packed_experts.shape
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    header = encode_header(expert_count, expert_size_bytes)
    with open(path, "wb") as f:
        f.write(header)
        f.write(packed_experts.tobytes())
    return BulkHeader(version=VERSION, expert_count=expert_count, expert_size_bytes=expert_size_bytes)
