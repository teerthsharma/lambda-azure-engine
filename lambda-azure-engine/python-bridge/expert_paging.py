import os

import numpy as np
import torch

from bulk_format import HEADER_SIZE, read_header


class ExpertPager:
    def __init__(self, mmap_path, expert_size_bytes, num_experts, use_cuda=None):
        self.mmap_path = mmap_path
        self.use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
        self.data_offset = 0

        header = read_header(mmap_path)
        if header is not None:
            expert_size_bytes = header.expert_size_bytes
            num_experts = header.expert_count
            self.data_offset = HEADER_SIZE

        self.expert_size_bytes = expert_size_bytes
        self.num_experts = num_experts

        if not os.path.exists(mmap_path):
            raise FileNotFoundError(f"Bulk file {mmap_path} not found.")

        self.mmap_data = np.memmap(
            mmap_path,
            dtype="uint8",
            mode="r",
            offset=self.data_offset,
        )

        if self.use_cuda:
            self.buffer_a = torch.empty(expert_size_bytes, dtype=torch.uint8, pin_memory=True)
            self.buffer_b = torch.empty(expert_size_bytes, dtype=torch.uint8, pin_memory=True)
            self.stream_a = torch.cuda.Stream()
            self.stream_b = torch.cuda.Stream()
        else:
            self.buffer_a = torch.empty(expert_size_bytes, dtype=torch.uint8)
            self.buffer_b = torch.empty(expert_size_bytes, dtype=torch.uint8)
            self.stream_a = None
            self.stream_b = None

    def async_prefetch_expert(self, expert_id, buffer_idx):
        """
        Prefetches an expert's weights into pinned CPU memory.
        If CUDA is available, also performs an async DMA copy to VRAM.
        """
        if expert_id >= self.num_experts:
            raise ValueError(f"Invalid expert ID {expert_id}")

        buffer = self.buffer_a if buffer_idx == 0 else self.buffer_b
        stream = self.stream_a if buffer_idx == 0 else self.stream_b

        start = expert_id * self.expert_size_bytes
        end = start + self.expert_size_bytes
        buffer.numpy()[:] = self.mmap_data[start:end]

        if not self.use_cuda:
            return buffer.clone(), None

        with torch.cuda.stream(stream):
            vram_buffer = torch.empty_like(buffer, device="cuda")
            vram_buffer.copy_(buffer, non_blocking=True)

        return vram_buffer, stream


if __name__ == "__main__":
    pager = ExpertPager("dummy_bulk.bin", 1024 * 1024, 8, use_cuda=False)
    print("Initiating async prefetch of expert 2 to buffer 0...")
    vram_buf, stream = pager.async_prefetch_expert(2, 0)
    if stream is not None:
        stream.synchronize()
    print("Prefetch complete.")
