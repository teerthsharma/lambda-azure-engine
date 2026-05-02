import argparse
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch

from bulk_format import write_packed_experts
from expert_paging import ExpertPager
from kv_cache_homology import cluster_and_compress
from lae_math import (
    SheafContext,
    encode_trits,
    padic_distance,
    perfectoid_tilt,
)
from triton_ternary_gemm import pack_ternary, ternary_gemm_bitwise_cpu


logger = logging.getLogger("lae")


@dataclass
class EngineConfig:
    d_model: int = 64
    n_experts: int = 4
    n_layers: int = 2
    kv_threshold: float = 0.2
    seed: int = 42
    bulk_path: str = "lae_bulk.bin"
    use_cuda: bool | None = None

    @property
    def k_packed(self) -> int:
        return (self.d_model + 15) // 16

    @property
    def expert_size_bytes(self) -> int:
        return self.k_packed * self.d_model * 4


class LambdaAzureEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = "cuda" if (config.use_cuda is not False and torch.cuda.is_available()) else "cpu"
        self.use_cuda = self.device == "cuda"

        rng = np.random.default_rng(config.seed)
        hyperplanes = max(1, int(np.log2(config.n_experts)))
        self.pq_hyperplanes = rng.standard_normal((config.d_model, hyperplanes))

        self.sheaf = SheafContext()
        self._ensure_bulk_file(rng)
        self.pager = ExpertPager(
            config.bulk_path,
            config.expert_size_bytes,
            config.n_experts,
            use_cuda=self.use_cuda,
        )

    def _ensure_bulk_file(self, rng: np.random.Generator):
        if os.path.exists(self.config.bulk_path):
            return

        logger.info("Bulk file missing; creating a compact packed expert file.")
        k = self.config.d_model
        k_packed = self.config.k_packed
        packed_experts = np.zeros(
            (self.config.n_experts, self.config.expert_size_bytes), dtype=np.uint8
        )

        for i in range(self.config.n_experts):
            weights = rng.choice([-1, 0, 1], size=(k, k)).astype(np.int8)
            packed = np.zeros((k_packed, k), dtype=np.uint32)
            for col in range(k):
                packed[:, col] = pack_ternary(weights[:, col])
            packed_experts[i] = np.frombuffer(packed.tobytes(), dtype=np.uint8)

        write_packed_experts(self.config.bulk_path, packed_experts)

    def tokenize(self, text):
        return [ord(c) % 256 for c in text][: max(1, len(text) // 2)]

    def embedding_lookup(self, tokens):
        return torch.randn(1, len(tokens), self.config.d_model, device=self.device)

    def pq_routing(self, hidden_state):
        batch, seq, _ = hidden_state.shape
        flat_x = hidden_state.view(-1, self.config.d_model).cpu().numpy()
        bits = (np.dot(flat_x, self.pq_hyperplanes) > 0).astype(int)
        expert_ids = np.zeros(batch * seq, dtype=int)
        for i in range(bits.shape[1]):
            expert_ids += bits[:, i] * (2**i)
        return expert_ids

    def _padic_metrics(self, keys: np.ndarray) -> float:
        if keys.shape[0] < 2:
            return 0.0
        trits_a = np.where(keys[0] > 0, 1, np.where(keys[0] < 0, 2, 0))
        trits_b = np.where(keys[1] > 0, 1, np.where(keys[1] < 0, 2, 0))
        digits_a = encode_trits(trits_a)
        digits_b = encode_trits(trits_b)
        return padic_distance(digits_a, digits_b)

    def _load_expert_weights(self, expert_id: int) -> np.ndarray:
        expert_tensor, stream = self.pager.async_prefetch_expert(expert_id, 0)
        if stream is not None:
            stream.synchronize()
        expert_bytes = expert_tensor.cpu().numpy().tobytes()
        packed = np.frombuffer(expert_bytes, dtype=np.uint32)
        return packed.reshape(self.config.k_packed, self.config.d_model)

    def generate(self, prompt, max_len=10):
        logger.info("Generating for prompt: %s", prompt)
        tokens = self.tokenize(prompt)[:max_len]
        x = self.embedding_lookup(tokens)

        for layer in range(self.config.n_layers):
            x = x + torch.randn_like(x) * 0.05
            keys = x[0].detach().cpu().numpy()
            padic_scale = 1.0 + self._padic_metrics(keys)
            kv_threshold = self.config.kv_threshold * padic_scale

            compressed_keys = cluster_and_compress(keys, threshold=kv_threshold)
            self.sheaf.add_section(layer, 0, compressed_keys)

            expert_ids = self.pq_routing(x)
            tilt = perfectoid_tilt(encode_trits(np.where(keys[0] > 0, 1, np.where(keys[0] < 0, 2, 0))))
            tilt_bias = int(np.sum(tilt)) % self.config.n_experts
            expert_id = int((expert_ids[0] + tilt_bias) % self.config.n_experts)

            weights = self._load_expert_weights(expert_id)
            activations = torch.sign(x[0]).cpu().numpy().astype(np.int8)
            a_packed = np.zeros((activations.shape[0], self.config.k_packed), dtype=np.uint32)
            for m in range(activations.shape[0]):
                a_packed[m] = pack_ternary(activations[m])

            output = ternary_gemm_bitwise_cpu(
                a_packed,
                weights,
                activations.shape[0],
                self.config.d_model,
                self.config.k_packed,
            )

            x = x + torch.tensor(output, device=self.device, dtype=x.dtype).unsqueeze(0) * 0.01

        return f"Simulated output ({len(tokens)} tokens, betti-0={self.sheaf.betti_numbers[0]})"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Lambda Azure Engine pipeline.")
    parser.add_argument("--prompt", default="Hello world")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--bulk-path", default="lae_bulk.bin")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = EngineConfig(
        d_model=args.d_model,
        n_experts=args.experts,
        n_layers=args.layers,
        bulk_path=args.bulk_path,
        use_cuda=False if args.cpu else None,
        seed=args.seed,
    )
    engine = LambdaAzureEngine(config)
    print(engine.generate(args.prompt))
