# Lambda Azure Engine (LAE)

Lambda Azure Engine is an experimental sandbox for exploring **ternary quantization**, **toy Mixture‑of‑Experts routing**, and **bit‑packed kernels**. It is **not** a full inference engine and it does **not** run a 14B/200B model. Many numbers and terms in earlier docs were aspirational; this README describes what actually exists in the repo today.

## What this repo contains

### Python prototypes (`lambda-azure-engine/python-bridge`)
- **`lambda_engine.py`**: A toy “engine” that simulates routing and layer execution with random tensors.
- **`expert_paging.py`**: A mock NVMe → pinned‑memory → VRAM prefetch flow using `numpy.memmap` and CUDA streams.
- **`kv_cache_homology.py`**: A simple Union‑Find clustering demo over key vectors (L2 distance), used as a stand‑in for “KV compression.”
- **`quantize_to_ternary.py`**: Ternary Weight Networks (TWN) quantization and 2‑bit packing utilities.
- **`triton_ternary_gemm.py`**: Packed ternary GEMM with a CPU reference path and an optional Triton kernel.
- **`train_ternary_moe.py`**: A minimal MoE training stub on dummy data.
- **`integration_test.py`**: A small smoke test that exercises the toy engine.

### Rust kernel (`lambda-azure-engine/rust-kernel`)
- **Bit‑packed ternary math** (`lattice_compute.rs`).
- **Toy p‑adic weight helpers** (`p_adic.rs`) and a simple “tilt” mapping (`perfectoid.rs`).
- **Mock boundary/paging structures** (`boundary.rs`, `bulk_space.rs`) that simulate large‑file access.
- **Union‑Find for connected components** (`homology.rs`).

## What this repo does **not** contain
- A real 14B/200B model or a production MoE inference pipeline.
- A mathematically rigorous implementation of p‑adic, perfectoid, or holographic methods.
- End‑to‑end training, evaluation, or deployment tooling.

If you need a working large‑scale model, this repository is **not** that; it is a collection of small, illustrative experiments.

## Quick start (toy demos)
These scripts are designed for experimentation and may require a CUDA‑capable system and `torch`. The Triton kernel is optional.

Suggested entry points:
- `python lambda-azure-engine/python-bridge/kv_cache_homology.py`
- `python lambda-azure-engine/python-bridge/triton_ternary_gemm.py`
- `python lambda-azure-engine/python-bridge/quantize_to_ternary.py`

## Notes and caveats
- `rust-kernel/src/main.rs` attempts to create a very large file if you run it as‑is. Review it first if disk usage matters.
- The Colab notebook generator (`build_colab.py`) assumes large downloads and long‑running steps; treat it as a demo script, not a reproducible pipeline.

## Directory layout
- `lambda-azure-engine/python-bridge/`: Python prototypes and demos.
- `lambda-azure-engine/rust-kernel/`: Rust crate with toy kernels and helpers.
- `200B_Scaling_Plan.md`: Aspirational notes; not an implemented plan.
