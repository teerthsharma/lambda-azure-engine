# Lambda Azure Engine (LAE)

Lambda Azure Engine is an experimental sandbox for exploring **ternary quantization**, **toy Mixture‑of‑Experts routing**, and **bit‑packed kernels**. It is **not** a full inference engine and it does **not** run a 14B/200B model. Many numbers and terms in earlier docs were aspirational; this README describes what actually exists in the repo today.

## What this repo contains

### Python prototypes (`lambda-azure-engine/python-bridge`)
- **`lambda_engine.py`**: A toy end‑to‑end pipeline that wires KV‑cache homology, PQ routing, expert paging, and ternary GEMM.
- **`expert_paging.py`**: NVMe → pinned‑memory → VRAM prefetch flow with CPU fallback and bulk header parsing.
- **`kv_cache_homology.py`**: Union‑Find clustering over key vectors (L2 distance) for “KV compression.”
- **`quantize_to_ternary.py`**: Ternary Weight Networks (TWN) quantization and 2‑bit packing utilities.
- **`triton_ternary_gemm.py`**: Packed ternary GEMM with a CPU reference path and optional Triton kernel.
- **`lae_math.py`**: Shared p‑adic, perfectoid tilt, and sheaf helpers.
- **`integration_test.py`**: Smoke test that exercises the pipeline.

### Rust kernel (`lambda-azure-engine/rust-kernel`)
- **Bit‑packed ternary math** (`lattice_compute.rs`).
- **Toy p‑adic weight helpers** (`p_adic.rs`) and a simple “tilt” mapping (`perfectoid.rs`).
- **Mock boundary/paging structures** (`boundary.rs`, `bulk_space.rs`) that simulate large‑file access.
- **Union‑Find for connected components** (`homology.rs`).

## What this repo does **not** contain
- A real 14B/200B model or a production MoE inference pipeline.

- End‑to‑end training, evaluation, or deployment tooling.

If you need a working large‑scale model, this repository is **not** that; it is a collection of small, illustrative experiments.

## Scope (what runs end‑to‑end)
The current pipeline is intentionally small but fully executable on CPU:
1. **KV‑cache homology**: Union‑Find clustering on key vectors (`kv_cache_homology.py`).
2. **MoE routing**: PQ‑style hashing over random hyperplanes (`lambda_engine.py`).
3. **Expert paging**: Memory‑mapped bulk file with a compact header (`expert_paging.py`).
4. **Ternary compute**: Packed ternary GEMM with bitwise CPU reference (`triton_ternary_gemm.py`).
5. **p‑adic / perfectoid / sheaf**: Executable helper math in `lae_math.py`.

## Pipeline boundaries (Python ↔ Rust)
- **Python**: prepares packed ternary expert weights, runs routing + KV compression, and exercises the ternary GEMM.
- **Rust**: owns the memory‑mapped bulk representation and bit‑packed ternary kernel.
- **Boundary format**: `LAEB` bulk file header (versioned, `expert_count`, `expert_size_bytes`) followed by contiguous expert blobs.

## Implemented formulas (inputs → outputs, tolerance)
- **p‑adic valuation**: digits → smallest non‑zero trit index (`lae_math.padic_valuation`, `p_adic.rs`), integer‑exact.
- **p‑adic distance**: digits → `3^(-v_p(x-y))` (`lae_math.padic_distance`, `p_adic.rs`), float tolerance `1e‑8`.
- **Perfectoid tilt**: digits → coefficients in `F_3[t]` (`lae_math.perfectoid_tilt`, `perfectoid.rs`), integer‑exact.
- **Sheaf gluing**: stalk vectors → boolean (`lae_math.SheafContext.can_glue`, `sheaf.rs`), exact equality.
- **Ternary packing**: int8 values → uint32 packed words (`triton_ternary_gemm.py`), bit‑exact.
- **Ternary dot product**: packed words → int32 accumulation (`lattice_compute.rs`, `triton_ternary_gemm.py`), integer‑exact.
- **KV homology**: key vectors → compressed keys + Betti‑0 (`kv_cache_homology.py`, `homology.rs`), float tolerance `1e‑6`.

## System hygiene
- **Config**: `lambda_engine.py` exposes CLI flags for model size, layers, and bulk path.
- **Logging**: Python pipeline uses `logging` with consistent prefixes.
- **Safe I/O**: bulk header validation avoids accidental huge files.
- **CPU/GPU fallback**: GPU is optional; CPU paths are explicit in the engine and pager.

## Quick start (toy demos)
Suggested entry points:
- `python lambda-azure-engine/python-bridge/lambda_engine.py --cpu`
- `python lambda-azure-engine/python-bridge/kv_cache_homology.py`
- `python lambda-azure-engine/python-bridge/triton_ternary_gemm.py`
- `python -m unittest discover lambda-azure-engine/python-bridge/tests`

## Notes and caveats
- The Colab notebook generator (`build_colab.py`) assumes large downloads and long‑running steps; treat it as a demo script.

## Directory layout
- `lambda-azure-engine/python-bridge/`: Python prototypes and demos.
- `lambda-azure-engine/rust-kernel/`: Rust crate with toy kernels and helpers.
- `200B_Scaling_Plan.md`: Aspirational notes; not an implemented plan.
