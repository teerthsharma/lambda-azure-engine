# Scaling Notes (Aspirational, Not Implemented)

This repository does **not** implement a 14B or 200B model. The earlier “200B scaling” write‑up was speculative and read like a finished system. This document is a **brainstorming note** only.

## Current reality
- The codebase is a set of toy experiments for ternary packing, mock paging, and simple routing.
- There is no trained large‑scale model in this repo.
- Any references to specific VRAM budgets or parameter counts are placeholders.

## If real scaling were attempted
The work would look like ordinary large‑model engineering, not the “guaranteed” claims in the previous text:

1. **Data and training pipeline**
   - Build or adopt a large‑scale dataset pipeline.
   - Implement distributed training and evaluation.

2. **Model architecture**
   - Define a real MoE architecture and verify quality/latency trade‑offs.
   - Establish a router, load balancing, and fallback behavior.

3. **Quantization and packing**
   - Measure accuracy impacts of ternary or low‑bit schemes.
   - Build safe, tested kernels for packing/unpacking and inference.

4. **Storage and paging**
   - Engineer real storage formats and IO pipelines.
   - Validate end‑to‑end performance on actual hardware.

## Why keep this file
It keeps future discussions grounded: scaling to 200B is **not implemented here** and would require conventional, well‑validated engineering work.
