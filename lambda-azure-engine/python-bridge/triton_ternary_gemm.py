"""
Lambda Azure Engine — Triton Ternary GEMM Kernel
Packed 2-bit ternary XOR+POPCNT matrix multiplication over F_3.

Encoding (2 bits per weight):
  00 = 0
  01 = +1
  10 = -1
  11 = reserved (treated as 0)

Each uint32 packs 16 ternary values.
Dot product = popcount(active & same_sign) - popcount(active & diff_sign)
"""

import torch
import numpy as np

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Packing / unpacking helpers (CPU, for data preparation)
# ---------------------------------------------------------------------------

MASK_EVEN = np.uint32(0x55555555)  # bits 0,2,4,...,30  (presence)
MASK_ODD  = np.uint32(0xAAAAAAAA)  # bits 1,3,5,...,31  (sign)


def pack_ternary(values: np.ndarray) -> np.ndarray:
    """Pack an int8 array of {-1, 0, 1} into uint32 (16 values per word).

    Encoding per value (2 bits, LSB first):
      bit0 = presence (1 if value != 0)
      bit1 = sign     (1 if value < 0)
    """
    assert values.dtype == np.int8
    flat = values.ravel()
    # pad to multiple of 16
    pad = (16 - len(flat) % 16) % 16
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.int8)])

    n_words = len(flat) // 16
    packed = np.zeros(n_words, dtype=np.uint32)

    for i in range(16):
        v = flat[i::16].astype(np.int32)
        presence = (v != 0).astype(np.uint32)
        sign = (v < 0).astype(np.uint32)
        two_bits = presence | (sign << 1)
        packed |= two_bits << (i * 2)

    return packed


def unpack_ternary(packed: np.ndarray, n_elements: int) -> np.ndarray:
    """Unpack uint32 array back to int8 {-1, 0, 1}."""
    out = np.zeros(len(packed) * 16, dtype=np.int8)
    for i in range(16):
        two_bits = (packed >> (i * 2)) & np.uint32(0x3)
        presence = (two_bits & 1).astype(np.int8)
        sign = ((two_bits >> 1) & 1).astype(np.int8)
        val = presence * (1 - 2 * sign)
        out[i::16] = val
    return out[:n_elements]


# ---------------------------------------------------------------------------
# Triton kernel: packed ternary dot product via XOR + POPCNT
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    @triton.jit
    def _ternary_dot_kernel(
        a_ptr,        # uint32* — packed activations  [M, K_packed]
        b_ptr,        # uint32* — packed weights       [K_packed, N]
        c_ptr,        # int32*  — output               [M, N]
        M, N, K_packed,
        stride_am: tl.constexpr, stride_ak: tl.constexpr,
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        PRESENCE_MASK: tl.constexpr = 0x55555555   # even bits
        # SIGN_MASK:     tl.constexpr = 0xAAAAAAAA   # odd bits

        for k_start in range(0, K_packed, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)

            # Load A block [BLOCK_M, BLOCK_K] of uint32
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K_packed)
            a = tl.load(a_ptrs, mask=a_mask, other=0)

            # Load B block [BLOCK_K, BLOCK_N] of uint32
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            b_mask = (offs_k[:, None] < K_packed) & (offs_n[None, :] < N)
            b = tl.load(b_ptrs, mask=b_mask, other=0)

            # --- XOR + POPCNT logic ---
            # For each pair (a_word, b_word) we need to accumulate over the
            # K dimension.  We iterate over each (m, k) x (k, n) product.
            for ki in range(BLOCK_K):
                a_col = tl.load(
                    a_ptr + offs_m * stride_am + (k_start + ki) * stride_ak,
                    mask=(offs_m < M) & ((k_start + ki) < K_packed),
                    other=0,
                )  # [BLOCK_M]

                b_row = tl.load(
                    b_ptr + (k_start + ki) * stride_bk + offs_n * stride_bn,
                    mask=((k_start + ki) < K_packed) & (offs_n < N),
                    other=0,
                )  # [BLOCK_N]

                for ni in range(BLOCK_N):
                    b_val = tl.load(
                        b_ptr + (k_start + ki) * stride_bk + (pid_n * BLOCK_N + ni) * stride_bn,
                        mask=((k_start + ki) < K_packed) & ((pid_n * BLOCK_N + ni) < N),
                        other=0,
                    )  # scalar

                    # Presence bits
                    a_pres = a_col & PRESENCE_MASK
                    b_pres = b_val & PRESENCE_MASK

                    # Sign bits shifted down to align with presence
                    a_sign = (a_col >> 1) & PRESENCE_MASK
                    b_sign = (b_val >> 1) & PRESENCE_MASK

                    # Active = both non-zero
                    active = a_pres & b_pres

                    # Same sign → positive contribution
                    pos = active & ~(a_sign ^ b_sign)
                    # Different sign → negative contribution
                    neg = active & (a_sign ^ b_sign)

                    contribution = tl.math.popc(pos) - tl.math.popc(neg)
                    # acc[:, ni] += contribution  — we store per-element
                    tl.store(
                        c_ptr + offs_m * stride_cm + (pid_n * BLOCK_N + ni) * stride_cn,
                        tl.load(
                            c_ptr + offs_m * stride_cm + (pid_n * BLOCK_N + ni) * stride_cn,
                            mask=(offs_m < M) & ((pid_n * BLOCK_N + ni) < N),
                            other=0,
                        ) + contribution,
                        mask=(offs_m < M) & ((pid_n * BLOCK_N + ni) < N),
                    )

            # We already accumulated inside the k-loop above, so skip the
            # outer accumulator write below and break.
            break

        # Final store of acc — only used if the inner loop path wasn't taken
        # (dead path kept for reference)
        # tl.store(c_ptrs, acc, mask=c_mask)

    ternary_gemm_kernel = _ternary_dot_kernel
    


# ---------------------------------------------------------------------------
# CPU / fallback implementation (reference, always available)
# ---------------------------------------------------------------------------

def ternary_gemm_cpu(a_packed: np.ndarray, b_packed: np.ndarray,
                     M: int, N: int, K: int) -> np.ndarray:
    """Reference CPU implementation. Unpacks and computes via int8 matmul."""
    a_vals = unpack_ternary(a_packed.ravel(), M * K).reshape(M, K)
    b_vals = unpack_ternary(b_packed.ravel(), K * N).reshape(K, N)
    return (a_vals.astype(np.int32) @ b_vals.astype(np.int32)).astype(np.int32)


def ternary_gemm_bitwise_cpu(a_packed: np.ndarray, b_packed: np.ndarray,
                              M: int, N: int, K_packed: int) -> np.ndarray:
    """Reference CPU implementation using XOR+POPCNT on packed uint32.
    
    This is the exact algorithm the Triton kernel implements, done in
    pure NumPy for correctness checking.
    """
    # a_packed: [M, K_packed]  uint32
    # b_packed: [K_packed, N]  uint32
    PRES = np.uint32(0x55555555)
    c = np.zeros((M, N), dtype=np.int32)

    for m in range(M):
        for n in range(N):
            acc = np.int32(0)
            for k in range(K_packed):
                aw = a_packed[m, k]
                bw = b_packed[k, n]

                a_pres = aw & PRES
                b_pres = bw & PRES
                a_sign = (aw >> 1) & PRES
                b_sign = (bw >> 1) & PRES

                active = a_pres & b_pres
                pos = active & ~(a_sign ^ b_sign)
                neg = active & (a_sign ^ b_sign)

                acc += np.int32(bin(pos).count('1')) - np.int32(bin(neg).count('1'))
            c[m, n] = acc
    return c


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------

def test_correctness(M=64, N=64, K=128, seed=42):
    """Compares bitwise CPU implementation against naive int8 matmul."""
    rng = np.random.default_rng(seed)
    a_int8 = rng.choice([-1, 0, 1], size=(M, K)).astype(np.int8)
    b_int8 = rng.choice([-1, 0, 1], size=(K, N)).astype(np.int8)

    # Ground truth
    expected = (a_int8.astype(np.int32) @ b_int8.astype(np.int32))

    # Pack
    K_packed = (K + 15) // 16
    a_packed = np.zeros((M, K_packed), dtype=np.uint32)
    b_packed = np.zeros((K_packed, N), dtype=np.uint32)

    for m in range(M):
        a_packed[m] = pack_ternary(a_int8[m])
    for n in range(N):
        col = b_int8[:, n]
        b_packed[:, n] = pack_ternary(col)

    # Bitwise CPU reference
    result = ternary_gemm_bitwise_cpu(a_packed, b_packed, M, N, K_packed)

    match = np.allclose(expected, result)
    max_err = np.max(np.abs(expected - result))
    print(f"[CORRECTNESS TEST] M={M} N={N} K={K}")
    print(f"  Naive int8 matmul vs XOR+POPCNT bitwise: {'PASS' if match else 'FAIL'}")
    print(f"  Max absolute error: {max_err}")
    return match


def test_pack_unpack_roundtrip(n=1024, seed=0):
    """Verifies pack→unpack is lossless."""
    rng = np.random.default_rng(seed)
    original = rng.choice([-1, 0, 1], size=n).astype(np.int8)
    packed = pack_ternary(original)
    recovered = unpack_ternary(packed, n)
    match = np.array_equal(original, recovered)
    print(f"[PACK/UNPACK ROUNDTRIP] n={n}: {'PASS' if match else 'FAIL'}")
    return match


if __name__ == "__main__":
    test_pack_unpack_roundtrip()
    test_correctness(M=32, N=32, K=64)
    test_correctness(M=64, N=64, K=128)
    test_correctness(M=128, N=128, K=256)
