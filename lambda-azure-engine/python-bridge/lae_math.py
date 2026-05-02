import numpy as np

from triton_ternary_gemm import pack_ternary, unpack_ternary

P_ADIC_BASE = 3
TRITS_PER_WEIGHT = 4


def encode_trits(trits: np.ndarray, trits_per_weight: int = TRITS_PER_WEIGHT) -> int:
    """Pack trits (0,1,2) into 2-bit digits for p-adic operations."""
    digits = 0
    for i, trit in enumerate(trits[:trits_per_weight]):
        digits |= (int(trit) & 0x3) << (i * 2)
    return digits


def padic_valuation(digits: int, trits_per_weight: int = TRITS_PER_WEIGHT) -> int:
    """Smallest i such that a_i != 0 for a truncated p-adic weight."""
    for i in range(trits_per_weight):
        trit = (digits >> (i * 2)) & 0b11
        if trit != 0:
            return i
    return trits_per_weight


def padic_distance(x_digits: int, y_digits: int, p: int = P_ADIC_BASE) -> float:
    """d_p(x, y) = p^(-v_p(x - y)) for truncated ternary digits."""
    diff = x_digits ^ y_digits
    return float(p) ** (-padic_valuation(diff))


def perfectoid_tilt(digits: int, trits_per_weight: int = TRITS_PER_WEIGHT) -> np.ndarray:
    """Tilt a 3-adic weight into coefficients in F_3[t]."""
    coeffs = np.zeros(trits_per_weight, dtype=np.uint8)
    for i in range(trits_per_weight):
        coeffs[i] = (digits >> (i * 2)) & 0b11
    return coeffs


def apply_padic_valuational_quantization(tensor, p: int = P_ADIC_BASE, scale: int = 10000):
    """
    Maps real weights into {-1, 0, 1} via p-adic divisibility.
    This is the executable proxy for p-adic valuational quantization.
    """
    w = tensor.cpu().float().numpy()
    if len(w.shape) < 2:
        return w
    w_int = np.round(w * scale).astype(np.int32)
    w_padic = np.where(w_int % p == 0, 0, np.sign(w_int))
    return w_padic.astype(np.int8)


class SheafContext:
    """Grothendieck-style context sheaf with simple gluing checks."""

    def __init__(self):
        self.stalks = {}
        self.betti_numbers = [1, 0, 0]

    @staticmethod
    def can_glue(stalk_a: np.ndarray, stalk_b: np.ndarray) -> bool:
        return np.array_equal(stalk_a, stalk_b)

    def add_section(self, turn: int, offset: int, stalk: np.ndarray):
        self.stalks[(turn, offset)] = stalk
        self.betti_numbers[0] = len(self.stalks)
