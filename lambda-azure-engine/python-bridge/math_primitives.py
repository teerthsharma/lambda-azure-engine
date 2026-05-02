from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


def padic_quantize(values: np.ndarray, p: int = 3, scale: int = 10000) -> np.ndarray:
    """Quantize real values into {-1, 0, 1} using p-adic divisibility."""
    w_int = np.round(values * scale).astype(np.int32)
    ternary = np.where(w_int % p == 0, 0, np.sign(w_int))
    return ternary.astype(np.int8)


def ternary_to_digits(ternary: np.ndarray) -> np.ndarray:
    """Map {-1, 0, 1} to {2, 0, 1} digits for balanced ternary storage."""
    digits = ternary.astype(np.int8)
    digits = np.where(digits == -1, 2, digits)
    return digits.astype(np.uint8)


def trits_from_vector(vector: Sequence[float], n_trits: int = 4, zero_threshold: float = 1e-3) -> List[int]:
    """Project a vector to the first n_trits balanced ternary digits."""
    digits: List[int] = []
    for value in list(vector)[:n_trits]:
        if abs(float(value)) <= zero_threshold:
            digits.append(0)
        elif value > 0:
            digits.append(1)
        else:
            digits.append(2)
    if len(digits) < n_trits:
        digits.extend([0] * (n_trits - len(digits)))
    return digits


@dataclass
class PAdicWeight:
    digits: List[int]

    @classmethod
    def from_trits(cls, trits: Iterable[int], n_trits: int = 4) -> "PAdicWeight":
        digits = list(trits)[:n_trits]
        if len(digits) < n_trits:
            digits.extend([0] * (n_trits - len(digits)))
        for digit in digits:
            if digit not in (0, 1, 2):
                raise ValueError(f"Invalid trit digit: {digit}")
        return cls(digits)

    def valuation(self) -> int:
        for idx, digit in enumerate(self.digits):
            if digit != 0:
                return idx
        return len(self.digits)

    def to_i32(self) -> int:
        value = 0
        power = 1
        for digit in self.digits:
            trit_val = 1 if digit == 1 else -1 if digit == 2 else 0
            value += trit_val * power
            power *= 3
        return value

    def packed_u8(self) -> int:
        packed = 0
        for i, digit in enumerate(self.digits):
            packed |= (digit & 0b11) << (i * 2)
        return packed


class PAdicLattice:
    @staticmethod
    def subtract(x: PAdicWeight, y: PAdicWeight) -> PAdicWeight:
        carry = 0
        diff_digits: List[int] = []
        for digit_x, digit_y in zip(x.digits, y.digits):
            vx = 1 if digit_x == 1 else -1 if digit_x == 2 else 0
            vy = 1 if digit_y == 1 else -1 if digit_y == 2 else 0
            diff = vx - vy - carry
            carry = 0
            while diff < -1:
                diff += 3
                carry -= 1
            while diff > 1:
                diff -= 3
                carry += 1
            out_trit = 1 if diff == 1 else 2 if diff == -1 else 0
            diff_digits.append(out_trit)
        return PAdicWeight(diff_digits)

    @staticmethod
    def distance(x: PAdicWeight, y: PAdicWeight, p: float = 3.0) -> float:
        diff = PAdicLattice.subtract(x, y)
        return float(p) ** (-diff.valuation())


@dataclass
class PerfectoidTilt:
    coefficients: List[int]

    @classmethod
    def tilt(cls, weight: PAdicWeight) -> "PerfectoidTilt":
        return cls(list(weight.digits))

    def multiply(self, other: "PerfectoidTilt", max_degree: int = 8) -> "PerfectoidTilt":
        res = [0] * max_degree
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                if i + j < max_degree:
                    res[i + j] = (res[i + j] + (a * b) % 3) % 3
        return PerfectoidTilt(res)

    def hash_base3(self) -> int:
        value = 0
        power = 1
        for coeff in self.coefficients:
            value += coeff * power
            power *= 3
        return value


@dataclass
class SheafContext:
    stalks: dict
    betti_numbers: List[int]

    @classmethod
    def new(cls) -> "SheafContext":
        return cls(stalks={}, betti_numbers=[1, 0, 0])

    @staticmethod
    def can_glue(stalk_a: Sequence[int], stalk_b: Sequence[int]) -> bool:
        return list(stalk_a) == list(stalk_b)

    def add_section(self, turn: int, offset: int, stalk: Sequence[int]) -> None:
        self.stalks[(turn, offset)] = list(stalk)
        self.betti_numbers[0] = len(self.stalks)
