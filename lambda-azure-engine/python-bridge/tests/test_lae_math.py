import unittest

import numpy as np

from lae_math import SheafContext, encode_trits, padic_distance, padic_valuation, perfectoid_tilt
from triton_ternary_gemm import pack_ternary, unpack_ternary


class TestLaeMath(unittest.TestCase):
    def test_padic_valuation(self):
        digits = encode_trits(np.array([0, 0, 1, 0]))
        self.assertEqual(padic_valuation(digits), 2)

    def test_padic_distance(self):
        digits_a = encode_trits(np.array([0, 1, 0, 0]))
        digits_b = encode_trits(np.array([0, 1, 0, 0]))
        self.assertAlmostEqual(padic_distance(digits_a, digits_b), 3 ** -4)

    def test_perfectoid_tilt(self):
        digits = encode_trits(np.array([1, 2, 0, 1]))
        coeffs = perfectoid_tilt(digits)
        self.assertEqual(coeffs.tolist(), [1, 2, 0, 1])

    def test_sheaf_gluing(self):
        sheaf = SheafContext()
        stalk_a = np.array([1, 2, 3], dtype=np.uint8)
        stalk_b = np.array([1, 2, 3], dtype=np.uint8)
        self.assertTrue(sheaf.can_glue(stalk_a, stalk_b))
        sheaf.add_section(0, 0, stalk_a)
        self.assertEqual(sheaf.betti_numbers[0], 1)

    def test_pack_unpack_roundtrip(self):
        rng = np.random.default_rng(0)
        values = rng.choice([-1, 0, 1], size=33).astype(np.int8)
        packed = pack_ternary(values)
        unpacked = unpack_ternary(packed, len(values))
        np.testing.assert_array_equal(unpacked, values)


if __name__ == "__main__":
    unittest.main()
