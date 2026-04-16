/// LAE-v4 Hardened Bit-Lattice Compute
/// Replaces metaphors with exact bit-manipulation for ternary weights.

pub struct TernaryKernel;

impl TernaryKernel {
    /// Ternary Dot Product (XOR-Sum)
    /// Processes 32 balanced ternary weights {-1, 0, 1} in a single word.
    /// W: Packed weights (2-bit per param)
    /// X: Packed activations (2-bit per param)
    #[inline(always)]
    pub fn dot_product_32(w: u64, x: u64) -> i32 {
        // Presence mask (bit 0, 2, 4...)
        let w_p = w & 0x5555555555555555;
        let x_p = x & 0x5555555555555555;
        
        // Sign mask (bit 1, 3, 5...)
        // Shifted down to align with presence bits for bitwise ops
        let w_s = (w >> 1) & 0x5555555555555555;
        let x_s = (x >> 1) & 0x5555555555555555;

        // Intersection: both are non-zero
        let active = w_p & x_p;
        
        // Positive match: (W+ and X+) or (W- and X-)
        // Same sign bits (XOR = 0)
        let pos = active & !(w_s ^ x_s);
        
        // Negative match: (W+ and X-) or (W- and X+)
        // Different sign bits (XOR = 1)
        let neg = active & (w_s ^ x_s);

        (pos.count_ones() as i32) - (neg.count_ones() as i32)
    }

    /// Batch XOR-Sum: Processes a row of 32-param words.
    pub fn batch_dot(w_row: &[u64], x_vec: &[u64]) -> i32 {
        w_row.iter().zip(x_vec.iter())
            .map(|(&w, &x)| Self::dot_product_32(w, x))
            .sum()
    }
}
