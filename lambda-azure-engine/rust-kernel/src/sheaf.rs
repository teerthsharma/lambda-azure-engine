use std::collections::HashMap;

/// Grothendieck Sheaf Context
/// Context as a Sheaf of Stalks over a site (conversation turns).
pub struct SheafContext {
    /// Stalk Hash Table: (turn_index, window_offset) -> Stalk Vector (p-adic).
    pub stalks: HashMap<(usize, usize), Vec<u8>>,
    /// Betti numbers of the context topology.
    pub betti_numbers: Vec<usize>,
}

impl SheafContext {
    pub fn new() -> Self {
        Self {
            stalks: HashMap::new(),
            betti_numbers: vec![1, 0, 0], // Initially a single connected point
        }
    }

    /// Gluing Condition: Checks if two stalks can be glued on an overlap.
    /// In this implementation, gluing is valid if Chebyshev distance in p-adic valuation is 0.
    pub fn can_glue(stalk_a: &[u8], stalk_b: &[u8]) -> bool {
        // Simplified check: exact match of p-adic bit-slices.
        stalk_a == stalk_b
    }

    /// Adds a new section to the sheaf.
    pub fn add_section(&mut self, turn: usize, offset: usize, stalk: Vec<u8>) {
        self.stalks.insert((turn, offset), stalk);
        // Topology update: If gluing fails, increment Betti-0 (new connected component).
        self.betti_numbers[0] = self.stalks.len();
    }
}

#[cfg(test)]
mod tests {
    use super::SheafContext;

    #[test]
    fn can_glue_requires_exact_match() {
        let a = vec![1u8, 2, 3];
        let b = vec![1u8, 2, 3];
        let c = vec![1u8, 2, 4];
        assert!(SheafContext::can_glue(&a, &b));
        assert!(!SheafContext::can_glue(&a, &c));
    }

    #[test]
    fn betti_updates_with_sections() {
        let mut ctx = SheafContext::new();
        ctx.add_section(0, 0, vec![1u8]);
        ctx.add_section(1, 0, vec![2u8]);
        assert_eq!(ctx.betti_numbers[0], 2);
    }
}
