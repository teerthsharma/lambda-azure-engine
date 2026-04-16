/// p-adic Valuational Logic (p=3)
/// Weights are truncated p-adic integers in balanced ternary.
pub struct PAdicWeight {
    /// 4 trits per weight, 2 bits each = 8 bits.
    /// Digits: 0=0, 1=1, 2=-1
    pub digits: u8,
}

impl PAdicWeight {
    /// Computes the p-adic valuation v_p(w).
    /// Smallest i such that a_i != 0.
    pub fn valuation(&self) -> u32 {
        for i in 0..4 {
            let trit = (self.digits >> (i * 2)) & 0b11;
            if trit != 0 {
                return i;
            }
        }
        4 // Zero weight has valuation infinity (truncated to 4)
    }

    /// Balanced ternary value reconstruction for debugging.
    pub fn to_i32(&self) -> i32 {
        let mut val = 0;
        let mut power = 1;
        for i in 0..4 {
            let trit = (self.digits >> (i * 2)) & 0b11;
            let trit_val = match trit {
                1 => 1,
                2 => -1,
                _ => 0,
            };
            val += trit_val * power;
            power *= 3;
        }
        val
    }
}

pub struct PAdicLattice;

impl PAdicLattice {
    pub fn subtract(x: &PAdicWeight, y: &PAdicWeight) -> PAdicWeight {
        let mut diff_digits = 0;
        let mut carry = 0;
        
        for i in 0..4 {
            let shift = i * 2;
            let trit_x = (x.digits >> shift) & 0b11;
            let trit_y = (y.digits >> shift) & 0b11;
            
            let vx = match trit_x { 1 => 1, 2 => -1, _ => 0 };
            let vy = match trit_y { 1 => 1, 2 => -1, _ => 0 };
            
            let mut diff = vx - vy - carry;
            carry = 0;
            
            while diff < -1 {
                diff += 3;
                carry -= 1;
            }
            while diff > 1 {
                diff -= 3;
                carry += 1;
            }
            
            let out_trit = match diff {
                1 => 1,
                -1 => 2,
                _ => 0,
            };
            
            diff_digits |= out_trit << shift;
        }
        
        PAdicWeight { digits: diff_digits }
    }

    /// Computes correct p-adic distance d_p(x, y) = p^(-v_p(x - y)).
    pub fn distance(x: &PAdicWeight, y: &PAdicWeight) -> f64 {
        let diff = Self::subtract(x, y);
        (3.0f64).powi(-(diff.valuation() as i32))
    }
}
