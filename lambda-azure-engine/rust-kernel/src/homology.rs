/// 0-D Persistent Homology (KV-Cache Partitioning)
/// Uses Union-Find to compute connected components in p-adic key-space.
pub struct PersistenceUnionFind {
    pub parent: Vec<usize>,
    pub num_components: usize,
}

impl PersistenceUnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            num_components: n,
        }
    }

    pub fn find(&mut self, i: usize) -> usize {
        if self.parent[i] == i {
            i
        } else {
            self.parent[i] = self.find(self.parent[i]);
            self.parent[i]
        }
    }

    /// Unions two keys if their p-adic distance is below threshold epsilon.
    pub fn union_if_close(&mut self, i: usize, j: usize, distance: f64, epsilon: f64) {
        if distance <= epsilon {
            let root_i = self.find(i);
            let root_j = self.find(j);
            if root_i != root_j {
                self.parent[root_i] = root_j;
                self.num_components -= 1;
            }
        }
    }

    /// Betti-0: Number of connected components at threshold epsilon.
    pub fn betti_0(&self) -> usize {
        self.num_components
    }
}

#[cfg(test)]
mod tests {
    use super::PersistenceUnionFind;

    #[test]
    fn unions_reduce_components() {
        let mut uf = PersistenceUnionFind::new(3);
        uf.union_if_close(0, 1, 0.1, 0.2);
        assert_eq!(uf.betti_0(), 2);
        uf.union_if_close(1, 2, 0.1, 0.2);
        assert_eq!(uf.betti_0(), 1);
    }
}
