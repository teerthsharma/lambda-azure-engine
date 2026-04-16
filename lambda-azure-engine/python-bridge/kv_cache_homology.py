import numpy as np

class PersistenceUnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.num_components = n
        self.sizes = np.ones(n, dtype=int)
        
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
        
    def union_if_close(self, i, j, dist, threshold):
        if dist <= threshold:
            root_i = self.find(i)
            root_j = self.find(j)
            
            if root_i != root_j:
                if self.sizes[root_i] < self.sizes[root_j]:
                    root_i, root_j = root_j, root_i
                    
                self.parent[root_j] = root_i
                self.sizes[root_i] += self.sizes[root_j]
                self.num_components -= 1
                return True
        return False

def cluster_and_compress(keys, threshold=0.1):
    """
    Implements Union-Find clustering on key vectors.
    keys: ndarray of shape [seq_len, head_dim]
    Returns: compressed_keys (ndarray)
    """
    seq_len = keys.shape[0]
    if seq_len == 0:
        return keys
        
    uf = PersistenceUnionFind(seq_len)
    
    # Compute pairwise L2 distances efficiently
    norms = np.sum(keys**2, axis=-1)
    dist_sq = norms[:, None] + norms[None, :] - 2 * np.dot(keys, keys.T)
    dist = np.sqrt(np.maximum(dist_sq, 0))
    
    # Apply union find based on threshold
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            uf.union_if_close(i, j, dist[i, j], threshold)
            
    # Compress by taking the mean of each component
    unique_roots = set()
    components = {}
    
    for i in range(seq_len):
        root = uf.find(i)
        unique_roots.add(root)
        if root not in components:
            components[root] = []
        components[root].append(i)
        
    compressed_keys = np.zeros((len(unique_roots), keys.shape[1]), dtype=keys.dtype)
    
    for idx, (root, indices) in enumerate(components.items()):
        # Merging redundant logical segments via averaging
        compressed_keys[idx] = np.mean(keys[indices], axis=0)
        
    print(f"Homology compression: {seq_len} -> {len(unique_roots)} (Betti-0: {uf.num_components})")
    return compressed_keys

if __name__ == "__main__":
    np.random.seed(42)
    # Simulate a sequence of 5 tokens, where the 2nd and 3rd are very similar
    test_keys = np.random.randn(5, 64)
    test_keys[2] = test_keys[1] + np.random.randn(64) * 0.05
    
    compressed = cluster_and_compress(test_keys, threshold=0.2)
    print("Compressed shape:", compressed.shape)
