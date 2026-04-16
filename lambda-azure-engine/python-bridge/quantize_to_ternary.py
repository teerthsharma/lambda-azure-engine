import torch
import numpy as np
import os
import struct

def pack_ternary(values: np.ndarray) -> np.ndarray:
    """Pack an int8 array of {-1, 0, 1} into uint32 (16 values per word)."""
    assert values.dtype == np.int8
    flat = values.ravel()
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

def apply_twn_quantization(tensor):
    """Applies Ternary Weight Networks (TWN) quantization to a tensor."""
    w = tensor.cpu().numpy()
    if len(w.shape) < 2:
        return w # Don't quantize 1D tensors (biases, layernorms)
        
    std_w = np.std(w)
    threshold = 0.7 * std_w
    
    w_ternary = np.sign(w) * (np.abs(w) > threshold)
    return w_ternary.astype(np.int8)

def main():
    checkpoint_path = 'moe_fp16.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_path = '/content/drive/MyDrive/LAE/moe_fp16.pt'
        if not os.path.exists(checkpoint_path):
            print(f"Error: Could not find checkpoint at {checkpoint_path}")
            # Create a dummy for dry run testing
            print("Creating a dummy checkpoint for testing...")
            dummy = {"lm_head.weight": torch.randn(32000, 512)}
            for i in range(6):
                for j in range(8):
                    dummy[f"blocks.{i}.mlp.experts.{j}.w1.weight"] = torch.randn(2048, 512)
                    dummy[f"blocks.{i}.mlp.experts.{j}.w2.weight"] = torch.randn(512, 2048)
            torch.save(dummy, 'moe_fp16.pt')
            checkpoint_path = 'moe_fp16.pt'
            
    print(f"Loading FP16 checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    quantized_dict = {}
    packed_dict = {}
    
    print("Applying TWN Quantization...")
    for k, v in state_dict.items():
        if "expert" in k and "weight" in k:
            # We specifically target expert weights for extreme ternary quantization
            w_ternary = apply_twn_quantization(v)
            packed = pack_ternary(w_ternary)
            packed_dict[k] = packed
            print(f"  Quantized {k}: shape {v.shape} -> {packed.shape} uint32s")
        else:
            quantized_dict[k] = v.numpy() # Keep as float32 for now
            
    out_path = 'moe_ternary_packed.bin'
    print(f"Saving packed binary blob to {out_path}...")
    
    # Simple binary format for the packed experts
    # [num_tensors (int32)]
    # for each tensor:
    #   [name_len (int32)] [name (bytes)] [num_uint32s (int32)] [data (bytes)]
    with open(out_path, 'wb') as f:
        f.write(struct.pack('I', len(packed_dict)))
        for k, v in packed_dict.items():
            name_bytes = k.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('I', len(v)))
            f.write(v.tobytes())
            
    print("Quantization complete.")

if __name__ == "__main__":
    main()
