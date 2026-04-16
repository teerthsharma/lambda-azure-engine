import os
import torch
import numpy as np
import struct
import gc
from huggingface_hub import hf_hub_download, model_info

# Try to import safetensors, handle gracefully for Colab
try:
    from safetensors.torch import load_file
except ImportError:
    print("Please install safetensors: pip install safetensors")

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
    """Applies Ternary Weight Networks (TWN) quantization to a PyTorch tensor."""
    w = tensor.cpu().float().numpy()
    if len(w.shape) < 2:
        return w # Skip 1D tensors (biases, norms)
        
    std_w = np.std(w)
    threshold = 0.7 * std_w
    
    w_ternary = np.sign(w) * (np.abs(w) > threshold)
    return w_ternary.astype(np.int8)

def stream_and_quantize(model_id="deepseek-ai/DeepSeek-V2", output_file="200b_lae_ternary_packed.bin"):
    print(f"Initializing streaming quantization for {model_id}...")
    
    info = model_info(model_id)
    safetensor_files = [f.rfilename for f in info.siblings if f.rfilename.endswith('.safetensors')]
    total_files = len(safetensor_files)
    
    print(f"Discovered {total_files} shards.")
    
    with open(output_file, 'wb') as f_out:
        # We will write the total file headers later, but since it's streaming,
        # we append sequentially. For simplicity in streaming, we use a custom format:
        # Sequence of: [name_len (int32)] [name (bytes)] [is_ternary (bool byte)] [num_elements (int32)] [data]
        
        for idx, shard_filename in enumerate(safetensor_files):
            print(f"\n[{idx+1}/{total_files}] Downloading {shard_filename}...")
            
            # Download a single shard to the cache
            shard_path = hf_hub_download(repo_id=model_id, filename=shard_filename)
            print(f"  -> File downloaded to {shard_path} (size: {os.path.getsize(shard_path)/1e9:.2f} GB)")
            
            # Load into RAM
            tensors = load_file(shard_path, device="cpu")
            print(f"  -> Processing {len(tensors)} tensors...")
            
            for name, tensor in tensors.items():
                name_bytes = name.encode('utf-8')
                f_out.write(struct.pack('I', len(name_bytes)))
                f_out.write(name_bytes)
                
                if "expert" in name and "weight" in name and len(tensor.shape) >= 2:
                    # Dynamically shunt to Z_3
                    w_ternary = apply_twn_quantization(tensor)
                    packed = pack_ternary(w_ternary)
                    
                    f_out.write(struct.pack('B', 1)) # is_ternary = True
                    f_out.write(struct.pack('I', len(packed)))
                    f_out.write(packed.tobytes())
                else:
                    # Keep as float16/bfloat16 numpy array
                    arr = tensor.cpu().numpy()
                    data_bytes = arr.tobytes()
                    f_out.write(struct.pack('B', 0)) # is_ternary = False
                    f_out.write(struct.pack('I', len(data_bytes)))
                    f_out.write(data_bytes)
            
            # Evict from RAM
            del tensors
            gc.collect()
            
            # THE CRITICAL STEP: Purge the uncompressed shard from disk to survive Colab's 75GB limit
            try:
                os.remove(shard_path)
                print(f"  -> Purged shard from disk cache.")
            except Exception as e:
                print(f"  -> Warning: Could not delete shard immediately (symlink/locking): {e}")
                
    print(f"\nStreaming quantization complete. Final packed model saved to {output_file}.")
    print(f"Final LAE disk footprint: {os.path.getsize(output_file)/1e9:.2f} GB.")

if __name__ == "__main__":
    # We use a smaller model for dry run/testing if needed, but deepseek is configured here.
    # stream_and_quantize(model_id="deepseek-ai/DeepSeek-V2")
    print("Streamer ready. In Colab, execute stream_and_quantize() directly.")
