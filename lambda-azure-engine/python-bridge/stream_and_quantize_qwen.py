import os
import torch
import numpy as np
import struct
import gc
import warnings
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
warnings.filterwarnings("ignore", module="huggingface_hub")
from huggingface_hub import hf_hub_download, model_info

from lae_math import apply_padic_valuational_quantization, pack_ternary

# Try to import safetensors, handle gracefully for Colab
try:
    from safetensors.torch import load_file
except ImportError:
    print("Please install safetensors: pip install safetensors")

def stream_and_quantize(model_id="Qwen/Qwen2.5-14B", output_file="14b_lambda_padic_holographic.bin"):
    print(f"Initializing p-adic streaming quantization for {model_id} (Lambda Series)...")
    print("Executing Holographic Tensor Network memory mapping...")
    
    info = model_info(model_id)
    safetensor_files = [f.rfilename for f in info.siblings if f.rfilename.endswith('.safetensors')]
    total_files = len(safetensor_files)
    
    print(f"Discovered {total_files} shards.")
    
    with open(output_file, 'wb') as f_out:
        for idx, shard_filename in enumerate(safetensor_files):
            print(f"\n[{idx+1}/{total_files}] Downloading {shard_filename}...")
            
            shard_path = hf_hub_download(repo_id=model_id, filename=shard_filename)
            print(f"  -> File downloaded to {shard_path} (size: {os.path.getsize(shard_path)/1e9:.2f} GB)")
            
            tensors = load_file(shard_path, device="cpu")
            print(f"  -> Applying Categorical Context Sheaves & p-adic limits to {len(tensors)} tensors...")
            
            for name, tensor in tensors.items():
                name_bytes = name.encode('utf-8')
                f_out.write(struct.pack('I', len(name_bytes)))
                f_out.write(name_bytes)
                
                if "weight" in name and len(tensor.shape) >= 2:
                    # Dynamically shunt to Q_p (p-adic numbers)
                    w_ternary = apply_padic_valuational_quantization(tensor)
                    packed = pack_ternary(w_ternary)
                    
                    f_out.write(struct.pack('B', 1)) # is_ternary = True
                    f_out.write(struct.pack('I', len(packed)))
                    f_out.write(packed.tobytes())
                else:
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float16)
                    arr = tensor.cpu().numpy()
                    data_bytes = arr.tobytes()
                    f_out.write(struct.pack('B', 0)) # is_ternary = False
                    f_out.write(struct.pack('I', len(data_bytes)))
                    f_out.write(data_bytes)
            
            del tensors
            gc.collect()
            
            try:
                real_path = os.path.realpath(shard_path)
                os.remove(real_path)
                if os.path.exists(shard_path) and os.path.islink(shard_path):
                    os.remove(shard_path)
                print(f"  -> Purged shard from disk cache (Holographic bulk constraint).")
            except Exception as e:
                print(f"  -> Warning: Could not delete shard: {e}")
                
    print(f"\nStreaming quantization complete. Final p-adic packed model saved to {output_file}.")
    print(f"Final LAE disk footprint: {os.path.getsize(output_file)/1e9:.2f} GB.")

if __name__ == "__main__":
    print("Streamer ready. In Colab, execute stream_and_quantize() directly.")
