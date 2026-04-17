import os
import torch
import struct
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import logging

# Suppress HuggingFace logs
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL_ID = "Qwen/Qwen2.5-1.5B"
BIN_FILE = "1_5b_lambda_padic_holographic.bin"

def unpack_ternary_torch(packed_uint32: torch.Tensor, shape) -> torch.Tensor:
    packed_int32 = packed_uint32.to(torch.int32)
    out = torch.zeros(packed_int32.size(0), 16, dtype=torch.int8, device=packed_int32.device)
    for i in range(16):
        two_bits = (packed_int32 >> (i * 2)) & 0b11
        presence = two_bits & 0b01
        sign = (two_bits >> 1) & 0b01
        val = presence.to(torch.int8) * (1 - 2 * sign.to(torch.int8))
        out[:, i] = val
    flat = out.flatten()
    num_elements = math.prod(shape)
    return flat[:num_elements].reshape(shape).to(torch.float16)

def load_holographic_model(model_id, bin_path):
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    print(f"Loading empty model architecture for {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    
    print(f"Restoring p-adic packed weights from {bin_path}...")
    state_dict = model.state_dict()
    
    with open(bin_path, 'rb') as f:
        while True:
            name_len_bytes = f.read(4)
            if not name_len_bytes or len(name_len_bytes) < 4:
                break
            
            name_len = struct.unpack('I', name_len_bytes)[0]
            name = f.read(name_len).decode('utf-8')
            
            is_ternary_byte = f.read(1)
            is_ternary = struct.unpack('B', is_ternary_byte)[0]
            
            orig_tensor = state_dict.get(name, None)
            
            if is_ternary:
                num_uint32s = struct.unpack('I', f.read(4))[0]
                data = f.read(num_uint32s * 4)
                packed_np = np.frombuffer(data, dtype=np.uint32)
                packed_torch = torch.from_numpy(packed_np.copy())
                
                if orig_tensor is not None:
                    unpacked = unpack_ternary_torch(packed_torch, orig_tensor.shape)
                    # Heuristic scaling to restore variance of the tensor
                    scale_factor = 1.0 / math.sqrt(orig_tensor.shape[1]) if len(orig_tensor.shape) > 1 else 0.02
                    state_dict[name].copy_(unpacked * scale_factor)
            else:
                num_bytes = struct.unpack('I', f.read(4))[0]
                data = f.read(num_bytes)
                if orig_tensor is not None:
                    # Attempt to load as float16 (since we cast bfloat16 to float16 in streaming)
                    # For some tensors like embedding or layernorm
                    try:
                        arr = np.frombuffer(data, dtype=np.float16).copy()
                        state_dict[name].copy_(torch.from_numpy(arr).reshape(orig_tensor.shape))
                    except ValueError:
                        # Fallback for float32 buffers
                        arr = np.frombuffer(data, dtype=np.float32).copy()
                        state_dict[name].copy_(torch.from_numpy(arr).reshape(orig_tensor.shape))
                    
    print("P-adic weights successfully grafted onto Holographic Boundary.")
    return model, tokenizer

def chat():
    if not os.path.exists(BIN_FILE):
        print(f"Error: {BIN_FILE} not found. Please run stream_and_quantize_qwen.py first.")
        return

    model, tokenizer = load_holographic_model(MODEL_ID, BIN_FILE)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving model to {device}...")
    model.to(device)
    model.eval()
    
    print("\\n" + "="*60)
    print("Lambda Series Holographic Inference Engine Ready")
    print("Mathematical Topology: Q_p (p=3), Ultimateric Space")
    print("Type 'exit' to quit.")
    print("="*60 + "\\n")
    
    while True:
        try:
            prompt = input("User: ")
            if prompt.strip().lower() == 'exit':
                break
                
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=50, 
                    temperature=0.6, 
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Lambda: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    # Test generation automatically if no tty, otherwise start chat loop
    import sys
    if not sys.stdin.isatty():
        print("Non-interactive mode detected. Running quick evaluation...")
        model, tokenizer = load_holographic_model(MODEL_ID, BIN_FILE)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        prompt = "Hello, what is the nature of the universe?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        print("Output:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    else:
        chat()
