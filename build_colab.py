import json
import os

base_dir = r"c:\Users\seal\Desktop\New folder (13)\lambda-azure-engine\python-bridge"

GITHUB_REPO_PLACEHOLDER = "teerthsharma/lambda-azure-engine"

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code(lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}

cells = []

# --- Header ---
cells.append(md(
    "# Lambda Azure Engine — Omega Point Runtime\n"
    "## 200B Parameter Inference on 6GB VRAM via Ternary XOR+POPCNT\n\n"
    "This notebook clones the LAE repository from GitHub, runs the full pipeline, "
    "and produces a downloadable output artifact.\n\n"
    "**Instructions:** Set your GitHub repo URL in Cell 1, then **Runtime → Run All**."
))

# --- Cell 1: Config ---
cells.append(code([
    '# ====== CONFIGURATION — EDIT THIS ======\n',
    f'GITHUB_REPO = "https://github.com/{GITHUB_REPO_PLACEHOLDER}.git"\n',
    'BRANCH = "main"\n',
    '# ========================================\n',
]))

# --- Cell 2: Clone & setup ---
cells.append(md("## 1. Clone Repository & Install Dependencies"))
cells.append(code([
    'import os, sys, subprocess\n',
    '\n',
    '# Clone the repo\n',
    'if not os.path.exists("/content/lambda-azure-engine"):\n',
    '    !git clone --branch {BRANCH} --depth 1 {GITHUB_REPO} /content/lambda-azure-engine\n',
    'else:\n',
    '    print("Repo already cloned.")\n',
    '\n',
    '# Add python-bridge to path\n',
    'BRIDGE_DIR = "/content/lambda-azure-engine/lambda-azure-engine/python-bridge"\n',
    'if BRIDGE_DIR not in sys.path:\n',
    '    sys.path.insert(0, BRIDGE_DIR)\n',
    'os.chdir(BRIDGE_DIR)\n',
    '\n',
    '# Install dependencies\n',
    '!pip install -q torch numpy matplotlib safetensors huggingface_hub transformers\n',
    'try:\n',
    '    !pip install -q triton\n',
    'except:\n',
    '    print("[WARN] Triton install skipped — CPU fallback will be used.")\n',
    '\n',
    'print("Setup complete.")\n',
]))

# --- Cell 3: Triton kernel validation ---
cells.append(md("## 2. Validate Triton Ternary GEMM Kernel"))
cells.append(code([
    'from triton_ternary_gemm import test_pack_unpack_roundtrip, test_correctness\n',
    '\n',
    'test_pack_unpack_roundtrip()\n',
    'test_correctness(M=64, N=64, K=128)\n',
]))

# --- Cell 4: KV-Cache Homology demo ---
cells.append(md("## 3. KV-Cache Persistent Homology Compression"))
cells.append(code([
    'import numpy as np\n',
    'from kv_cache_homology import cluster_and_compress\n',
    '\n',
    'np.random.seed(42)\n',
    'keys = np.random.randn(100, 64)\n',
    '# Inject 20 near-duplicate tokens to show compression\n',
    'for i in range(20, 40):\n',
    '    keys[i] = keys[i - 20] + np.random.randn(64) * 0.03\n',
    '\n',
    'compressed = cluster_and_compress(keys, threshold=0.5)\n',
    'print(f"Original: {keys.shape[0]} tokens -> Compressed: {compressed.shape[0]} clusters")\n',
    'print(f"Memory reduction: {(1 - compressed.shape[0]/keys.shape[0])*100:.1f}%")\n',
]))

# --- Cell 5: Train small MoE ---
cells.append(md("## 4. Train 150M MoE Model (Quick Validation)"))
cells.append(code([
    'from train_ternary_moe import TernaryMoEModel, MoEConfig, train\n',
    'train()\n',
]))

# --- Cell 6: Quantize to ternary ---
cells.append(md("## 5. TWN Quantization → Packed 2-bit $\\mathbb{Z}_3$"))
cells.append(code([
    'from quantize_to_ternary import main as quantize_main\n',
    'quantize_main()\n',
]))

# --- Cell 7: Stream DeepSeek-V2 (optional, long) ---
cells.append(md(
    "## 6. (Optional) Stream & Quantize DeepSeek-V2 (236B)\n\n"
    "> **Warning:** This cell downloads ~472GB of data, processes it shard-by-shard, "
    "and produces a ~55GB packed binary. It will take **several hours** on Colab. "
    "Skip this cell if you only want to validate the math.\n\n"
    "Uncomment the last line to run."
))
cells.append(code([
    'from stream_and_quantize_deepseek import stream_and_quantize\n',
    '\n',
    '# Uncomment the line below to begin the full 200B streaming quantization:\n',
    '# stream_and_quantize(model_id="deepseek-ai/DeepSeek-V2", output_file="200b_lae_ternary_packed.bin")\n',
]))

# --- Cell 8: Evaluate perplexity ---
cells.append(md("## 7. Evaluate Perplexity"))
cells.append(code([
    'from evaluate_perplexity import evaluate\n',
    'evaluate()\n',
]))

# --- Cell 9: Benchmark suite ---
cells.append(md("## 8. Benchmark Suite (Perplexity vs VRAM)"))
cells.append(code([
    'from benchmark_suite import main as bench_main\n',
    'bench_main()\n',
]))

# --- Cell 9.5: Automated Testing ---
cells.append(md("## 9. Automated LLM Testing"))
cells.append(code([
    'from integration_test import run_tests\n',
    'run_tests()\n',
]))

# --- Cell 10: Collect & download output ---
cells.append(md("## 10. Download Output Artifacts"))
cells.append(code([
    'import os, zipfile, shutil\n',
    '\n',
    'OUTPUT_DIR = "/content/lae_output"\n',
    'os.makedirs(OUTPUT_DIR, exist_ok=True)\n',
    '\n',
    '# Collect all generated artifacts\n',
    'artifacts = [\n',
    '    "moe_fp16.pt",\n',
    '    "moe_ternary_packed.bin",\n',
    '    "benchmark.png",\n',
    '    "200b_lae_ternary_packed.bin",\n',
    '    "test_report.txt",\n',
    ']\n',
    '\n',
    'for f in artifacts:\n',
    '    if os.path.exists(f):\n',
    '        shutil.copy(f, os.path.join(OUTPUT_DIR, f))\n',
    '        print(f"  Collected: {f} ({os.path.getsize(f)/1e6:.1f} MB)")\n',
    '    else:\n',
    '        print(f"  Skipped (not found): {f}")\n',
    '\n',
    '# Create downloadable zip\n',
    'zip_path = "/content/lae_artifacts.zip"\n',
    'with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:\n',
    '    for root, dirs, files in os.walk(OUTPUT_DIR):\n',
    '        for file in files:\n',
    '            filepath = os.path.join(root, file)\n',
    '            zf.write(filepath, os.path.relpath(filepath, OUTPUT_DIR))\n',
    '\n',
    'print(f"\\nArtifact archive: {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")\n',
    '\n',
    '# Trigger browser download\n',
    'try:\n',
    '    from google.colab import files\n',
    '    files.download(zip_path)\n',
    '    print("Download triggered.")\n',
    'except ImportError:\n',
    '    print("Not running in Colab — file saved to", zip_path)\n',
]))

# --- Build notebook ---
nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(base_dir, "Omega_Point_Colab.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=2)

print(f"Created {out_path} successfully.")
