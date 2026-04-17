import os

with open('build_colab.py', 'r', encoding='utf-8') as f:
    content = f.read()

# We need to remove the Cell 5 that trains the 150M MoE Model.
cell_to_remove = '''# --- Cell 5: Train small MoE ---
cells.append(md("## 4. Train 150M MoE Model (Quick Validation)"))
cells.append(code([
    'from train_ternary_moe import TernaryMoEModel, MoEConfig, train\\n',
    'train()\\n',
]))'''

content = content.replace(cell_to_remove, '')

# We should also fix the cell numbers and headers to ensure the notebook looks correct
content = content.replace('## 5. TWN Quantization', '## 4. TWN Quantization')
content = content.replace('## 6. (Optional) Stream & Quantize Qwen2.5-14B', '## 5. (Optional) Stream & Quantize Qwen2.5-14B')
content = content.replace('## 7. Evaluate Perplexity', '## 6. Evaluate Perplexity')
content = content.replace('## 8. Benchmark Suite', '## 7. Benchmark Suite')
content = content.replace('## 9. Automated LLM Testing', '## 8. Automated LLM Testing')
content = content.replace('## 10. Download Output Artifacts', '## 9. Download Output Artifacts')

# And remove it from artifacts download
artifacts_to_remove = '''    "moe_fp16.pt",\\n    "moe_ternary_packed.bin",\\n'''
content = content.replace(artifacts_to_remove, '')

with open('build_colab.py', 'w', encoding='utf-8') as f:
    f.write(content)
