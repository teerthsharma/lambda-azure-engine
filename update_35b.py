import os

def replace_in_file(path, replacements):
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

replace_in_file('build_colab.py', [
    ('80B Parameter Holographic', '35B Parameter Holographic'),
    ('Qwen3-Coder-Next (80B)', 'Qwen3.5-35B-A3B (35B)'),
    ('Qwen/Qwen3-Coder-Next', 'Qwen/Qwen3.5-35B-A3B'),
    ('80b_lambda_padic_holographic.bin', '35b_lambda_padic_holographic.bin'),
    ('~160GB', '~70GB'),
    ('~20GB', '~9GB')
])

replace_in_file('lambda-azure-engine/python-bridge/stream_and_quantize_qwen.py', [
    ('Qwen/Qwen3-Coder-Next', 'Qwen/Qwen3.5-35B-A3B'),
    ('80b_lambda_padic_holographic.bin', '35b_lambda_padic_holographic.bin')
])

replace_in_file('README.md', [
    ('80B', '35B'),
    ('80B/6GB', '35B/6GB'),
    ('800M', '350M')
])

replace_in_file('200B_Scaling_Plan.md', [
    ('80B', '35B'),
    ('800M', '350M')
])

