import os

def replace_in_file(path, replacements):
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

replacements_colab = [
    ('35B Parameter Holographic', '14B Parameter Holographic'),
    ('Qwen3.5-35B-A3B (35B)', 'Qwen2.5-14B (14B)'),
    ('Qwen/Qwen3.5-35B-A3B', 'Qwen/Qwen2.5-14B'),
    ('35b_lambda_padic_holographic.bin', '14b_lambda_padic_holographic.bin'),
    ('~70GB', '~28GB'),
    ('~9GB', '~3.5GB')
]

replace_in_file('build_colab.py', replacements_colab)

replace_in_file('lambda-azure-engine/python-bridge/stream_and_quantize_qwen.py', [
    ('Qwen/Qwen3.5-35B-A3B', 'Qwen/Qwen2.5-14B'),
    ('35b_lambda_padic_holographic.bin', '14b_lambda_padic_holographic.bin')
])

replace_in_file('README.md', [
    ('35B', '14B'),
    ('35B/6GB', '14B/6GB'),
    ('350M', '140M')
])

replace_in_file('200B_Scaling_Plan.md', [
    ('35B', '14B'),
    ('350M', '140M')
])
