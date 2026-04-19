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
    ('14B Parameter Holographic', '1.5B Parameter Holographic'),
    ('Qwen2.5-14B (14B)', 'Qwen2.5-1.5B (1.5B)'),
    ('Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-1.5B'),
    ('14b_lambda_padic_holographic.bin', '1_5b_lambda_padic_holographic.bin'),
    ('~28GB', '~3GB'),
    ('~3.5GB', '~350MB')
]

replace_in_file('build_colab.py', replacements_colab)

replace_in_file('lambda-azure-engine/python-bridge/stream_and_quantize_qwen.py', [
    ('Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-1.5B'),
    ('14b_lambda_padic_holographic.bin', '1_5b_lambda_padic_holographic.bin')
])

replace_in_file('README.md', [
    ('14B', '1.5B'),
    ('14B/6GB', '1.5B/6GB'),
    ('140M', '15M')
])

replace_in_file('200B_Scaling_Plan.md', [
    ('14B', '1.5B'),
    ('140M', '15M')
])
