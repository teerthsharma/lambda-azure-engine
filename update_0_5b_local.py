import os

def replace_in_file(path, replacements):
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

replace_in_file('lambda-azure-engine/python-bridge/stream_and_quantize_qwen.py', [
    ('Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-0.5B'),
    ('1_5b_lambda_padic_holographic.bin', '0_5b_lambda_padic_holographic.bin'),
    ('print("Streamer ready. In Colab, execute stream_and_quantize() directly.")', 'print("Streamer ready. Running stream_and_quantize() locally.")\\n    stream_and_quantize()')
])

