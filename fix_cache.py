import os

with open('lambda-azure-engine/python-bridge/stream_and_quantize_qwen.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_cache_clear = '''            try:
                os.remove(shard_path)
                print(f"  -> Purged shard from disk cache (Holographic bulk constraint).")
            except Exception as e:
                print(f"  -> Warning: Could not delete shard: {e}")'''

new_cache_clear = '''            try:
                real_path = os.path.realpath(shard_path)
                os.remove(real_path)
                if os.path.exists(shard_path) and os.path.islink(shard_path):
                    os.remove(shard_path)
                print(f"  -> Purged actual blob and symlink from disk cache.")
            except Exception as e:
                print(f"  -> Warning: Could not delete shard: {e}")'''

content = content.replace(old_cache_clear, new_cache_clear)

with open('lambda-azure-engine/python-bridge/stream_and_quantize_qwen.py', 'w', encoding='utf-8') as f:
    f.write(content)
