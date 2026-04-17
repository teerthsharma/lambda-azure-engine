import os

with open('build_colab.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_clone = '''    '# Clone the repo\\n',
    'if not os.path.exists("/content/lambda-azure-engine"):\\n',
    '    !git clone --branch {BRANCH} --depth 1 {GITHUB_REPO} /content/lambda-azure-engine\\n',
    'else:\\n',
    '    print("Repo already cloned.")\\n','''

new_clone = '''    '# Clone the repo (always pull latest)\\n',
    'if os.path.exists("/content/lambda-azure-engine"):\\n',
    '    !rm -rf /content/lambda-azure-engine\\n',
    '!git clone --branch {BRANCH} --depth 1 {GITHUB_REPO} /content/lambda-azure-engine\\n','''

content = content.replace(old_clone, new_clone)

with open('build_colab.py', 'w', encoding='utf-8') as f:
    f.write(content)
