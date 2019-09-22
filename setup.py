from setuptools import setup

setup(
  name = 'gpt_2_finetuning',   
  packages = ['gpt_2_finetuning'],
  version = '0.1',
  license='MIT',
  description = 'Package for finetuning GPT-2 models',
  author = 'Jonathan Heng',
  author_email = 'jonheng@users.noreply.github.com',
  url = 'https://github.com/jonheng/gpt-2-finetuning',
  download_url = 'https://github.com/jonheng/gpt-2-finetuning/archive/v0.1.tar.gz',
  keywords = ['language model', 'machine learning', 'gpt2'],
  install_requires=['numpy', 'fire', 'regex', 'requests', 'tqdm', 'toposort'],
  classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
  ],
)