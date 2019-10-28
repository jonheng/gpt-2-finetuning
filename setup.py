from setuptools import setup

setup(
  name = 'gpt_2_finetuning',   
  packages = ['gpt_2_finetuning'],
  version = '1.0.13',
  license='MIT',
  description = 'Package for finetuning GPT-2 models',
  author = 'Jonathan Heng',
  author_email = 'jonheng@users.noreply.github.com',
  url = 'https://github.com/jonheng/gpt-2-finetuning',
  keywords = ['language model', 'machine learning', 'gpt2'],
  install_requires=['numpy', 'regex', 'requests', 'tqdm', 'toposort', 'tensorflow==1.14'],
  classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
  ],
  scripts=[
    "bin/download_gpt2_model"
  ]
)