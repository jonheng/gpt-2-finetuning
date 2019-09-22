## Generating distribution archives

```sh
python3 setup.py sdist bdist_wheel
```

## Upload distribution archives

```sh
python3 -m twine upload dist/*
```