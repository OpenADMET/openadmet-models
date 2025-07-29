# Integration testing

Run integration testing on high level components here

To run the CPU only tests use the `-m` flag of pytest to only run tests with the CPU mark

```python
pytest -vv test_anvil_integration.py -m cpu
```

To run the GPU tests use then `-m gpu` flag instead

```python
pytest -vv test_anvil_integration.py -m gpu
```
