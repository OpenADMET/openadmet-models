import os
import sys
import numpy as np
import multiprocessing as mp
import pytest
from openadmet.models.features.molfeat_fingerprint import FingerprintFeaturizer
from openadmet.models.features.molfeat_properties import DescriptorFeaturizer

@pytest.fixture
def smiles_data():
    """Test SMILES data"""
    return ["CCO", "CCN", "CCO"] * 5

@pytest.fixture(params=[2, mp.cpu_count()])
def process_pool(request):
    """Create process pool with different CPU counts"""
    processes_to_use = request.param
    pool = mp.Pool(processes=processes_to_use)
    yield pool
    pool.close()
    pool.join()

def process_smiles(smiles_list):
    """Process feature extraction in a child process"""
    print(f"Process {os.getpid()} started")
    featurizer = DescriptorFeaturizer(descr_type="mordred")
    # Print actual number of processes used by the featurizer
    print(f"Featurizer n_jobs setting: {featurizer.n_jobs}")
    X, idx = featurizer.featurize(smiles_list)
    print(f"Process {os.getpid()} feature extraction completed: shape {X.shape}")
    return X.shape

def test_multiprocessing_feature_extraction(smiles_data, process_pool):
    """Test multiprocessing feature extraction with different CPU counts"""
    total_cpus = mp.cpu_count()
    processes_to_use = process_pool._processes
    print(f"Total system CPUs: {total_cpus}")
    print(f"Current test using processes: {processes_to_use}")
    print(f"Default processes used by molfeat: {-1} (all available cores)")

    results = []
    for i in range(3):
        result = process_pool.apply_async(process_smiles, (smiles_data,))
        results.append(result)

    for i, result in enumerate(results):
        try:
            shape = result.get(timeout=30)
            print(f"Result {i}: {shape}")
            assert isinstance(shape, tuple), "Result should be a tuple"
            assert len(shape) == 2, "Result should have 2 dimensions"
        except mp.TimeoutError:
            pytest.fail(f"Result {i}: Timeout - possibly hanging")
