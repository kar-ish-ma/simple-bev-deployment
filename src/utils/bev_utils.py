import numpy as np

def generate_dummy_bev(size=256):
    # 3-channel BEV tensor
    return np.random.rand(3, size, size).astype(np.float32)