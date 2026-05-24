import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here
    l1 = (x@W1) + b1
    l1 = np.clip(l1,0,None)
    l2 = (l1@W2) + b2
    return l2