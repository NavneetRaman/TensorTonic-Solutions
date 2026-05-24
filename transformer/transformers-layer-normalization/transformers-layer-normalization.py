import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    # Your code here
    mean = np.mean(x, axis = -1, keepdims = True)
    var = np.std(x, axis= -1, keepdims = True)**2
    return (gamma * ((x-mean) / (var+ eps)**0.5)) + beta