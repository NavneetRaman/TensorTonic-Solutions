import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    rows = np.arange(seq_length)
    cols = np.arange(d_model)
    out = np.ones((seq_length,d_model),dtype=np.float32)
    for i in rows:
        for j in cols:
            x = j//2
            w = 10000 ** (-2 * x / d_model)
            if j%2 == 0:
                out[i,j] *= np.sin(i*w)
            else : 
                out[i,j] *= np.cos(i*w)
    return out