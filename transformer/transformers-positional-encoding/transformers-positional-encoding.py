import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pos = np.arange(seq_length)[:, np.newaxis]
    i = np.arange(d_model // 2)
    angle_rates = 1.0 / (10000 ** (2 * i / d_model))
    angles = pos * angle_rates
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(angles)  # Even indices: sin
    pe[:, 1::2] = np.cos(angles)  # Odd indices: cos
    
    return pe