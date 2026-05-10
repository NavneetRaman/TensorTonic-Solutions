import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    Q_ = Q@W_q
    K_ = K@W_k
    V_ = V@W_v

    batch, seq, d_model = Q.shape
    d_k = d_model // num_heads

    Q_ = Q_.reshape(batch, seq, num_heads, d_k)
    K_ = K_.reshape(batch, seq, num_heads, d_k)
    V_ = V_.reshape(batch, seq, num_heads, d_k)
    
    Q_ = Q_.transpose(0, 2, 1, 3) # (B,nh,T,dk)
    K_ = K_.transpose(0, 2, 1, 3)
    K_ = np.swapaxes(K_,-2,-1) # K^T
    V_ = V_.transpose(0, 2, 1, 3)

    compatibility_matrix = Q_@K_
    compatibility_matrix /= (d_k**0.5)
    compatibility_matrix = softmax(compatibility_matrix)
    attention = compatibility_matrix@V_
    
    attention = attention.transpose(0,2,1,3) #(B,T,nh,dk)
    attention = attention.reshape(batch,seq,d_model)
    attention = attention @ W_o

    return attention