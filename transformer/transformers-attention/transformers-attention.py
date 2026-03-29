import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    dk = Q.shape[-1]
    K = K.transpose(1,2)
    compatibility_matrix = Q@K
    compatibility_matrix /= (dk**0.5)
    attention = F.softmax(compatibility_matrix,dim=-1)
    attention = attention@V
    return attention