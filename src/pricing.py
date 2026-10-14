import numpy as np

def payoff_call(ST, K):
    """Payoff d'un call européen à maturité : max(ST - K, 0)."""
    return np.maximum(ST - K, 0)

def payoff_put(ST, K):
    """Payoff d'un put européen à maturité : max(K - ST, 0)."""
    return np.maximum(K - ST, 0)