"""Day 30 · LoRA by hand

LoRA freezes a big weight matrix W (d_out × d_in) and learns a small update B @ A instead:
    A is r × d_in, B is d_out × r, and the layer computes  x @ W.T + (alpha / r) * x @ A.T @ B.T
B starts at zero, so a fresh LoRA layer behaves exactly like the original one.

Use NumPy (pip install numpy). Run `python quest.py 30` after each TODO.
"""
import numpy as np  # noqa: F401


def lora_params(d_in: int, d_out: int, r: int) -> int:
    """TODO 30.1 · How many numbers does LoRA train for one d_out × d_in layer at rank r?"""
    return r * d_in + d_out * r


def full_params(d_in: int, d_out: int) -> int:
    """TODO 30.1 · How many numbers does full fine-tuning train for the same layer?"""
    return d_out * d_in


class LoRALinear:
    def __init__(self, W: np.ndarray, r: int = 8, alpha: float = 16, seed: int = 0):
        """TODO 30.2 · Keep W (frozen), r and alpha. Create
             A: shape (r, d_in), random normal values times 0.01 (use np.random.default_rng(seed))
             B: shape (d_out, r), all zeros
        """
        self.W = W
        self.r = r 
        self.alpha = alpha

        d_out, d_in = W.shape 
        rng = np.random.default_rng(seed)
        self.A = rng.normal(size=(r, d_in))*0.01
        self.B = np.zeros((d_out, r))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """TODO 30.2 · x has shape (batch, d_in). Return x @ W.T + (alpha / r) * x @ A.T @ B.T"""
        return x @ self.W.T + (self.alpha / self.r) * x @ self.A.T @ self.B.T

    def merged(self) -> np.ndarray:
        """TODO 30.3 · After training, fold the update into one matrix: W + (alpha / r) * B @ A.
        Using it should give the same output as forward()."""
        return self.W + (self.alpha / self.r) * self.B @ self.A


def lora_budget(n_layers: int, d_model: int, r: int, targets: int = 2) -> int:
    """TODO 30.4 · LoRA on `targets` square d_model × d_model projections (like q_proj and v_proj)
    in every one of n_layers layers. How many trainable numbers in total?"""
    one_proj = lora_params(d_model, d_model, r)
    one_layer = one_proj*targets 
    total = n_layers*one_layer 
    return total
