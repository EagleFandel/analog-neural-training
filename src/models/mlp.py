from __future__ import annotations

from typing import List, Tuple
import numpy as np


class MLP:
    """Minimal NumPy MLP with parameter vectorization utilities.

    - Hidden activation: ReLU
    - Output: linear (regression) or logits (classification)
    """

    def __init__(self, layer_sizes: List[int], rng: np.random.Generator | None = None, weight_scale: float = 0.1):
        assert len(layer_sizes) >= 2, "Need at least input and output layer"
        self.layer_sizes = layer_sizes
        self.rng = np.random.default_rng(None if rng is None else rng)
        self.weight_scale = float(weight_scale)

        self.shapes: List[Tuple[int, int]] = []  # [(in_dim, out_dim), ...]
        self.param_slices: List[Tuple[slice, slice]] = []  # weight slice, bias slice in theta
        self.theta0 = self._init_params()

    def _init_params(self) -> np.ndarray:
        parts = []
        offset = 0
        self.shapes.clear()
        self.param_slices.clear()
        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.shapes.append((in_dim, out_dim))
            w = (self.rng.normal(0.0, self.weight_scale, size=(in_dim, out_dim))).astype(np.float64)
            b = (self.rng.normal(0.0, self.weight_scale, size=(out_dim,))).astype(np.float64)
            w_flat = w.ravel()
            b_flat = b.ravel()
            w_slice = slice(offset, offset + w_flat.size)
            offset += w_flat.size
            b_slice = slice(offset, offset + b_flat.size)
            offset += b_flat.size
            self.param_slices.append((w_slice, b_slice))
            parts.append(w_flat)
            parts.append(b_flat)
        return np.concatenate(parts, axis=0)

    def unpack(self, theta: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        params: List[Tuple[np.ndarray, np.ndarray]] = []
        for (in_dim, out_dim), (w_slice, b_slice) in zip(self.shapes, self.param_slices):
            w = theta[w_slice].reshape(in_dim, out_dim)
            b = theta[b_slice].reshape(out_dim)
            params.append((w, b))
        return params

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def relu_grad(h: np.ndarray) -> np.ndarray:
        return (h > 0.0).astype(h.dtype)

    def forward(self, theta: np.ndarray, x: np.ndarray, task: str = "regression") -> np.ndarray:
        h = x
        caches = []
        for li, (w, b) in enumerate(self.unpack(theta)):
            z = h @ w + b
            is_last = (li == len(self.shapes) - 1)
            if not is_last:
                h = self.relu(z)
            else:
                h = z  # logits or linear output
            caches.append((z, h))
        return h

    def loss_and_grad(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray, task: str = "regression") -> Tuple[float, np.ndarray]:
        # Forward pass with caches for backprop
        activations = [x]
        preacts = []
        h = x
        params = self.unpack(theta)
        L = len(params)
        for li, (w, b) in enumerate(params):
            z = h @ w + b
            preacts.append(z)
            is_last = (li == L - 1)
            if not is_last:
                h = self.relu(z)
            else:
                h = z
            activations.append(h)

        if task == "regression":
            # MSE
            # Ensure y has the same shape as output
            y_reshaped = y.reshape(activations[-1].shape)
            diff = activations[-1] - y_reshaped
            loss = float(0.5 * np.mean(diff ** 2))
            grad_out = diff / diff.shape[0]
        elif task == "classification":
            # softmax-crossentropy
            logits = activations[-1]
            logits = logits - np.max(logits, axis=1, keepdims=True)
            exp = np.exp(logits)
            probs = exp / np.sum(exp, axis=1, keepdims=True)
            # y can be int labels shape (n,) or one-hot (n, C)
            if y.ndim == 1:
                n = y.shape[0]
                loss = -float(np.mean(np.log(probs[np.arange(n), y])))
                grad_out = probs
                grad_out[np.arange(n), y] -= 1.0
                grad_out /= n
            else:
                n = y.shape[0]
                loss = -float(np.mean(np.sum(y * np.log(probs + 1e-12), axis=1)))
                grad_out = (probs - y) / n
        else:
            raise ValueError(f"Unknown task: {task}")

        # Backprop
        grads_w: List[np.ndarray] = []
        grads_b: List[np.ndarray] = []
        g = grad_out
        for li in reversed(range(L)):
            a_prev = activations[li]
            z = preacts[li]
            w, b = params[li]
            is_last = (li == L - 1)
            # grad w.r.t weights and biases
            gw = a_prev.T @ g
            gb = np.sum(g, axis=0)
            grads_w.append(gw)
            grads_b.append(gb)
            if li > 0:
                g = g @ w.T
                g = g * self.relu_grad(activations[li])

        grads_w.reverse()
        grads_b.reverse()
        # pack
        parts = []
        for (w, b), gw, gb in zip(params, grads_w, grads_b):
            parts.append(gw.ravel())
            parts.append(gb.ravel())
        grad_theta = np.concatenate(parts, axis=0)
        return loss, grad_theta




