"""
networks.py — Actor and Critic neural networks for DDPG.

Both networks are implemented as two-hidden-layer MLPs using NumPy only.
No autograd framework is required; gradients are derived analytically.
"""

import numpy as np


# ── Activations ───────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


# ── Weight initialization ──────────────────────────────────────────────────────

def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Kaiming He initialization — appropriate for ReLU-activated layers."""
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)


def uniform_init(fan_in: int, fan_out: int, limit: float = 3e-3) -> np.ndarray:
    """Small uniform init for output layers — keeps initial policy near zero."""
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


# ── Base MLP ──────────────────────────────────────────────────────────────────

class MLP:
    """
    Two-hidden-layer fully-connected network.

    Supports two output activations:
      - 'tanh'   : used by the actor (output in (-1, 1), scaled externally)
      - 'linear' : used by the critic (unbounded Q-value)

    All intermediate values needed for backprop are stored in self._cache
    after each forward pass.
    """

    def __init__(
        self,
        in_dim: int,
        h1: int,
        h2: int,
        out_dim: int,
        out_activation: str = "linear",
    ) -> None:
        self.out_activation = out_activation

        # Weight matrices and bias vectors
        self.W1 = he_init(in_dim, h1);       self.b1 = np.zeros((1, h1))
        self.W2 = he_init(h1, h2);           self.b2 = np.zeros((1, h2))
        self.W3 = uniform_init(h2, out_dim); self.b3 = np.zeros((1, out_dim))

        self._cache: dict = {}
        self._grads: dict = {}

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x  @ self.W1 + self.b1;  a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2;  a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3

        out = np.tanh(z3) if self.out_activation == "tanh" else z3
        self._cache = dict(x=x, z1=z1, a1=a1, z2=z2, a2=a2, z3=z3)
        return out

    # ── Backward pass ─────────────────────────────────────────────────────────

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backpropagate gradient dout (∂L/∂output) through the network.
        Returns ∂L/∂x (gradient w.r.t. the input), useful for the actor
        policy-gradient step where we need ∂Q/∂a.
        """
        N = self._cache["x"].shape[0]

        # Output layer gradient
        dz3 = dout * tanh_grad(self._cache["z3"]) if self.out_activation == "tanh" else dout
        dW3 = self._cache["a2"].T @ dz3 / N
        db3 = dz3.mean(axis=0, keepdims=True)

        # Second hidden layer
        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_grad(self._cache["z2"])
        dW2 = self._cache["a1"].T @ dz2 / N
        db2 = dz2.mean(axis=0, keepdims=True)

        # First hidden layer
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(self._cache["z1"])
        dW1 = self._cache["x"].T @ dz1 / N
        db1 = dz1.mean(axis=0, keepdims=True)

        # Input gradient (used by critic → actor chain)
        dx = dz1 @ self.W1.T

        self._grads = dict(W1=dW1, b1=db1, W2=dW2, b2=db2, W3=dW3, b3=db3)
        return dx

    # ── Parameter helpers ─────────────────────────────────────────────────────

    def params(self) -> list:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def grads(self) -> list:
        return [
            self._grads["W1"], self._grads["b1"],
            self._grads["W2"], self._grads["b2"],
            self._grads["W3"], self._grads["b3"],
        ]

    def copy_weights_from(self, other: "MLP") -> None:
        for dst, src in zip(self.params(), other.params()):
            dst[:] = src

    def soft_update(self, other: "MLP", tau: float) -> None:
        """Polyak averaging: θ_target ← τ·θ_online + (1−τ)·θ_target"""
        for dst, src in zip(self.params(), other.params()):
            dst[:] = tau * src + (1.0 - tau) * dst


# ── Actor ─────────────────────────────────────────────────────────────────────

class Actor:
    """
    Deterministic policy π_θ(s) → a ∈ [−action_scale, +action_scale].

    Output: tanh(z) * action_scale
    """

    def __init__(self, state_dim: int, action_dim: int, action_scale: float = 2.0) -> None:
        self.scale = action_scale
        self.net   = MLP(state_dim, 128, 128, action_dim, out_activation="tanh")

    def forward(self, s: np.ndarray) -> np.ndarray:
        return self.net.forward(s) * self.scale

    def update(self, s: np.ndarray, da_from_critic: np.ndarray, optimizer) -> None:
        """
        Deterministic policy gradient ascent step.

        da_from_critic: ∂Q/∂a — gradient of Q w.r.t. action (from critic backward).
        We negate it because Adam minimizes, but we want to maximize Q.
        """
        self.net.forward(s)                          # populate cache
        da_scaled = -da_from_critic * self.scale     # chain rule through scaling
        self.net.backward(da_scaled)
        optimizer.step(self.net.params(), self.net.grads())

    def soft_update(self, other: "Actor", tau: float) -> None:
        self.net.soft_update(other.net, tau)

    def copy_from(self, other: "Actor") -> None:
        self.net.copy_weights_from(other.net)


# ── Critic ────────────────────────────────────────────────────────────────────

class Critic:
    """
    Action-value function Q_φ(s, a) → scalar.

    Input: concatenation of state and action → [s ; a]
    Output: unbounded scalar (linear activation)
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.state_dim = state_dim
        self.net = MLP(state_dim + action_dim, 128, 128, 1, out_activation="linear")

    def forward(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        sa = np.concatenate([s, a], axis=1)
        return self.net.forward(sa)

    def update(self, s, a, target_q, optimizer) -> tuple[float, np.ndarray]:
        """
        MSE Bellman error update.

        Returns
        -------
        loss   : scalar MSE value (for logging)
        da     : ∂Q/∂a — passed to actor for the policy gradient step
        """
        q    = self.forward(s, a)
        loss = float(np.mean((q - target_q) ** 2))

        dq  = 2.0 * (q - target_q) / len(s)
        dsa = self.net.backward(dq)
        da  = dsa[:, self.state_dim:]          # split off action gradient

        optimizer.step(self.net.params(), self.net.grads())
        return loss, da

    def soft_update(self, other: "Critic", tau: float) -> None:
        self.net.soft_update(other.net, tau)

    def copy_from(self, other: "Critic") -> None:
        self.net.copy_weights_from(other.net)
