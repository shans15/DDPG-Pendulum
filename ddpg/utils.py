"""
utils.py — Replay buffer, Ornstein-Uhlenbeck noise, and Adam optimizer.
"""

import numpy as np


class ReplayBuffer:
    """
    Fixed-size circular buffer storing (s, a, r, s', done) transitions.

    Transitions are sampled uniformly at random for each mini-batch update.
    Using a circular buffer (pointer wraps around) avoids expensive list
    operations while keeping memory usage constant.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.ptr      = 0
        self.size     = 0

        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

    def push(
        self,
        state:      np.ndarray,
        action:     np.ndarray,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        i = self.ptr % self.capacity
        self.states[i]      = state
        self.actions[i]     = action
        self.rewards[i]     = reward
        self.next_states[i] = next_state
        self.dones[i]       = float(done)
        self.ptr  += 1
        self.size  = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        idx = np.random.randint(0, self.size, batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size


class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    The process reverts to mean μ at rate θ, with perturbations scaled by σ:
        x_{t+1} = x_t + θ(μ − x_t)Δt + σ·N(0,1)

    Temporal correlation makes consecutive actions smoother — useful for
    physical systems with inertia like the pendulum.
    """

    def __init__(
        self,
        size:  int,
        mu:    float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.20,
    ) -> None:
        self.size  = size
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self) -> None:
        """Reset noise state to the long-run mean at the start of each episode."""
        self.x = np.copy(self.mu)

    def sample(self) -> np.ndarray:
        dx     = self.theta * (self.mu - self.x) + self.sigma * np.random.randn(self.size)
        self.x = self.x + dx
        return self.x.copy()


class Adam:
    """
    Adam optimizer (Kingma & Ba, 2014).

    Maintains per-parameter first and second moment estimates with
    bias-correction, giving adaptive learning rates for each weight.
    """

    def __init__(
        self,
        params: list,
        lr:     float = 1e-3,
        beta1:  float = 0.9,
        beta2:  float = 0.999,
        eps:    float = 1e-8,
    ) -> None:
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m     = [np.zeros_like(p) for p in params]
        self.v     = [np.zeros_like(p) for p in params]

    def step(self, params: list, grads: list) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t   # bias-correction denominators
        bc2 = 1.0 - self.beta2 ** self.t
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g ** 2
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2
            p    -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
