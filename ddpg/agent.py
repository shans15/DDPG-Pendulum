"""
agent.py — DDPGAgent: orchestrates actor, critic, buffer, noise, and updates.
"""

import numpy as np
from ddpg.networks import Actor, Critic
from ddpg.utils    import ReplayBuffer, OUNoise, Adam


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent.

    Maintains:
      - Online actor  π_θ       and target actor  π_θ'
      - Online critic Q_φ       and target critic Q_φ'
      - Experience replay buffer D
      - Ornstein-Uhlenbeck noise process for exploration

    Call flow per environment step
    ──────────────────────────────
    1. agent.select_action(state)         → clipped action + OU noise
    2. env.step(action)                   → (next_state, reward, done)
    3. agent.store(s, a, r, s', done)     → push to buffer
    4. agent.update()                     → one gradient step (if buffer ready)
    """

    def __init__(
        self,
        state_dim:    int,
        action_dim:   int,
        action_scale: float,
        config:       dict,
    ) -> None:
        self.action_scale = action_scale
        self.config       = config

        # Online networks
        self.actor  = Actor (state_dim, action_dim, action_scale)
        self.critic = Critic(state_dim, action_dim)

        # Target networks — initialized as exact copies
        self.actor_target  = Actor (state_dim, action_dim, action_scale)
        self.critic_target = Critic(state_dim, action_dim)
        self.actor_target .copy_from(self.actor)
        self.critic_target.copy_from(self.critic)

        # Optimizers
        self.actor_opt  = Adam(self.actor .net.params(), lr=config["lr_actor"])
        self.critic_opt = Adam(self.critic.net.params(), lr=config["lr_critic"])

        # Experience replay
        self.buffer = ReplayBuffer(
            config["buffer_size"], state_dim, action_dim
        )

        # Exploration noise
        self.noise = OUNoise(
            action_dim,
            theta=config["noise_theta"],
            sigma=config["noise_sigma"],
        )

    # ── Interaction ───────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        s = state.reshape(1, -1)
        a = self.actor.forward(s).flatten()
        if explore:
            a = a + self.noise.sample()
        return np.clip(a, -self.action_scale, self.action_scale)

    def store(self, s, a, r, s2, done) -> None:
        self.buffer.push(s, a, r, s2, done)

    def reset_noise(self) -> None:
        self.noise.reset()

    # ── Learning ──────────────────────────────────────────────────────────────

    def update(self) -> tuple[float | None, None]:
        """
        One full DDPG update step.

        Returns
        -------
        critic_loss : float | None   (None if buffer not yet full enough)
        actor_loss  : None           (implicit — we don't track it separately)
        """
        cfg = self.config
        if len(self.buffer) < cfg["batch_size"]:
            return None, None

        s, a, r, s2, done = self.buffer.sample(cfg["batch_size"])

        # ── 1. Compute Bellman targets using frozen target networks ───────────
        a2_target = self.actor_target.forward(s2)
        q_next    = self.critic_target.forward(s2, a2_target)
        y         = r + cfg["gamma"] * (1.0 - done) * q_next

        # ── 2. Critic update: minimize MSE(Q_φ(s,a), y) ──────────────────────
        critic_loss, da = self.critic.update(s, a, y, self.critic_opt)

        # ── 3. Actor update: ascend ∂Q/∂a · ∂a/∂θ ────────────────────────────
        self.actor.update(s, da, self.actor_opt)

        # ── 4. Soft target network updates ───────────────────────────────────
        tau = cfg["tau"]
        self.actor_target .soft_update(self.actor,  tau)
        self.critic_target.soft_update(self.critic, tau)

        return critic_loss, None
