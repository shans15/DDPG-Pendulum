"""
train.py — Entry point for training DDPG on Pendulum-v1.

Usage
-----
    python train.py                        # default config
    python train.py --episodes 300         # longer run
    python train.py --seed 0 --episodes 200 --no-render

All hyperparameters live in the CONFIG dict below and are also
exposed as CLI flags for easy sweeping / reproducibility.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import gymnasium as gym

from ddpg import DDPGAgent
from plot import make_plots


# ── Default hyperparameters ───────────────────────────────────────────────────

CONFIG = {
    # Environment
    "env_name":    "Pendulum-v1",
    "seed":        42,
    "max_episodes": 100,
    "max_steps":    200,

    # Learning
    "gamma":        0.99,
    "tau":          0.005,
    "lr_actor":     1e-3,
    "lr_critic":    1e-3,
    "batch_size":   64,
    "buffer_size":  20_000,
    "warmup_steps": 300,

    # Exploration noise (Ornstein-Uhlenbeck)
    "noise_theta":  0.15,
    "noise_sigma":  0.20,
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DDPG on Pendulum-v1")
    p.add_argument("--seed",        type=int,   default=CONFIG["seed"])
    p.add_argument("--episodes",    type=int,   default=CONFIG["max_episodes"])
    p.add_argument("--lr-actor",    type=float, default=CONFIG["lr_actor"])
    p.add_argument("--lr-critic",   type=float, default=CONFIG["lr_critic"])
    p.add_argument("--gamma",       type=float, default=CONFIG["gamma"])
    p.add_argument("--tau",         type=float, default=CONFIG["tau"])
    p.add_argument("--batch-size",  type=int,   default=CONFIG["batch_size"])
    p.add_argument("--output-dir",  type=str,   default="results")
    return p.parse_args()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(config: dict) -> dict:
    np.random.seed(config["seed"])

    env = gym.make(config["env_name"])
    env.reset(seed=config["seed"])

    state_dim    = env.observation_space.shape[0]     # 3
    action_dim   = env.action_space.shape[0]          # 1
    action_scale = float(env.action_space.high[0])    # 2.0

    agent = DDPGAgent(state_dim, action_dim, action_scale, config)

    episode_rewards: list[float] = []
    critic_losses:   list[float] = []
    avg_rewards:     list[float] = []
    total_steps = 0

    print(f"\n{'Episode':>8}  {'Reward':>10}  {'Avg-20':>10}  {'Critic Loss':>12}")
    print("─" * 48)

    t_start = time.time()

    for ep in range(config["max_episodes"]):
        state, _ = env.reset()
        agent.reset_noise()
        ep_reward     = 0.0
        ep_critic_losses: list[float] = []

        for _ in range(config["max_steps"]):
            total_steps += 1
            explore = total_steps >= config["warmup_steps"]

            if total_steps < config["warmup_steps"]:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, explore=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(state, action, reward, next_state, done)
            critic_loss, _ = agent.update()

            if critic_loss is not None:
                ep_critic_losses.append(critic_loss)

            state      = next_state
            ep_reward += reward

            if done:
                break

        episode_rewards.append(ep_reward)
        avg20 = float(np.mean(episode_rewards[-20:]))
        avg_rewards.append(avg20)
        mean_cl = float(np.mean(ep_critic_losses)) if ep_critic_losses else 0.0
        critic_losses.append(mean_cl)

        if (ep + 1) % 10 == 0:
            print(f"{ep+1:>8}  {ep_reward:>10.1f}  {avg20:>10.1f}  {mean_cl:>12.4f}")

    env.close()
    elapsed = time.time() - t_start

    results = {
        "config":          config,
        "episode_rewards": episode_rewards,
        "critic_losses":   critic_losses,
        "avg_rewards":     avg_rewards,
        "final_avg_20":    float(np.mean(episode_rewards[-20:])),
        "best_reward":     float(max(episode_rewards)),
        "best_episode":    int(np.argmax(episode_rewards)) + 1,
        "elapsed_sec":     round(elapsed, 1),
    }
    print(f"\n✓ Training complete in {elapsed:.1f}s")
    print(f"  Best episode : {results['best_reward']:.1f}  (ep {results['best_episode']})")
    print(f"  Final avg-20 : {results['final_avg_20']:.1f}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    cfg = CONFIG.copy()
    cfg.update({
        "seed":         args.seed,
        "max_episodes": args.episodes,
        "lr_actor":     args.lr_actor,
        "lr_critic":    args.lr_critic,
        "gamma":        args.gamma,
        "tau":          args.tau,
        "batch_size":   args.batch_size,
    })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = train(cfg)

    # Save metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    # Save plots
    plot_path = out_dir / "training_curves.png"
    make_plots(
        results["episode_rewards"],
        results["critic_losses"],
        results["avg_rewards"],
        save_path=str(plot_path),
    )
    print(f"  Plot saved    → {plot_path}")
