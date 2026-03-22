"""
PPO training for UQM Melee - CleanRL single-file style.

This file is the PRIMARY MODIFICATION TARGET for competing agents.
Agents can change anything: algorithm, architecture integration,
preprocessing, frame stacking, learning rate schedules, etc.

Based on CleanRL's PPO implementation.
"""

import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.agent import MeleeAgent, preprocess_obs
from training.config import TrainingConfig


def make_env(config: TrainingConfig, env_id: int = 0):
    """Create a single MeleeEnv instance."""
    from uqm_env.melee_env import MeleeEnv
    return MeleeEnv(
        ship_p1=config.ship_p1,
        ship_p2=config.ship_p2,
        p2_cyborg=config.p2_cyborg,
        frame_skip=config.frame_skip,
        headless=True,
        seed=config.ship_p1 * 1000 + env_id,
    )


def train(config: TrainingConfig):
    """
    Main PPO training loop.

    Logs training metrics to config.log_dir and saves checkpoints
    to config.checkpoint_dir. Periodically evaluates against Cyborg
    and records time-to-competency.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # Create vectorized environments
    envs = [make_env(config, i) for i in range(config.num_envs)]

    # Create agent
    try:
        agent = MeleeAgent(
            encoder_type="siglip",
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            encoder_name=config.encoder_name,
            encoder_pretrained=config.encoder_pretrained,
        ).to(device)
        print("Using SigLIP encoder")
    except (ImportError, RuntimeError) as e:
        print(f"SigLIP unavailable ({e}), falling back to CNN encoder")
        agent = MeleeAgent(
            encoder_type="cnn",
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
        ).to(device)

    # Only optimize trainable parameters (encoder is frozen)
    trainable_params = [p for p in agent.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=config.learning_rate, eps=1e-5)

    # Rollout storage
    batch_size = config.num_envs * config.num_steps
    minibatch_size = batch_size // config.num_minibatches
    num_updates = config.total_timesteps // batch_size

    obs_buf = torch.zeros((config.num_steps, config.num_envs, 240, 320, 3), dtype=torch.uint8)
    actions_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long)
    logprobs_buf = torch.zeros((config.num_steps, config.num_envs))
    rewards_buf = torch.zeros((config.num_steps, config.num_envs))
    dones_buf = torch.zeros((config.num_steps, config.num_envs))
    values_buf = torch.zeros((config.num_steps, config.num_envs))

    # Initialize environments
    next_obs_list = []
    next_done = torch.zeros(config.num_envs)
    for env in envs:
        obs, _ = env.reset()
        next_obs_list.append(obs)
    next_obs = torch.from_numpy(np.array(next_obs_list))

    # Training metrics
    start_time = time.time()
    global_step = 0
    best_win_rate = 0.0
    competency_reached_at = None
    training_log = []

    print(f"Starting PPO training for {config.total_timesteps} timesteps")
    print(f"  Batch size: {batch_size}, Minibatch size: {minibatch_size}")
    print(f"  Updates: {num_updates}")

    for update in range(1, num_updates + 1):
        # Annealing learning rate (linear)
        frac = 1.0 - (update - 1.0) / num_updates
        lr = frac * config.learning_rate
        optimizer.param_groups[0]["lr"] = lr

        # Collect rollout
        for step in range(config.num_steps):
            global_step += config.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # Step environments
            next_obs_list = []
            for i, env in enumerate(envs):
                obs, reward, terminated, truncated, info = env.step(action[i].item())
                rewards_buf[step, i] = reward
                next_done[i] = float(terminated or truncated)

                if terminated or truncated:
                    obs, _ = env.reset()

                next_obs_list.append(obs)

            next_obs = torch.from_numpy(np.array(next_obs_list))

        # Compute GAE advantages
        with torch.no_grad():
            next_value = agent.get_value(next_obs).squeeze(-1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = (rewards_buf[t] + config.gamma * nextvalues * nextnonterminal
                         - values_buf[t])
                advantages[t] = lastgaelam = (
                    delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        # Flatten batch
        b_obs = obs_buf.reshape(-1, 240, 320, 3)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # PPO update
        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                optimizer.step()

        # Log metrics
        elapsed = time.time() - start_time
        sps = int(global_step / elapsed)

        log_entry = {
            "global_step": global_step,
            "wall_clock_seconds": elapsed,
            "sps": sps,
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "lr": lr,
            "clipfrac": np.mean(clipfracs),
        }

        # Periodic evaluation
        if global_step % config.eval_interval < batch_size:
            win_rate = evaluate_agent(agent, config)
            log_entry["win_rate"] = win_rate
            print(f"Step {global_step:>8d} | "
                  f"Win rate: {win_rate:.2%} | "
                  f"SPS: {sps} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Time: {elapsed:.0f}s")

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(agent.state_dict(),
                           os.path.join(config.checkpoint_dir, "best.pt"))

            if win_rate >= config.target_win_rate and competency_reached_at is None:
                competency_reached_at = elapsed
                print(f"TARGET REACHED: {config.target_win_rate:.0%} win rate "
                      f"at {elapsed:.1f}s ({global_step} steps)")
        else:
            if update % 10 == 0:
                print(f"Step {global_step:>8d} | SPS: {sps} | Loss: {loss.item():.4f}")

        training_log.append(log_entry)

        # Checkpoint
        if global_step % config.checkpoint_interval < batch_size:
            torch.save(agent.state_dict(),
                       os.path.join(config.checkpoint_dir, f"step_{global_step}.pt"))

    # Save final results
    results = {
        "total_timesteps": global_step,
        "wall_clock_seconds": time.time() - start_time,
        "best_win_rate": best_win_rate,
        "competency_reached_at_seconds": competency_reached_at,
        "training_log": training_log,
    }

    with open(os.path.join(config.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save(agent.state_dict(),
               os.path.join(config.checkpoint_dir, "final.pt"))

    # Clean up
    for env in envs:
        env.close()

    print(f"\nTraining complete.")
    print(f"  Total time: {results['wall_clock_seconds']:.1f}s")
    print(f"  Best win rate: {best_win_rate:.2%}")
    if competency_reached_at is not None:
        print(f"  Time to {config.target_win_rate:.0%}: {competency_reached_at:.1f}s")
    else:
        print(f"  Did not reach {config.target_win_rate:.0%} win rate")

    return results


def evaluate_agent(agent, config: TrainingConfig) -> float:
    """
    Evaluate agent against Cyborg AI.

    Returns win rate over config.eval_episodes episodes.
    """
    env = make_env(config, env_id=999)
    wins = 0

    for ep in range(config.eval_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action, _, _, _ = agent.get_action_and_value(obs_tensor)

            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

        if info.get("winner") == 0:
            wins += 1

    env.close()
    return wins / max(config.eval_episodes, 1)
