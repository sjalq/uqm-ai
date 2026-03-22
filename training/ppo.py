"""
PPO training for UQM Melee - CleanRL single-file style.

Agent 1 - Round 1: Reward shaping + fast convergence + robustness.
- Dense reward signals via RewardShaper
- Wall-clock time limit (300s)
- Entropy annealing for explore-then-exploit
- Crash resilience (try/except, incremental saves, OOM handling)
"""

import time
import os
import json
import logging
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.agent import MeleeAgent, preprocess_obs
from training.config import TrainingConfig
from uqm_env.reward import RewardShaper

logger = logging.getLogger(__name__)


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


def save_results(config, global_step, start_time, best_win_rate,
                 competency_reached_at, training_log):
    """Incrementally save training results to disk."""
    try:
        results = {
            "total_timesteps": global_step,
            "wall_clock_seconds": time.time() - start_time,
            "best_win_rate": best_win_rate,
            "competency_reached_at_seconds": competency_reached_at,
            "training_log": training_log,
        }
        results_path = os.path.join(config.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")


def save_checkpoint(agent, path, label="checkpoint"):
    """Save agent checkpoint with error handling."""
    try:
        torch.save(agent.state_dict(), path)
        logger.info(f"Saved {label} to {path}")
    except Exception as e:
        logger.warning(f"Failed to save {label}: {e}")


def train(config: TrainingConfig):
    """
    Main PPO training loop with reward shaping and wall-clock limit.

    Key improvements over baseline:
    - RewardShaper provides dense learning signals (damage scaling, combos, survival)
    - Entropy annealing: high exploration early, exploitation later
    - 300s wall-clock hard limit with best checkpoint saving
    - Crash resilience: OOM recovery, incremental saves, env restart on failure
    """
    # Set up logging
    os.makedirs(config.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.log_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Training state - declared early so cleanup can reference them
    envs = []
    agent = None
    best_win_rate = 0.0
    competency_reached_at = None
    training_log = []
    global_step = 0
    start_time = time.time()

    try:
        # Create vectorized environments
        envs = []
        for i in range(config.num_envs):
            try:
                envs.append(make_env(config, i))
            except Exception as e:
                logger.error(f"Failed to create env {i}: {e}")
                raise

        # Create agent
        try:
            agent = MeleeAgent(
                encoder_type="siglip",
                hidden_dim=config.hidden_dim,
                action_dim=config.action_dim,
                encoder_name=config.encoder_name,
                encoder_pretrained=config.encoder_pretrained,
            ).to(device)
            logger.info("Using SigLIP encoder")
        except (ImportError, RuntimeError) as e:
            logger.info(f"SigLIP unavailable ({e}), falling back to CNN encoder")
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

        obs_buf = torch.zeros(
            (config.num_steps, config.num_envs, 240, 320, 3), dtype=torch.uint8
        )
        actions_buf = torch.zeros(
            (config.num_steps, config.num_envs), dtype=torch.long
        )
        logprobs_buf = torch.zeros((config.num_steps, config.num_envs))
        rewards_buf = torch.zeros((config.num_steps, config.num_envs))
        dones_buf = torch.zeros((config.num_steps, config.num_envs))
        values_buf = torch.zeros((config.num_steps, config.num_envs))

        # Initialize reward shaper
        reward_shaper = RewardShaper(config.num_envs)

        # Initialize environments
        next_obs_list = []
        next_done = torch.zeros(config.num_envs)
        for i, env in enumerate(envs):
            obs, info = env.reset()
            next_obs_list.append(obs)
            reward_shaper.reset_env(i, info)
        next_obs = torch.from_numpy(np.array(next_obs_list))

        # Entropy annealing setup
        ent_coef_start = config.ent_coef
        ent_coef_final = getattr(config, "ent_coef_final", 0.005)

        logger.info(f"Starting PPO training for {config.total_timesteps} timesteps")
        logger.info(f"  Batch size: {batch_size}, Minibatch size: {minibatch_size}")
        logger.info(f"  Updates: {num_updates}")
        logger.info(f"  Wall-clock limit: {config.wall_clock_limit}s")

        for update in range(1, num_updates + 1):
            # Check wall-clock limit
            elapsed = time.time() - start_time
            if elapsed >= config.wall_clock_limit:
                logger.info(
                    f"Wall-clock limit reached ({elapsed:.1f}s >= "
                    f"{config.wall_clock_limit}s). Stopping training."
                )
                break

            # Learning rate annealing (linear)
            frac = 1.0 - (update - 1.0) / num_updates
            lr = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lr

            # Entropy coefficient annealing (linear)
            ent_coef = ent_coef_start + (ent_coef_final - ent_coef_start) * (1.0 - frac)

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
                    try:
                        obs, env_reward, terminated, truncated, info = env.step(
                            action[i].item()
                        )

                        # Use reward shaper instead of raw env reward
                        shaped_reward = reward_shaper.compute_reward(
                            i, info, terminated or truncated, env_reward
                        )
                        rewards_buf[step, i] = shaped_reward
                        next_done[i] = float(terminated or truncated)

                        if terminated or truncated:
                            obs, info = env.reset()
                            reward_shaper.reset_env(i, info)

                        next_obs_list.append(obs)
                    except Exception as e:
                        # If an env crashes, try to restart it
                        logger.warning(f"Env {i} crashed: {e}. Restarting.")
                        try:
                            envs[i].close()
                        except Exception:
                            pass
                        try:
                            envs[i] = make_env(config, i)
                            obs, info = envs[i].reset()
                            reward_shaper.reset_env(i, info)
                            next_obs_list.append(obs)
                            rewards_buf[step, i] = 0.0
                            next_done[i] = 1.0
                        except Exception as e2:
                            logger.error(f"Env {i} restart failed: {e2}")
                            # Use zero obs as fallback
                            next_obs_list.append(
                                np.zeros((240, 320, 3), dtype=np.uint8)
                            )
                            rewards_buf[step, i] = 0.0
                            next_done[i] = 1.0

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
                    delta = (
                        rewards_buf[t]
                        + config.gamma * nextvalues * nextnonterminal
                        - values_buf[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + config.gamma
                        * config.gae_lambda
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + values_buf

            # Flatten batch
            b_obs = obs_buf.reshape(-1, 240, 320, 3)
            b_logprobs = logprobs_buf.reshape(-1)
            b_actions = actions_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)

            # PPO update with OOM handling
            b_inds = np.arange(batch_size)
            clipfracs = []

            try:
                for epoch in range(config.update_epochs):
                    np.random.shuffle(b_inds)

                    for start in range(0, batch_size, minibatch_size):
                        end = start + minibatch_size
                        mb_inds = b_inds[start:end]

                        _, newlogprob, entropy, newvalue = (
                            agent.get_action_and_value(
                                b_obs[mb_inds], b_actions[mb_inds]
                            )
                        )

                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = logratio.exp()

                        with torch.no_grad():
                            clipfracs.append(
                                ((ratio - 1.0).abs() > config.clip_coef)
                                .float()
                                .mean()
                                .item()
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
                        v_loss = 0.5 * (
                            (newvalue - b_returns[mb_inds]) ** 2
                        ).mean()

                        # Entropy loss (with annealed coefficient)
                        entropy_loss = entropy.mean()

                        loss = (
                            pg_loss
                            - ent_coef * entropy_loss
                            + config.vf_coef * v_loss
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            trainable_params, config.max_grad_norm
                        )
                        optimizer.step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"GPU OOM during update. Clearing cache.")
                    torch.cuda.empty_cache()
                    # Skip this update, continue training
                else:
                    raise

            # Log metrics
            elapsed = time.time() - start_time
            sps = int(global_step / max(elapsed, 1))

            log_entry = {
                "global_step": global_step,
                "wall_clock_seconds": elapsed,
                "sps": sps,
                "pg_loss": pg_loss.item(),
                "v_loss": v_loss.item(),
                "entropy": entropy_loss.item(),
                "lr": lr,
                "ent_coef": ent_coef,
                "clipfrac": np.mean(clipfracs) if clipfracs else 0.0,
            }

            # Periodic evaluation
            if global_step % config.eval_interval < batch_size:
                # Check time budget before eval
                time_remaining = config.wall_clock_limit - elapsed
                if time_remaining > 30:  # Only eval if we have >30s left
                    win_rate = evaluate_agent(agent, config, device)
                    log_entry["win_rate"] = win_rate
                    logger.info(
                        f"Step {global_step:>8d} | "
                        f"Win rate: {win_rate:.2%} | "
                        f"SPS: {sps} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Ent: {ent_coef:.4f} | "
                        f"Time: {elapsed:.0f}s"
                    )

                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        save_checkpoint(
                            agent,
                            os.path.join(config.checkpoint_dir, "best.pt"),
                            "best checkpoint",
                        )

                    if (
                        win_rate >= config.target_win_rate
                        and competency_reached_at is None
                    ):
                        competency_reached_at = elapsed
                        logger.info(
                            f"TARGET REACHED: {config.target_win_rate:.0%} win rate "
                            f"at {elapsed:.1f}s ({global_step} steps)"
                        )
                else:
                    logger.info(
                        f"Step {global_step:>8d} | SPS: {sps} | "
                        f"Skipping eval (only {time_remaining:.0f}s left)"
                    )
            else:
                if update % 5 == 0:
                    logger.info(
                        f"Step {global_step:>8d} | SPS: {sps} | "
                        f"Loss: {loss.item():.4f}"
                    )

            training_log.append(log_entry)

            # Periodic checkpoint
            if global_step % config.checkpoint_interval < batch_size:
                save_checkpoint(
                    agent,
                    os.path.join(config.checkpoint_dir, f"step_{global_step}.pt"),
                    f"step {global_step}",
                )

            # Incremental results save every 10 updates
            if update % 10 == 0:
                save_results(
                    config,
                    global_step,
                    start_time,
                    best_win_rate,
                    competency_reached_at,
                    training_log,
                )

    except Exception as e:
        logger.error(f"Training crashed: {e}\n{traceback.format_exc()}")
        # Save whatever we have
        if agent is not None:
            save_checkpoint(
                agent,
                os.path.join(config.checkpoint_dir, "crash_checkpoint.pt"),
                "crash checkpoint",
            )
    finally:
        # Always save final results
        save_results(
            config,
            global_step,
            start_time,
            best_win_rate,
            competency_reached_at,
            training_log,
        )

        # Save final checkpoint
        if agent is not None:
            save_checkpoint(
                agent,
                os.path.join(config.checkpoint_dir, "final.pt"),
                "final checkpoint",
            )

        # Clean up environments
        for env in envs:
            try:
                env.close()
            except Exception:
                pass

        total_time = time.time() - start_time
        logger.info(f"\nTraining complete.")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Best win rate: {best_win_rate:.2%}")
        if competency_reached_at is not None:
            logger.info(
                f"  Time to {config.target_win_rate:.0%}: "
                f"{competency_reached_at:.1f}s"
            )
        else:
            logger.info(
                f"  Did not reach {config.target_win_rate:.0%} win rate"
            )

    return {
        "total_timesteps": global_step,
        "wall_clock_seconds": time.time() - start_time,
        "best_win_rate": best_win_rate,
        "competency_reached_at_seconds": competency_reached_at,
        "training_log": training_log,
    }


def evaluate_agent(agent, config: TrainingConfig, device=None) -> float:
    """
    Evaluate agent against Cyborg AI.

    Returns win rate over config.eval_episodes episodes.
    Includes timeout per episode to prevent hangs.
    """
    if device is None:
        device = next(agent.parameters()).device

    env = None
    try:
        env = make_env(config, env_id=999)
        wins = 0
        total_damage = 0.0
        total_survival = 0

        for ep in range(config.eval_episodes):
            try:
                obs, info = env.reset()
                done = False
                ep_frames = 0
                max_frames = 10000  # Safety limit per episode

                while not done and ep_frames < max_frames:
                    with torch.no_grad():
                        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                        action, _, _, _ = agent.get_action_and_value(obs_tensor)

                    obs, reward, terminated, truncated, info = env.step(
                        action.item()
                    )
                    done = terminated or truncated
                    ep_frames += 1

                total_survival += ep_frames
                if info.get("winner") == 0:
                    wins += 1

            except Exception as e:
                logger.warning(f"Eval episode {ep} failed: {e}")
                # Try to restart env for next episode
                try:
                    env.close()
                except Exception:
                    pass
                env = make_env(config, env_id=999)

        return wins / max(config.eval_episodes, 1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 0.0
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
