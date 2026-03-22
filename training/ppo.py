"""
PPO training for UQM Melee - CleanRL single-file style.

Round 3 Agent 3: Architecture improvements + value learning + exploration.
- Deeper MLP heads with LayerNorm (Agent 3)
- Cosine LR annealing with warmup (Agent 3)
- Clipped value loss for stable value function updates (Agent 1)
- Running mean/std reward normalization (Agent 1)
- Hash-based exploration bonus for state coverage (Agent 1)
- Batch-level advantage normalization (Agent 1)
"""

import time
import os
import json
import logging
import math
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from training.agent import MeleeAgent, preprocess_obs
from training.config import TrainingConfig
from uqm_env.reward import RewardShaper

logger = logging.getLogger(__name__)


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x):
        """Update with a batch of values (numpy array or scalar)."""
        batch = np.asarray(x).ravel()
        batch_mean = batch.mean()
        batch_var = batch.var()
        batch_count = len(batch)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class ExplorationBonus:
    """Hash-based state visitation counting for exploration bonus."""

    def __init__(self, num_buckets=4096, coef=0.01):
        self.num_buckets = num_buckets
        self.coef = coef
        self.counts = np.zeros(num_buckets, dtype=np.float64)

    def _hash_obs(self, obs_tensor):
        """Hash observation to a bucket. obs_tensor: (C, H, W) float."""
        # Downsample to 8x8 and discretize for fast hashing
        flat = obs_tensor.reshape(-1)
        # Use a subset of pixels for speed
        stride = max(1, len(flat) // 64)
        key_pixels = flat[::stride]
        # Simple hash: discretize to 16 levels, compute polynomial hash
        discretized = (key_pixels * 15).int().numpy().astype(np.int64)
        h = 0
        for v in discretized:
            h = (h * 31 + v) % self.num_buckets
        return int(h)

    def get_bonus(self, obs_batch):
        """
        Compute exploration bonus for a batch of observations.
        obs_batch: (B, C, H, W) tensor on CPU.
        Returns: numpy array of bonuses (B,)
        """
        bonuses = np.zeros(obs_batch.shape[0])
        for i in range(obs_batch.shape[0]):
            h = self._hash_obs(obs_batch[i])
            self.counts[h] += 1.0
            # Bonus proportional to 1/sqrt(count) - decreases with visits
            bonuses[i] = self.coef / np.sqrt(self.counts[h])
        return bonuses


def make_env(config: TrainingConfig, env_id: int = 0):
    from uqm_env.melee_env import MeleeEnv
    return MeleeEnv(
        ship_p1=config.ship_p1, ship_p2=config.ship_p2,
        p2_cyborg=config.p2_cyborg, frame_skip=config.frame_skip,
        headless=True, seed=config.ship_p1 * 1000 + env_id,
    )


def _save_results(config, global_step, start_time, best_win_rate,
                  competency_reached_at, training_log, best_eval_metrics=None):
    results = {
        "total_timesteps": global_step,
        "wall_clock_seconds": time.time() - start_time,
        "best_win_rate": best_win_rate,
        "competency_reached_at_seconds": competency_reached_at,
        "training_log": training_log,
    }
    if best_eval_metrics:
        results["best_eval_metrics"] = best_eval_metrics
    try:
        tmp = os.path.join(config.output_dir, "results.json.tmp")
        with open(tmp, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp, os.path.join(config.output_dir, "results.json"))
    except Exception as e:
        logger.warning(f"Failed to save results.json: {e}")
    return results


def train(config: TrainingConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(config.log_dir, "training.log")), logging.StreamHandler()],
        force=True,
    )

    global_step = 0
    best_win_rate = 0.0
    competency_reached_at = None
    training_log = []
    best_eval_metrics = None
    start_time = time.time()
    envs = []
    agent = None

    try:
        for i in range(config.num_envs):
            envs.append(make_env(config, i))

        frame_stack = getattr(config, "frame_stack", 4)

        try:
            agent = MeleeAgent(
                encoder_type="siglip", hidden_dim=config.hidden_dim,
                action_dim=config.action_dim, encoder_name=config.encoder_name,
                encoder_pretrained=config.encoder_pretrained,
                frame_stack=frame_stack,
            ).to(device)
            logger.info("Using SigLIP encoder")
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.info(f"SigLIP unavailable ({e}), using CNN encoder")
            agent = MeleeAgent(
                encoder_type="cnn", hidden_dim=config.hidden_dim,
                action_dim=config.action_dim, frame_stack=frame_stack,
            ).to(device)

        trainable_params = [p for p in agent.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        optimizer = optim.Adam(trainable_params, lr=config.learning_rate, eps=1e-5)

        batch_size = config.num_envs * config.num_steps
        minibatch_size = batch_size // config.num_minibatches
        num_updates = config.total_timesteps // batch_size
        input_size = agent.input_size
        if agent.encoder_type == "siglip":
            obs_channels = 3
        else:
            obs_channels = frame_stack

        obs_buf = torch.zeros((config.num_steps, config.num_envs, obs_channels, input_size, input_size), dtype=torch.float32)
        actions_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long)
        logprobs_buf = torch.zeros((config.num_steps, config.num_envs))
        rewards_buf = torch.zeros((config.num_steps, config.num_envs))
        dones_buf = torch.zeros((config.num_steps, config.num_envs))
        values_buf = torch.zeros((config.num_steps, config.num_envs))

        reward_shaper = RewardShaper(config.num_envs, config)

        # Round 3 Agent 1: reward normalization + exploration bonus
        reward_rms = RunningMeanStd() if getattr(config, "reward_normalization", False) else None
        exploration = ExplorationBonus(
            num_buckets=getattr(config, "exploration_hash_buckets", 4096),
            coef=getattr(config, "exploration_bonus_coef", 0.01),
        ) if getattr(config, "exploration_bonus", False) else None
        lr_warmup_steps = getattr(config, "lr_warmup_steps", 0)
        clip_vloss = getattr(config, "clip_vloss", False)
        batch_adv_norm = getattr(config, "batch_advantage_norm", False)

        # Frame stacking: maintain per-env frame buffers for CNN path
        # Each buffer is (frame_stack, H, W) on CPU
        frame_bufs = None
        if frame_stack > 1 and agent.encoder_type != "siglip":
            frame_bufs = torch.zeros((config.num_envs, frame_stack, input_size, input_size), dtype=torch.float32)

        next_obs_list = []
        next_done = torch.zeros(config.num_envs)
        for i, env in enumerate(envs):
            obs, info = env.reset()
            next_obs_list.append(obs)
            reward_shaper.reset_env(i, info)
        # Preprocess to single-channel grayscale
        next_obs_single = preprocess_obs(np.array(next_obs_list), target_size=input_size, device="cpu")
        # Initialize frame buffers by repeating first frame
        if frame_bufs is not None:
            for i in range(config.num_envs):
                frame_bufs[i] = next_obs_single[i, 0:1].expand(frame_stack, -1, -1)
            next_obs_processed = frame_bufs.clone()
        else:
            next_obs_processed = next_obs_single

        # Curriculum learning: start with restricted combat actions, expand later
        has_curriculum = hasattr(config, "curriculum_phase1_steps") and hasattr(config, "combat_actions")
        if has_curriculum:
            agent.set_action_mask(config.combat_actions)
            logger.info(f"Curriculum phase 1: {len(config.combat_actions)} actions until step {config.curriculum_phase1_steps}")
        curriculum_expanded = False

        # Entropy annealing setup
        ent_coef_start = config.ent_coef
        ent_coef_final = getattr(config, "ent_coef_final", config.ent_coef)

        budget = getattr(config, "wall_clock_budget", 290.0)
        logger.info(f"Starting PPO (budget: {budget:.0f}s, batch: {batch_size}, updates: {num_updates}, frame_stack: {frame_stack})")
        logger.info(f"Entropy annealing: {ent_coef_start} -> {ent_coef_final}")
        logger.info(f"R3A1: clip_vloss={clip_vloss}, reward_norm={reward_rms is not None}, "
                     f"exploration={exploration is not None}, batch_adv_norm={batch_adv_norm}, "
                     f"lr_warmup={lr_warmup_steps}")

        use_amp = (device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for update in range(1, num_updates + 1):
            elapsed = time.time() - start_time
            if elapsed >= budget:
                logger.info(f"Budget reached at {elapsed:.1f}s")
                break

            # R3A3: Cosine LR annealing with warmup (better than linear for short training)
            progress = (update - 1.0) / max(num_updates, 1)
            if lr_warmup_steps > 0 and global_step < lr_warmup_steps:
                # Linear warmup phase
                lr = config.learning_rate * (global_step / lr_warmup_steps)
            else:
                # Cosine decay from peak to ~0
                lr = config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
            optimizer.param_groups[0]["lr"] = lr

            # Entropy annealing: linear decay from start to final
            current_ent_coef = ent_coef_start + (ent_coef_final - ent_coef_start) * (1.0 - frac)

            # Curriculum: expand to full action space after phase 1
            if has_curriculum and not curriculum_expanded and global_step >= config.curriculum_phase1_steps:
                agent.set_action_mask(None)
                curriculum_expanded = True
                logger.info(f"Curriculum phase 2: all {config.action_dim} actions unlocked at step {global_step}")

            for step in range(config.num_steps):
                global_step += config.num_envs
                obs_buf[step] = next_obs_processed
                dones_buf[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs_processed.to(device))
                    values_buf[step] = value.cpu()
                actions_buf[step] = action.cpu()
                logprobs_buf[step] = logprob.cpu()

                next_obs_list = []
                for i, env in enumerate(envs):
                    try:
                        obs, raw_reward, terminated, truncated, info = env.step(action[i].item())
                    except Exception:
                        try:
                            obs, info = env.reset()
                        except Exception:
                            envs[i] = make_env(config, i)
                            obs, info = envs[i].reset()
                        raw_reward, terminated, truncated = 0.0, False, False
                        reward_shaper.reset_env(i, info)
                    done = terminated or truncated
                    shaped_reward = reward_shaper.shape_reward(i, raw_reward, info, done)
                    # Exploration bonus: reward visiting novel states
                    if exploration is not None:
                        exp_bonus = exploration.get_bonus(next_obs_processed[i:i+1])
                        shaped_reward += exp_bonus[0]
                    rewards_buf[step, i] = shaped_reward
                    next_done[i] = float(done)
                    if done:
                        try:
                            obs, info = env.reset()
                        except Exception:
                            envs[i] = make_env(config, i)
                            obs, info = envs[i].reset()
                        reward_shaper.reset_env(i, info)
                    next_obs_list.append(obs)

                next_obs_single = preprocess_obs(np.array(next_obs_list), target_size=input_size, device="cpu")
                if frame_bufs is not None:
                    for i in range(config.num_envs):
                        if next_done[i]:
                            # Reset frame buffer on episode end
                            frame_bufs[i] = next_obs_single[i, 0:1].expand(frame_stack, -1, -1)
                        else:
                            # Shift left (drop oldest), append new frame
                            frame_bufs[i] = torch.cat([frame_bufs[i, 1:], next_obs_single[i, 0:1]], dim=0)
                    next_obs_processed = frame_bufs.clone()
                else:
                    next_obs_processed = next_obs_single

            # Reward normalization: update running stats and normalize
            if reward_rms is not None:
                reward_rms.update(rewards_buf.numpy())
                normalized_rewards = torch.tensor(
                    reward_rms.normalize(rewards_buf.numpy()), dtype=torch.float32
                )
            else:
                normalized_rewards = rewards_buf

            with torch.no_grad():
                next_value = agent.get_value(next_obs_processed.to(device)).squeeze(-1).cpu()
                advantages = torch.zeros_like(rewards_buf)
                lastgaelam = 0
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nnt, nv = 1.0 - next_done, next_value
                    else:
                        nnt, nv = 1.0 - dones_buf[t + 1], values_buf[t + 1]
                    delta = normalized_rewards[t] + config.gamma * nv * nnt - values_buf[t]
                    advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nnt * lastgaelam
                returns = advantages + values_buf

            b_obs = obs_buf.reshape(-1, obs_channels, input_size, input_size)
            b_logprobs = logprobs_buf.reshape(-1)
            b_actions = actions_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)
            b_inds = np.arange(batch_size)
            clipfracs = []

            # Batch-level advantage normalization (more stable than per-minibatch)
            if batch_adv_norm:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            for epoch in range(config.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    mb = b_inds[start:start + minibatch_size]
                    mb_obs = b_obs[mb].to(device)
                    mb_act = b_actions[mb].to(device)
                    mb_lp = b_logprobs[mb].to(device)
                    mb_adv = b_advantages[mb].to(device)
                    mb_ret = b_returns[mb].to(device)
                    mb_val = b_values[mb].to(device) if clip_vloss else None
                    try:
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            _, nlp, ent, nv = agent.get_action_and_value(mb_obs, mb_act)
                            ratio = (nlp - mb_lp).exp()
                            with torch.no_grad():
                                clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())
                            # Use pre-normalized advantages if batch norm is on, else per-minibatch
                            if batch_adv_norm:
                                adv = mb_adv
                            else:
                                adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                            pg1 = -adv * ratio
                            pg2 = -adv * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                            pg_loss = torch.max(pg1, pg2).mean()
                            # Clipped value loss: prevent large value function updates
                            if clip_vloss and mb_val is not None:
                                v_clipped = mb_val + torch.clamp(nv - mb_val, -config.clip_coef, config.clip_coef)
                                v_loss_unclipped = (nv - mb_ret) ** 2
                                v_loss_clipped = (v_clipped - mb_ret) ** 2
                                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                            else:
                                v_loss = 0.5 * ((nv - mb_ret) ** 2).mean()
                            entropy_loss = ent.mean()
                            loss = pg_loss - current_ent_coef * entropy_loss + config.vf_coef * v_loss
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.warning("GPU OOM, skipping")
                            torch.cuda.empty_cache()
                            optimizer.zero_grad()
                            continue
                        raise

            elapsed = time.time() - start_time
            sps = int(global_step / max(elapsed, 1))
            phase_str = "Ph1" if (has_curriculum and not curriculum_expanded) else "Ph2"
            log_entry = {
                "global_step": global_step, "wall_clock_seconds": elapsed, "sps": sps,
                "pg_loss": pg_loss.item(), "v_loss": v_loss.item(), "entropy": entropy_loss.item(),
                "lr": lr, "ent_coef": current_ent_coef, "phase": phase_str,
                "clipfrac": np.mean(clipfracs) if clipfracs else 0.0,
            }

            if global_step % config.eval_interval < batch_size:
                if time.time() - start_time < budget - 30:
                    win_rate, eval_metrics = evaluate_agent(agent, config, device)
                    log_entry["win_rate"] = win_rate
                    log_entry.update(eval_metrics)
                    logger.info(f"Step {global_step:>8d} | Win: {win_rate:.0%} | SPS: {sps} | Loss: {loss.item():.4f} | Ent: {current_ent_coef:.4f} | {phase_str} | {elapsed:.0f}s")
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_eval_metrics = eval_metrics
                        torch.save(agent.state_dict(), os.path.join(config.checkpoint_dir, "best.pt"))
                    if win_rate >= config.target_win_rate and competency_reached_at is None:
                        competency_reached_at = elapsed
                        logger.info(f"TARGET: {config.target_win_rate:.0%} at {elapsed:.1f}s")
                else:
                    logger.info(f"Step {global_step:>8d} | SPS: {sps} | Skip eval (budget low)")
            elif update % 10 == 0:
                logger.info(f"Step {global_step:>8d} | SPS: {sps} | Loss: {loss.item():.4f} | Ent: {current_ent_coef:.4f} | {phase_str}")

            training_log.append(log_entry)
            if update % 10 == 0:
                _save_results(config, global_step, start_time, best_win_rate, competency_reached_at, training_log, best_eval_metrics)
            if global_step % config.checkpoint_interval < batch_size:
                torch.save(agent.state_dict(), os.path.join(config.checkpoint_dir, f"step_{global_step}.pt"))

    except Exception as e:
        logger.error(f"Crash: {e}\n{traceback.format_exc()}")
        if agent is not None:
            try:
                torch.save(agent.state_dict(), os.path.join(config.checkpoint_dir, "crash_checkpoint.pt"))
            except Exception:
                pass

    finally:
        results = _save_results(config, global_step, start_time, best_win_rate, competency_reached_at, training_log, best_eval_metrics)
        if agent is not None:
            # Clear action mask for clean eval-ready checkpoint
            agent.set_action_mask(None)
            try:
                torch.save(agent.state_dict(), os.path.join(config.checkpoint_dir, "final.pt"))
            except Exception:
                pass
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        logger.info(f"Done. {results['wall_clock_seconds']:.1f}s, best={best_win_rate:.2%}")

    return results


def evaluate_agent(agent, config: TrainingConfig, device=None) -> tuple:
    if device is None:
        device = next(agent.parameters()).device
    try:
        env = make_env(config, env_id=999)
    except Exception as e:
        logger.warning(f"Eval env failed: {e}")
        return 0.0, {}

    wins = surv = dmg_d = dmg_t = g_dmg = 0
    # Save and clear action mask for evaluation (use all actions)
    saved_mask = agent._action_mask
    agent.set_action_mask(None)
    agent.eval()
    try:
        for ep in range(config.eval_episodes):
            # Reset frame buffer for clean eval episodes
            if hasattr(agent, "reset_frame_buffer"):
                agent.reset_frame_buffer()
            try:
                obs, info = env.reset()
            except Exception:
                try:
                    env.close()
                except Exception:
                    pass
                env = make_env(config, env_id=999)
                obs, info = env.reset()
            done, frames = False, 0
            ip2, ip1 = info.get("p2_crew", 0), info.get("p1_crew", 0)
            while not done:
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(torch.from_numpy(obs).unsqueeze(0))
                try:
                    obs, _, terminated, truncated, info = env.step(action.item())
                except Exception:
                    break
                done = terminated or truncated
                frames += 1
            surv += frames
            dd = max(0, ip2 - info.get("p2_crew", ip2))
            dt = max(0, ip1 - info.get("p1_crew", ip1))
            dmg_d += dd
            dmg_t += dt
            if dd > 0:
                g_dmg += 1
            if info.get("winner") == 0:
                wins += 1
    finally:
        agent.train()
        # Restore action mask for training
        agent._action_mask = saved_mask
        try:
            env.close()
        except Exception:
            pass

    n = max(config.eval_episodes, 1)
    return wins / n, {"avg_survival_frames": surv / n, "avg_damage_dealt": dmg_d / n, "avg_damage_taken": dmg_t / n, "games_with_damage": g_dmg}
