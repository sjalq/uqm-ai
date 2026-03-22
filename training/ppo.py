"""
PPO training for UQM Melee - CleanRL single-file style.

Round 4 Agent 1: Integrated best practices from R3 losers.
- RunningMeanStd reward normalization (R3A1)
- Clipped value loss (R3A1)
- LR warmup + cosine annealing (R3A1 + R3A3)
- All R3A2 throughput preserved: GPU preprocessing, torch.compile, pin memory
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

from training.agent import MeleeAgent, preprocess_obs, preprocess_obs_gpu
from training.config import TrainingConfig
from uqm_env.reward import RewardShaper

logger = logging.getLogger(__name__)


class RunningMeanStd:
    """Welford's online algorithm for running mean/variance. Used for reward normalization (R3A1)."""

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

        use_layernorm = getattr(config, "use_layernorm", False)
        deep_heads = getattr(config, "deep_heads", False)
        try:
            agent = MeleeAgent(
                encoder_type="siglip", hidden_dim=config.hidden_dim,
                action_dim=config.action_dim, encoder_name=config.encoder_name,
                encoder_pretrained=config.encoder_pretrained,
                frame_stack=frame_stack,
                use_layernorm=use_layernorm, deep_heads=deep_heads,
            ).to(device)
            logger.info("Using SigLIP encoder")
        except (ImportError, RuntimeError) as e:
            logger.info(f"SigLIP unavailable ({e}), using CNN encoder")
            agent = MeleeAgent(
                encoder_type="cnn", hidden_dim=config.hidden_dim,
                action_dim=config.action_dim, frame_stack=frame_stack,
                use_layernorm=use_layernorm, deep_heads=deep_heads,
            ).to(device)

        # R3A2: Apply torch.compile for kernel fusion speedup
        if getattr(config, "use_torch_compile", True):
            agent.try_compile()
            if agent._compiled:
                logger.info("torch.compile applied to encoder")

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

        # R3A2: Pin memory for fast async GPU transfer
        use_pin = getattr(config, "pin_memory", True) and device.type == "cuda"
        obs_buf = torch.zeros((config.num_steps, config.num_envs, obs_channels, input_size, input_size), dtype=torch.float32, pin_memory=use_pin)
        actions_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, pin_memory=use_pin)
        logprobs_buf = torch.zeros((config.num_steps, config.num_envs), pin_memory=use_pin)
        rewards_buf = torch.zeros((config.num_steps, config.num_envs), pin_memory=use_pin)
        dones_buf = torch.zeros((config.num_steps, config.num_envs), pin_memory=use_pin)
        values_buf = torch.zeros((config.num_steps, config.num_envs), pin_memory=use_pin)

        # R4A1: RunningMeanStd reward normalization (R3A1)
        use_reward_norm = getattr(config, "use_reward_normalization", False)
        reward_rms = RunningMeanStd() if use_reward_norm else None

        reward_shaper = RewardShaper(config.num_envs, config)

        # R3A2: GPU-side frame stacking buffer
        use_gpu_preprocess = getattr(config, "gpu_preprocess", True) and device.type == "cuda"
        frame_bufs = None
        if frame_stack > 1 and agent.encoder_type != "siglip":
            if use_gpu_preprocess:
                frame_bufs = torch.zeros((config.num_envs, frame_stack, input_size, input_size), dtype=torch.float32, device=device)
            else:
                frame_bufs = torch.zeros((config.num_envs, frame_stack, input_size, input_size), dtype=torch.float32)

        next_obs_list = []
        next_done = torch.zeros(config.num_envs)
        for i, env in enumerate(envs):
            obs, info = env.reset()
            next_obs_list.append(obs)
            reward_shaper.reset_env(i, info)

        # Initial preprocessing
        if use_gpu_preprocess and frame_bufs is not None:
            raw_obs_tensor = torch.from_numpy(np.array(next_obs_list)).to(device, non_blocking=True)
            next_obs_single = preprocess_obs_gpu(raw_obs_tensor, target_size=input_size)
            for i in range(config.num_envs):
                frame_bufs[i] = next_obs_single[i, 0:1].expand(frame_stack, -1, -1)
            next_obs_processed = frame_bufs.clone()
        else:
            next_obs_single = preprocess_obs(np.array(next_obs_list), target_size=input_size, device="cpu")
            if frame_bufs is not None:
                for i in range(config.num_envs):
                    frame_bufs[i] = next_obs_single[i, 0:1].expand(frame_stack, -1, -1)
                next_obs_processed = frame_bufs.clone()
            else:
                next_obs_processed = next_obs_single

        # Curriculum learning
        has_curriculum = hasattr(config, "curriculum_phase1_steps") and hasattr(config, "combat_actions")
        if has_curriculum:
            agent.set_action_mask(config.combat_actions)
            logger.info(f"Curriculum phase 1: {len(config.combat_actions)} actions until step {config.curriculum_phase1_steps}")
        curriculum_expanded = False

        # Entropy annealing setup
        ent_coef_start = config.ent_coef
        ent_coef_final = getattr(config, "ent_coef_final", config.ent_coef)

        # R4A1: LR schedule config
        use_cosine_lr = getattr(config, "use_cosine_lr", False)
        use_lr_warmup = getattr(config, "use_lr_warmup", False)
        lr_warmup_frac = getattr(config, "lr_warmup_frac", 0.05)
        warmup_updates = int(num_updates * lr_warmup_frac) if use_lr_warmup else 0
        use_clipped_vloss = getattr(config, "use_clipped_vloss", False)

        budget = getattr(config, "wall_clock_budget", 290.0)
        logger.info(f"Starting PPO (budget: {budget:.0f}s, batch: {batch_size}, updates: {num_updates}, frame_stack: {frame_stack}, num_envs: {config.num_envs})")
        logger.info(f"Entropy annealing: {ent_coef_start} -> {ent_coef_final}")
        logger.info(f"R4A1: layernorm={use_layernorm}, deep_heads={deep_heads}, cosine_lr={use_cosine_lr}, lr_warmup={use_lr_warmup}({warmup_updates}), reward_norm={use_reward_norm}, clipped_vloss={use_clipped_vloss}")
        logger.info(f"R3A2: gpu_preprocess={use_gpu_preprocess}, pin_memory={use_pin}, torch.compile={agent._compiled}")

        use_amp = (device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # R3A2: Pre-allocate numpy array for raw observations
        raw_obs_np = np.zeros((config.num_envs,) + next_obs_list[0].shape, dtype=next_obs_list[0].dtype)

        for update in range(1, num_updates + 1):
            elapsed = time.time() - start_time
            if elapsed >= budget:
                logger.info(f"Budget reached at {elapsed:.1f}s")
                break

            # R4A1: LR schedule with warmup + cosine/linear decay
            progress = (update - 1.0) / max(num_updates - 1, 1)
            if use_lr_warmup and update <= warmup_updates:
                # Linear warmup
                lr = config.learning_rate * (update / max(warmup_updates, 1))
            elif use_cosine_lr:
                # Cosine annealing after warmup (R3A3)
                if warmup_updates > 0:
                    cosine_progress = (update - warmup_updates) / max(num_updates - warmup_updates, 1)
                else:
                    cosine_progress = progress
                lr = config.learning_rate * 0.5 * (1.0 + np.cos(np.pi * cosine_progress))
            else:
                # Original linear decay
                frac = 1.0 - progress
                lr = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lr

            # Entropy annealing: linear decay from start to final
            current_ent_coef = ent_coef_start + (ent_coef_final - ent_coef_start) * progress

            # Curriculum: expand to full action space after phase 1
            if has_curriculum and not curriculum_expanded and global_step >= config.curriculum_phase1_steps:
                agent.set_action_mask(None)
                curriculum_expanded = True
                logger.info(f"Curriculum phase 2: all {config.action_dim} actions unlocked at step {global_step}")

            for step in range(config.num_steps):
                global_step += config.num_envs

                if use_gpu_preprocess and frame_bufs is not None:
                    obs_buf[step] = next_obs_processed.cpu()
                else:
                    obs_buf[step] = next_obs_processed
                dones_buf[step] = next_done

                with torch.no_grad():
                    if use_gpu_preprocess and frame_bufs is not None:
                        obs_gpu = next_obs_processed
                    else:
                        obs_gpu = next_obs_processed.to(device, non_blocking=use_pin)
                    action, logprob, _, value = agent.get_action_and_value(obs_gpu)
                    values_buf[step] = value.cpu()
                actions_buf[step] = action.cpu()
                logprobs_buf[step] = logprob.cpu()

                # R3A2: Collect actions once, step all envs with pre-allocated buffer
                actions_np = action.cpu().numpy()
                for i, env in enumerate(envs):
                    try:
                        obs, raw_reward, terminated, truncated, info = env.step(int(actions_np[i]))
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
                    rewards_buf[step, i] = float(shaped_reward)
                    next_done[i] = float(done)
                    if done:
                        try:
                            obs, info = env.reset()
                        except Exception:
                            envs[i] = make_env(config, i)
                            obs, info = envs[i].reset()
                        reward_shaper.reset_env(i, info)
                    raw_obs_np[i] = obs

                # R3A2: GPU-side preprocessing for frame stacking
                if use_gpu_preprocess and frame_bufs is not None:
                    raw_obs_tensor = torch.from_numpy(raw_obs_np).to(device, non_blocking=True)
                    next_obs_single = preprocess_obs_gpu(raw_obs_tensor, target_size=input_size)
                    for i in range(config.num_envs):
                        if next_done[i]:
                            frame_bufs[i] = next_obs_single[i, 0:1].expand(frame_stack, -1, -1)
                        else:
                            # Roll + overwrite last slot - faster than cat
                            frame_bufs[i] = frame_bufs[i].roll(-1, dims=0)
                            frame_bufs[i, -1] = next_obs_single[i, 0]
                    next_obs_processed = frame_bufs.clone()
                else:
                    next_obs_single = preprocess_obs(raw_obs_np, target_size=input_size, device="cpu")
                    if frame_bufs is not None:
                        for i in range(config.num_envs):
                            if next_done[i]:
                                frame_bufs[i] = next_obs_single[i, 0:1].expand(frame_stack, -1, -1)
                            else:
                                frame_bufs[i] = torch.cat([frame_bufs[i, 1:], next_obs_single[i, 0:1]], dim=0)
                        next_obs_processed = frame_bufs.clone()
                    else:
                        next_obs_processed = next_obs_single

            # R4A1: Reward normalization (R3A1)
            if reward_rms is not None:
                step_rewards = rewards_buf.numpy()
                reward_rms.update(step_rewards.ravel())
                rewards_buf = torch.tensor(reward_rms.normalize(step_rewards), dtype=torch.float32)

            # GAE computation
            with torch.no_grad():
                if use_gpu_preprocess and frame_bufs is not None:
                    next_value = agent.get_value(next_obs_processed).squeeze(-1).cpu()
                else:
                    next_value = agent.get_value(next_obs_processed.to(device, non_blocking=use_pin)).squeeze(-1).cpu()
                advantages = torch.zeros_like(rewards_buf)
                lastgaelam = 0
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nnt, nv = 1.0 - next_done, next_value
                    else:
                        nnt, nv = 1.0 - dones_buf[t + 1], values_buf[t + 1]
                    delta = rewards_buf[t] + config.gamma * nv * nnt - values_buf[t]
                    advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nnt * lastgaelam
                returns = advantages + values_buf

            b_obs = obs_buf.reshape(-1, obs_channels, input_size, input_size)
            b_logprobs = logprobs_buf.reshape(-1)
            b_actions = actions_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_buf.reshape(-1)  # R4A1: needed for clipped value loss
            b_inds = np.arange(batch_size)
            clipfracs = []

            for epoch in range(config.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    mb = b_inds[start:start + minibatch_size]
                    # R3A2: non_blocking transfer for pinned memory
                    mb_obs = b_obs[mb].to(device, non_blocking=use_pin)
                    mb_act = b_actions[mb].to(device, non_blocking=use_pin)
                    mb_lp = b_logprobs[mb].to(device, non_blocking=use_pin)
                    mb_adv = b_advantages[mb].to(device, non_blocking=use_pin)
                    mb_ret = b_returns[mb].to(device, non_blocking=use_pin)
                    mb_val = b_values[mb].to(device, non_blocking=use_pin)
                    try:
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            _, nlp, ent, nv = agent.get_action_and_value(mb_obs, mb_act)
                            ratio = (nlp - mb_lp).exp()
                            with torch.no_grad():
                                clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())
                            adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                            pg1 = -adv * ratio
                            pg2 = -adv * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                            pg_loss = torch.max(pg1, pg2).mean()
                            # R4A1: Clipped value loss (R3A1) - prevents large value function updates
                            if use_clipped_vloss:
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
    saved_mask = agent._action_mask
    agent.set_action_mask(None)
    agent.eval()
    try:
        for ep in range(config.eval_episodes):
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
        agent._action_mask = saved_mask
        try:
            env.close()
        except Exception:
            pass

    n = max(config.eval_episodes, 1)
    return wins / n, {"avg_survival_frames": surv / n, "avg_damage_dealt": dmg_d / n, "avg_damage_taken": dmg_t / n, "games_with_damage": g_dmg}
