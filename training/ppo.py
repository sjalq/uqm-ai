"""
PPO training for UQM Melee - CleanRL single-file style.

Agent 2 - Round 1: Max throughput, preprocessed obs buffer, mixed precision, robust.
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

logger = logging.getLogger(__name__)


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

        try:
            agent = MeleeAgent(
                encoder_type="siglip", hidden_dim=config.hidden_dim,
                action_dim=config.action_dim, encoder_name=config.encoder_name,
                encoder_pretrained=config.encoder_pretrained,
            ).to(device)
            logger.info("Using SigLIP encoder")
        except (ImportError, RuntimeError) as e:
            logger.info(f"SigLIP unavailable ({e}), using CNN encoder")
            agent = MeleeAgent(encoder_type="cnn", hidden_dim=config.hidden_dim, action_dim=config.action_dim).to(device)

        trainable_params = [p for p in agent.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        optimizer = optim.Adam(trainable_params, lr=config.learning_rate, eps=1e-5)

        batch_size = config.num_envs * config.num_steps
        minibatch_size = batch_size // config.num_minibatches
        num_updates = config.total_timesteps // batch_size
        input_size = agent.input_size
        in_channels = 1 if agent.encoder_type != "siglip" else 3

        obs_buf = torch.zeros((config.num_steps, config.num_envs, in_channels, input_size, input_size), dtype=torch.float32)
        actions_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long)
        logprobs_buf = torch.zeros((config.num_steps, config.num_envs))
        rewards_buf = torch.zeros((config.num_steps, config.num_envs))
        dones_buf = torch.zeros((config.num_steps, config.num_envs))
        values_buf = torch.zeros((config.num_steps, config.num_envs))

        next_obs_list = []
        next_done = torch.zeros(config.num_envs)
        for env in envs:
            obs, _ = env.reset()
            next_obs_list.append(obs)
        next_obs_processed = preprocess_obs(np.array(next_obs_list), target_size=input_size, device="cpu")

        budget = getattr(config, "wall_clock_budget", 290.0)
        logger.info(f"Starting PPO (budget: {budget:.0f}s, batch: {batch_size}, updates: {num_updates})")

        use_amp = (device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        for update in range(1, num_updates + 1):
            elapsed = time.time() - start_time
            if elapsed >= budget:
                logger.info(f"Budget reached at {elapsed:.1f}s")
                break

            frac = 1.0 - (update - 1.0) / num_updates
            lr = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lr

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
                        obs, reward, terminated, truncated, info = env.step(action[i].item())
                    except Exception:
                        try:
                            obs, _ = env.reset()
                        except Exception:
                            envs[i] = make_env(config, i)
                            obs, _ = envs[i].reset()
                        reward, terminated, truncated = 0.0, False, False
                    rewards_buf[step, i] = reward
                    next_done[i] = float(terminated or truncated)
                    if terminated or truncated:
                        try:
                            obs, _ = env.reset()
                        except Exception:
                            envs[i] = make_env(config, i)
                            obs, _ = envs[i].reset()
                    next_obs_list.append(obs)

                next_obs_processed = preprocess_obs(np.array(next_obs_list), target_size=input_size, device="cpu")

            with torch.no_grad():
                next_value = agent.get_value(next_obs_processed.to(device)).squeeze(-1).cpu()
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

            b_obs = obs_buf.reshape(-1, in_channels, input_size, input_size)
            b_logprobs = logprobs_buf.reshape(-1)
            b_actions = actions_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_inds = np.arange(batch_size)
            clipfracs = []

            for epoch in range(config.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    mb = b_inds[start:start + minibatch_size]
                    mb_obs = b_obs[mb].to(device)
                    mb_act = b_actions[mb].to(device)
                    mb_lp = b_logprobs[mb].to(device)
                    mb_adv = b_advantages[mb].to(device)
                    mb_ret = b_returns[mb].to(device)
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
                            v_loss = 0.5 * ((nv - mb_ret) ** 2).mean()
                            entropy_loss = ent.mean()
                            loss = pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
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
            log_entry = {
                "global_step": global_step, "wall_clock_seconds": elapsed, "sps": sps,
                "pg_loss": pg_loss.item(), "v_loss": v_loss.item(), "entropy": entropy_loss.item(),
                "lr": lr, "clipfrac": np.mean(clipfracs) if clipfracs else 0.0,
            }

            if global_step % config.eval_interval < batch_size:
                if time.time() - start_time < budget - 30:
                    win_rate, eval_metrics = evaluate_agent(agent, config, device)
                    log_entry["win_rate"] = win_rate
                    log_entry.update(eval_metrics)
                    logger.info(f"Step {global_step:>8d} | Win: {win_rate:.0%} | SPS: {sps} | Loss: {loss.item():.4f} | {elapsed:.0f}s")
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
                logger.info(f"Step {global_step:>8d} | SPS: {sps} | Loss: {loss.item():.4f}")

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
    agent.eval()
    try:
        for ep in range(config.eval_episodes):
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
        try:
            env.close()
        except Exception:
            pass

    n = max(config.eval_episodes, 1)
    return wins / n, {"avg_survival_frames": surv / n, "avg_damage_dealt": dmg_d / n, "avg_damage_taken": dmg_t / n, "games_with_damage": g_dmg}
