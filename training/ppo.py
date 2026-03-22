"""
PPO training for UQM Melee - CleanRL single-file style.

Agent 3 - Round 1: Training loop optimization + curriculum learning.
- Wall-clock budget enforcement (stops 25s before 5min limit)
- Curriculum: restricted action space phase 1, full action space phase 2
- Stateful reward shaping with combo bonuses (RewardShaper)
- GPU-pinned memory for faster transfers
- Entropy annealing, OOM handling, atomic saves
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

logger = logging.getLogger("ppo_agent3")


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)


def make_env(config, env_id=0):
    from uqm_env.melee_env import MeleeEnv
    return MeleeEnv(
        ship_p1=config.ship_p1, ship_p2=config.ship_p2,
        p2_cyborg=config.p2_cyborg, frame_skip=config.frame_skip,
        headless=True, seed=config.ship_p1 * 1000 + env_id,
    )


def parse_curriculum_actions(config):
    try:
        raw = getattr(config, "curriculum_actions_phase1", None)
        if raw is None:
            return None
        return [int(x.strip()) for x in raw.split(",") if 0 <= int(x.strip()) < config.action_dim]
    except (ValueError, AttributeError):
        return None


def save_results_atomic(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"Save failed: {e}")


def save_ckpt(agent, path, label="ckpt"):
    try:
        torch.save(agent.state_dict(), path)
    except Exception as e:
        logger.warning(f"Save {label} failed: {e}")


def train(config):
    setup_logging(config.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    curriculum_actions = parse_curriculum_actions(config)
    phase1_steps = getattr(config, "curriculum_phase1_steps", 80_000)

    envs = []
    agent = None
    best_win_rate = 0.0
    competency_reached_at = None
    training_log = []
    global_step = 0
    start_time = time.time()
    ep_count = 0
    ep_wins = 0
    results_path = os.path.join(config.output_dir, "results.json")
    wall_limit = getattr(config, "wall_clock_limit", 275.0)

    try:
        for i in range(config.num_envs):
            envs.append(make_env(config, i))

        try:
            agent = MeleeAgent(
                encoder_type="siglip", hidden_dim=config.hidden_dim,
                action_dim=config.action_dim, encoder_name=config.encoder_name,
                encoder_pretrained=config.encoder_pretrained,
            ).to(device)
            logger.info("SigLIP encoder")
        except (ImportError, RuntimeError) as e:
            logger.info(f"SigLIP fail ({e}), using CNN")
            agent = MeleeAgent(
                encoder_type="cnn", hidden_dim=config.hidden_dim,
                action_dim=config.action_dim,
            ).to(device)

        curriculum_expanded = False
        if curriculum_actions is not None:
            agent.set_action_mask(curriculum_actions)
            logger.info(f"Curriculum: {len(curriculum_actions)} actions")

        reward_shaper = RewardShaper(config.num_envs)
        trainable_params = [p for p in agent.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=config.learning_rate, eps=1e-5)

        batch_size = config.num_envs * config.num_steps
        minibatch_size = batch_size // config.num_minibatches
        num_updates = config.total_timesteps // batch_size
        pin = device.type == "cuda"

        obs_buf = torch.zeros((config.num_steps, config.num_envs, 240, 320, 3), dtype=torch.uint8, pin_memory=pin)
        actions_buf = torch.zeros((config.num_steps, config.num_envs), dtype=torch.long, pin_memory=pin)
        logprobs_buf = torch.zeros((config.num_steps, config.num_envs), pin_memory=pin)
        rewards_buf = torch.zeros((config.num_steps, config.num_envs), pin_memory=pin)
        dones_buf = torch.zeros((config.num_steps, config.num_envs), pin_memory=pin)
        values_buf = torch.zeros((config.num_steps, config.num_envs), pin_memory=pin)

        next_obs_list = []
        next_done = torch.zeros(config.num_envs)
        for idx, env in enumerate(envs):
            obs, info = env.reset()
            next_obs_list.append(obs)
            reward_shaper.reset_env(idx, info)
        next_obs = torch.from_numpy(np.array(next_obs_list))

        ent_start = config.ent_coef
        ent_final = getattr(config, "ent_coef_final", 0.005)

        logger.info(f"Training: {config.total_timesteps} steps, batch={batch_size}, wall={wall_limit}s")

        for update in range(1, num_updates + 1):
            elapsed = time.time() - start_time
            if elapsed >= wall_limit:
                logger.info(f"Wall limit reached ({elapsed:.1f}s)")
                break

            if not curriculum_expanded and curriculum_actions and global_step >= phase1_steps:
                agent.set_action_mask(None)
                curriculum_expanded = True
                logger.info(f"Phase 2: full actions at step {global_step}")

            frac = 1.0 - (update - 1.0) / num_updates
            lr = frac * config.learning_rate
            optimizer.param_groups[0]["lr"] = lr
            ent_coef = ent_final + frac * (ent_start - ent_final)

            for step in range(config.num_steps):
                global_step += config.num_envs
                obs_buf[step] = next_obs
                dones_buf[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values_buf[step] = value
                actions_buf[step] = action
                logprobs_buf[step] = logprob

                next_obs_list = []
                for i, env in enumerate(envs):
                    try:
                        obs, env_r, term, trunc, info = env.step(action[i].item())
                        shaped = reward_shaper.compute_reward(i, info, term or trunc, env_r)
                        rewards_buf[step, i] = shaped
                        next_done[i] = float(term or trunc)
                        if term or trunc:
                            ep_count += 1
                            if info.get("winner") == 0:
                                ep_wins += 1
                            obs, ri = env.reset()
                            reward_shaper.reset_env(i, ri)
                        next_obs_list.append(obs)
                    except Exception as e:
                        logger.warning(f"Env {i}: {e}")
                        try:
                            envs[i].close()
                        except Exception:
                            pass
                        try:
                            envs[i] = make_env(config, i)
                            obs, ri = envs[i].reset()
                            reward_shaper.reset_env(i, ri)
                        except Exception:
                            obs = np.zeros((240, 320, 3), dtype=np.uint8)
                        next_obs_list.append(obs)
                        rewards_buf[step, i] = 0.0
                        next_done[i] = 1.0
                next_obs = torch.from_numpy(np.array(next_obs_list))

            with torch.no_grad():
                next_value = agent.get_value(next_obs).squeeze(-1)
                advantages = torch.zeros_like(rewards_buf)
                lastgaelam = 0
                for t in reversed(range(config.num_steps)):
                    if t == config.num_steps - 1:
                        nnt = 1.0 - next_done
                        nv = next_value
                    else:
                        nnt = 1.0 - dones_buf[t + 1]
                        nv = values_buf[t + 1]
                    delta = rewards_buf[t] + config.gamma * nv * nnt - values_buf[t]
                    advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nnt * lastgaelam
                returns = advantages + values_buf

            b_obs = obs_buf.reshape(-1, 240, 320, 3)
            b_logprobs = logprobs_buf.reshape(-1)
            b_actions = actions_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

            b_inds = np.arange(batch_size)
            clipfracs = []
            pg_v = v_v = ent_v = 0.0

            try:
                for epoch in range(config.update_epochs):
                    np.random.shuffle(b_inds)
                    for start in range(0, batch_size, minibatch_size):
                        mb = b_inds[start:start + minibatch_size]
                        try:
                            _, nlp, ent, nv = agent.get_action_and_value(b_obs[mb], b_actions[mb])
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                torch.cuda.empty_cache()
                                continue
                            raise
                        lr2 = nlp - b_logprobs[mb]
                        ratio = lr2.exp()
                        with torch.no_grad():
                            clipfracs.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())
                        mb_adv = b_advantages[mb]
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        pg1 = -mb_adv * ratio
                        pg2 = -mb_adv * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                        pg_loss = torch.max(pg1, pg2).mean()
                        v_loss = 0.5 * ((nv - b_returns[mb]) ** 2).mean()
                        ent_loss = ent.mean()
                        loss = pg_loss - ent_coef * ent_loss + config.vf_coef * v_loss
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
                        optimizer.step()
                        pg_v, v_v, ent_v = pg_loss.item(), v_loss.item(), ent_loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                else:
                    raise

            elapsed = time.time() - start_time
            sps = int(global_step / max(elapsed, 1))
            log_entry = {
                "global_step": global_step, "wall_clock_seconds": elapsed,
                "sps": sps, "pg_loss": pg_v, "v_loss": v_v, "entropy": ent_v,
                "lr": lr, "ent_coef": ent_coef,
                "clipfrac": np.mean(clipfracs) if clipfracs else 0.0,
                "episodes": ep_count, "phase": 1 if not curriculum_expanded else 2,
            }

            if global_step % config.eval_interval < batch_size:
                if wall_limit - elapsed > 30:
                    wr, ei = evaluate_agent_detailed(agent, config)
                    log_entry["win_rate"] = wr
                    log_entry["eval_info"] = ei
                    ph = "P1" if not curriculum_expanded else "P2"
                    logger.info(f"Step {global_step:>8d} | Win: {wr:.0%} | SPS: {sps} | {ph} | {elapsed:.0f}s")
                    if wr > best_win_rate:
                        best_win_rate = wr
                        save_ckpt(agent, os.path.join(config.checkpoint_dir, "best.pt"), "best")
                    if wr >= config.target_win_rate and competency_reached_at is None:
                        competency_reached_at = elapsed
                        logger.info(f"TARGET: {config.target_win_rate:.0%} at {elapsed:.1f}s")
                else:
                    logger.info(f"Step {global_step:>8d} | Skip eval ({wall_limit - elapsed:.0f}s left)")
            elif update % 5 == 0:
                logger.info(f"Step {global_step:>8d} | SPS: {sps} | PG: {pg_v:.4f}")

            training_log.append(log_entry)

            if global_step % config.checkpoint_interval < batch_size:
                save_ckpt(agent, os.path.join(config.checkpoint_dir, f"step_{global_step}.pt"), f"s{global_step}")

            if update % 10 == 0:
                save_results_atomic(results_path, {
                    "total_timesteps": global_step, "wall_clock_seconds": elapsed,
                    "best_win_rate": best_win_rate,
                    "competency_reached_at_seconds": competency_reached_at,
                    "episodes": ep_count, "wins": ep_wins,
                    "training_log": training_log[-20:],
                })

    except Exception as e:
        logger.error(f"Crash: {e}\n{traceback.format_exc()}")
        if agent is not None:
            save_ckpt(agent, os.path.join(config.checkpoint_dir, "crash_checkpoint.pt"), "crash")
    finally:
        fe = time.time() - start_time
        save_results_atomic(results_path, {
            "total_timesteps": global_step, "wall_clock_seconds": fe,
            "best_win_rate": best_win_rate,
            "competency_reached_at_seconds": competency_reached_at,
            "episodes": ep_count, "wins": ep_wins, "training_log": training_log,
        })
        if agent is not None:
            save_ckpt(agent, os.path.join(config.checkpoint_dir, "final.pt"), "final")
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
        logger.info(f"Done: {fe:.1f}s, {global_step} steps, best={best_win_rate:.2%}")

    return {
        "total_timesteps": global_step, "wall_clock_seconds": time.time() - start_time,
        "best_win_rate": best_win_rate,
        "competency_reached_at_seconds": competency_reached_at,
        "training_log": training_log,
    }


def evaluate_agent_detailed(agent, config):
    env = None
    try:
        env = make_env(config, env_id=999)
        wins = surv = dmg_d = dmg_t = g_dmg = 0
        for ep in range(config.eval_episodes):
            try:
                obs, info = env.reset()
                done = False
                ip1, ip2 = info.get("p1_crew", 0), info.get("p2_crew", 0)
                fr = 0
                while not done and fr < 10000:
                    with torch.no_grad():
                        a, _, _, _ = agent.get_action_and_value(torch.from_numpy(obs).unsqueeze(0))
                    obs, _, term, trunc, info = env.step(a.item())
                    done = term or trunc
                    fr += 1
                surv += info.get("frame_count", fr)
                d = max(ip2 - info.get("p2_crew", 0), 0)
                dmg_d += d
                dmg_t += max(ip1 - info.get("p1_crew", 0), 0)
                if d > 0:
                    g_dmg += 1
                if info.get("winner") == 0:
                    wins += 1
            except Exception as e:
                logger.warning(f"Eval ep {ep}: {e}")
                try:
                    env.close()
                except Exception:
                    pass
                env = make_env(config, env_id=999)
        n = max(config.eval_episodes, 1)
        return wins / n, {"wins": wins, "avg_survival": surv / n, "avg_dmg_dealt": dmg_d / n,
                          "avg_dmg_taken": dmg_t / n, "games_with_dmg": g_dmg}
    except Exception as e:
        logger.error(f"Eval fail: {e}")
        return 0.0, {}
    finally:
        if env:
            try:
                env.close()
            except Exception:
                pass


def evaluate_agent(agent, config):
    wr, _ = evaluate_agent_detailed(agent, config)
    return wr
