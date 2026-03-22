"""
Reward shaping for UQM Melee.

Agent 3 - Round 1: Stateful reward shaper with dense signals.
Combo bonuses for consecutive hits, survival shaping, aggression incentives.
"""


class RewardShaper:
    """
    Stateful reward shaper that tracks per-environment episode state
    and produces dense reward signals for faster learning.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.prev_p1_crew = [0] * num_envs
        self.prev_p2_crew = [0] * num_envs
        self.prev_p1_energy = [0] * num_envs
        self.p1_max_crew = [1] * num_envs
        self.p2_max_crew = [1] * num_envs
        self.episode_frames = [0] * num_envs
        self.total_damage_dealt = [0.0] * num_envs
        self.last_damage_frame = [0] * num_envs
        self.initialized = [False] * num_envs

    def reset_env(self, env_idx: int, info: dict):
        """Reset tracking for a single environment on episode start."""
        self.prev_p1_crew[env_idx] = info.get("p1_crew", 0)
        self.prev_p2_crew[env_idx] = info.get("p2_crew", 0)
        self.prev_p1_energy[env_idx] = info.get("p1_energy", 0)
        self.p1_max_crew[env_idx] = max(info.get("p1_crew", 1), 1)
        self.p2_max_crew[env_idx] = max(info.get("p2_crew", 1), 1)
        self.episode_frames[env_idx] = 0
        self.total_damage_dealt[env_idx] = 0.0
        self.last_damage_frame[env_idx] = 0
        self.initialized[env_idx] = True

    def compute_reward(self, env_idx: int, info: dict,
                       terminated: bool, env_reward: float) -> float:
        """
        Compute shaped reward for a single environment step.
        Returns a shaped reward with dense signals for faster learning.
        """
        if not self.initialized[env_idx]:
            self.reset_env(env_idx, info)
            return env_reward

        self.episode_frames[env_idx] += 1
        frame = self.episode_frames[env_idx]

        p1_crew = info.get("p1_crew", 0)
        p2_crew = info.get("p2_crew", 0)
        winner = info.get("winner", -1)

        reward = 0.0

        # --- Damage dealt to opponent (primary signal) ---
        p2_crew_lost = self.prev_p2_crew[env_idx] - p2_crew
        if p2_crew_lost > 0:
            max_crew = self.p2_max_crew[env_idx]
            base_dmg_reward = p2_crew_lost / max_crew
            # Combo bonus for consecutive hits within 30 frames
            recency = frame - self.last_damage_frame[env_idx]
            combo_mult = 1.5 if recency < 30 else 1.0
            reward += 0.3 * base_dmg_reward * combo_mult
            self.total_damage_dealt[env_idx] += p2_crew_lost
            self.last_damage_frame[env_idx] = frame

        # --- Damage taken (moderate penalty) ---
        p1_crew_lost = self.prev_p1_crew[env_idx] - p1_crew
        if p1_crew_lost > 0:
            max_crew = self.p1_max_crew[env_idx]
            reward -= 0.15 * (p1_crew_lost / max_crew)

        # --- Energy usage = firing weapons (tiny aggression bonus) ---
        p1_energy = info.get("p1_energy", 0)
        energy_used = self.prev_p1_energy[env_idx] - p1_energy
        if energy_used > 0:
            reward += 0.001

        # --- Survival bonus (small, decays over time) ---
        reward += 0.0003

        # --- Time penalty after initial orientation period ---
        if frame > 40:
            reward -= 0.0002

        # --- Terminal rewards ---
        if terminated:
            if winner == 0:
                crew_ratio = p1_crew / self.p1_max_crew[env_idx] if self.p1_max_crew[env_idx] > 0 else 0
                reward += 2.0 + 0.5 * crew_ratio
            elif winner == 1:
                dmg_ratio = self.total_damage_dealt[env_idx] / self.p2_max_crew[env_idx] if self.p2_max_crew[env_idx] > 0 else 0
                reward -= 1.5 - 0.3 * min(dmg_ratio, 1.0)

        # Update state
        self.prev_p1_crew[env_idx] = p1_crew
        self.prev_p2_crew[env_idx] = p2_crew
        self.prev_p1_energy[env_idx] = p1_energy

        return reward


def compute_reward(prev_state: dict, curr_state: dict) -> float:
    """Legacy stateless reward function (kept for compatibility)."""
    reward = 0.0

    p2_crew_lost = prev_state.get("p2_crew", 0) - curr_state.get("p2_crew", 0)
    if p2_crew_lost > 0:
        max_crew = curr_state.get("p2_max_crew", 1) or 1
        reward += 0.3 * (p2_crew_lost / max_crew)

    p1_crew_lost = prev_state.get("p1_crew", 0) - curr_state.get("p1_crew", 0)
    if p1_crew_lost > 0:
        max_crew = curr_state.get("p1_max_crew", 1) or 1
        reward -= 0.15 * (p1_crew_lost / max_crew)

    reward += 0.0002

    if curr_state.get("done", False):
        winner = curr_state.get("winner", -1)
        if winner == 0:
            reward += 2.0
        elif winner == 1:
            reward -= 1.5

    return reward
