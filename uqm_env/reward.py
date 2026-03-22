"""
Reward shaping for UQM Melee.

This file is MODIFIABLE by competing agents.

Agent 1 - Round 2: Dense reward signals with combo tracking, asymmetric damage scaling,
survival bonuses, and efficiency-scaled terminal rewards.
"""


class RewardShaper:
    """
    Stateful reward shaper that tracks per-episode metrics to provide
    dense training signals beyond the sparse win/loss reward.

    Maintains state per environment to track damage combos, cumulative
    damage dealt/taken, and survival time.
    """

    def __init__(self, num_envs, config=None):
        self.num_envs = num_envs
        # R8A3: Simplified reward - strong crew kill signal, clear win/loss
        self.crew_kill_reward = 0.5      # +0.5 per enemy crew killed
        self.damage_taken_scale = 0.1    # Light penalty for taking damage
        self.win_reward = 3.0            # +3.0 for winning
        self.loss_penalty = -0.5         # -0.5 for losing

        # Per-env state
        self._reset_all()

    def _reset_all(self):
        self.prev_p1_crew = [0] * self.num_envs
        self.prev_p2_crew = [0] * self.num_envs
        self.combo_counter = [0] * self.num_envs       # consecutive steps with damage dealt
        self.steps_since_hit = [999] * self.num_envs   # steps since last hit landed
        self.total_damage_dealt = [0.0] * self.num_envs
        self.total_damage_taken = [0.0] * self.num_envs
        self.episode_steps = [0] * self.num_envs
        self.initial_p1_crew = [0] * self.num_envs
        self.initial_p2_crew = [0] * self.num_envs

    def reset_env(self, env_idx, info):
        """Call when an environment resets. Initializes tracking state."""
        p1_crew = info.get("p1_crew", 0)
        p2_crew = info.get("p2_crew", 0)
        self.prev_p1_crew[env_idx] = p1_crew
        self.prev_p2_crew[env_idx] = p2_crew
        self.initial_p1_crew[env_idx] = p1_crew
        self.initial_p2_crew[env_idx] = p2_crew
        self.combo_counter[env_idx] = 0
        self.steps_since_hit[env_idx] = 999
        self.total_damage_dealt[env_idx] = 0.0
        self.total_damage_taken[env_idx] = 0.0
        self.episode_steps[env_idx] = 0

    def shape_reward(self, env_idx, env_reward, info, terminated):
        """
        R8A3: Simplified reward - clear signals, no noise.
        +0.5 per enemy crew killed, +3.0 win, -0.5 loss.
        """
        p1_crew = info.get("p1_crew", self.prev_p1_crew[env_idx])
        p2_crew = info.get("p2_crew", self.prev_p2_crew[env_idx])

        # Compute frame damage deltas
        damage_dealt = max(0, self.prev_p2_crew[env_idx] - p2_crew)
        damage_taken = max(0, self.prev_p1_crew[env_idx] - p1_crew)

        self.episode_steps[env_idx] += 1

        reward = 0.0

        # +0.5 per enemy crew member killed
        if damage_dealt > 0:
            reward += self.crew_kill_reward * damage_dealt
            self.total_damage_dealt[env_idx] += damage_dealt

        # Light penalty for taking damage
        if damage_taken > 0:
            reward -= self.damage_taken_scale * damage_taken
            self.total_damage_taken[env_idx] += damage_taken

        # Terminal rewards
        if terminated:
            winner = info.get("winner", -1)
            if winner == 0:
                reward += self.win_reward
            elif winner == 1:
                reward += self.loss_penalty

        # Update state for next step
        self.prev_p1_crew[env_idx] = p1_crew
        self.prev_p2_crew[env_idx] = p2_crew

        return reward


def compute_reward(prev_state: dict, curr_state: dict) -> float:
    """
    Legacy per-frame reward function. Kept for backwards compatibility.
    The RewardShaper class above is preferred for training.
    """
    reward = 0.0

    p2_crew_lost = prev_state.get("p2_crew", 0) - curr_state.get("p2_crew", 0)
    if p2_crew_lost > 0:
        max_crew = curr_state.get("p2_max_crew", 1) or 1
        reward += 0.1 * (p2_crew_lost / max_crew)

    p1_crew_lost = prev_state.get("p1_crew", 0) - curr_state.get("p1_crew", 0)
    if p1_crew_lost > 0:
        max_crew = curr_state.get("p1_max_crew", 1) or 1
        reward -= 0.1 * (p1_crew_lost / max_crew)

    reward -= 0.001

    if curr_state.get("done", False):
        winner = curr_state.get("winner", -1)
        if winner == 0:
            reward += 1.0
        elif winner == 1:
            reward -= 1.0

    return reward
