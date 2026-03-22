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
        # Reward weights (can be tuned via config)
        self.damage_dealt_scale = 3.0    # Asymmetric: reward dealing damage heavily
        self.damage_taken_scale = 1.0    # Penalize taking damage less
        self.combo_bonus = 0.05          # Bonus per consecutive-hit step
        self.combo_decay_steps = 5       # Steps before combo resets
        self.survival_bonus = 0.0005     # Small per-step survival reward
        self.win_base = 1.0              # Base win reward
        self.loss_base = -1.0            # Base loss penalty
        self.efficiency_scale = 0.5      # Extra reward for efficient wins
        self.time_penalty = 0.0005       # Small time pressure

        if config is not None:
            self.damage_dealt_scale = getattr(config, "reward_damage_dealt_scale", self.damage_dealt_scale)
            self.damage_taken_scale = getattr(config, "reward_damage_taken_scale", self.damage_taken_scale)
            self.combo_bonus = getattr(config, "reward_combo_bonus", self.combo_bonus)
            self.survival_bonus = getattr(config, "reward_survival_bonus", self.survival_bonus)

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
        Compute shaped reward from env reward and info dict.

        Args:
            env_idx: which environment
            env_reward: raw reward from env.step()
            info: info dict from env.step() with p1_crew, p2_crew, winner
            terminated: whether episode ended

        Returns:
            float: shaped reward
        """
        p1_crew = info.get("p1_crew", self.prev_p1_crew[env_idx])
        p2_crew = info.get("p2_crew", self.prev_p2_crew[env_idx])

        # Compute frame damage deltas
        damage_dealt = max(0, self.prev_p2_crew[env_idx] - p2_crew)
        damage_taken = max(0, self.prev_p1_crew[env_idx] - p1_crew)

        self.episode_steps[env_idx] += 1

        # --- Asymmetric damage rewards ---
        max_p2 = max(self.initial_p2_crew[env_idx], 1)
        max_p1 = max(self.initial_p1_crew[env_idx], 1)

        reward = 0.0

        # Dealing damage is rewarded more heavily (asymmetric)
        if damage_dealt > 0:
            reward += self.damage_dealt_scale * (damage_dealt / max_p2) * 0.1
            self.total_damage_dealt[env_idx] += damage_dealt
            self.steps_since_hit[env_idx] = 0
            self.combo_counter[env_idx] += 1
        else:
            self.steps_since_hit[env_idx] += 1
            if self.steps_since_hit[env_idx] > self.combo_decay_steps:
                self.combo_counter[env_idx] = 0

        # Taking damage is penalized less (asymmetric)
        if damage_taken > 0:
            reward -= self.damage_taken_scale * (damage_taken / max_p1) * 0.1
            self.total_damage_taken[env_idx] += damage_taken

        # --- Combo bonus: consecutive hits get increasing reward ---
        if self.combo_counter[env_idx] > 1:
            combo_mult = min(self.combo_counter[env_idx], 10)  # cap at 10x
            reward += self.combo_bonus * combo_mult

        # --- Survival bonus: staying alive is good ---
        reward += self.survival_bonus

        # --- Small time pressure to encourage engagement ---
        reward -= self.time_penalty

        # --- Terminal rewards: scaled by damage efficiency ---
        if terminated:
            winner = info.get("winner", -1)
            total_dealt = self.total_damage_dealt[env_idx]
            total_taken = self.total_damage_taken[env_idx]

            if winner == 0:
                # Win: base reward + efficiency bonus
                # Efficiency = dealt more damage relative to taken
                efficiency = total_dealt / max(total_dealt + total_taken, 1)
                reward += self.win_base + self.efficiency_scale * efficiency
            elif winner == 1:
                # Loss: base penalty, but reduce penalty if dealt significant damage
                # This encourages agents that at least fight back
                fight_back = min(total_dealt / max_p2, 1.0)
                reward += self.loss_base + 0.3 * fight_back  # less harsh if fought back
            # else: draw or unknown, no terminal bonus

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
