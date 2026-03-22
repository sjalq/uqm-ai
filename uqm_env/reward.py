"""
Reward shaping for UQM Melee.

This file is MODIFIABLE by competing agents.

Agent 1 - Round 8: Aggressive reward shaping.
+1.0 per enemy crew killed, -0.3 per own crew lost, +5.0 win, -1.0 loss, -0.01 per step.
Simple, strong gradients to learn fast in limited steps.
"""


class RewardShaper:
    """
    Aggressive reward shaper focused on maximizing damage output.
    Large per-crew-kill rewards create strong learning signal even with few steps.
    """

    def __init__(self, num_envs, config=None):
        self.num_envs = num_envs
        self.crew_kill_reward = 1.0       # +1.0 per enemy crew killed
        self.crew_lost_penalty = -0.3     # -0.3 per own crew lost
        self.win_reward = 5.0             # +5.0 for winning
        self.loss_penalty = -1.0          # -1.0 for losing
        self.step_penalty = -0.01         # -0.01 per step (urgency)

        # Per-env state
        self.prev_p1_crew = [0] * num_envs
        self.prev_p2_crew = [0] * num_envs

    def reset_env(self, env_idx, info):
        """Call when an environment resets."""
        self.prev_p1_crew[env_idx] = info.get("p1_crew", 0)
        self.prev_p2_crew[env_idx] = info.get("p2_crew", 0)

    def shape_reward(self, env_idx, env_reward, info, terminated):
        """
        Compute shaped reward. Simple and aggressive.
        """
        p1_crew = info.get("p1_crew", self.prev_p1_crew[env_idx])
        p2_crew = info.get("p2_crew", self.prev_p2_crew[env_idx])

        damage_dealt = max(0, self.prev_p2_crew[env_idx] - p2_crew)
        damage_taken = max(0, self.prev_p1_crew[env_idx] - p1_crew)

        reward = 0.0

        # Per-crew rewards (absolute, not normalized)
        reward += self.crew_kill_reward * damage_dealt
        reward += self.crew_lost_penalty * damage_taken

        # Step penalty for urgency
        reward += self.step_penalty

        # Terminal rewards
        if terminated:
            winner = info.get("winner", -1)
            if winner == 0:
                reward += self.win_reward
            elif winner == 1:
                reward += self.loss_penalty

        # Update state
        self.prev_p1_crew[env_idx] = p1_crew
        self.prev_p2_crew[env_idx] = p2_crew

        return reward


def compute_reward(prev_state: dict, curr_state: dict) -> float:
    """Legacy per-frame reward function. Kept for backwards compatibility."""
    reward = 0.0

    p2_crew_lost = prev_state.get("p2_crew", 0) - curr_state.get("p2_crew", 0)
    if p2_crew_lost > 0:
        reward += 1.0 * p2_crew_lost

    p1_crew_lost = prev_state.get("p1_crew", 0) - curr_state.get("p1_crew", 0)
    if p1_crew_lost > 0:
        reward -= 0.3 * p1_crew_lost

    reward -= 0.01

    if curr_state.get("done", False):
        winner = curr_state.get("winner", -1)
        if winner == 0:
            reward += 5.0
        elif winner == 1:
            reward -= 1.0

    return reward
