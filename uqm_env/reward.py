"""
Reward shaping for UQM Melee.

This file is MODIFIABLE by competing agents.

R9 Agent 3: Tuned for frame_skip=16 (each step = 16 game frames).
With fewer agent steps per battle (~5 steps), rewards are scaled up per-step
to maintain strong gradients. Step penalty reduced since each step spans more time.
"""


class RewardShaper:
    """
    Reward shaper tuned for high frame_skip (16).
    With ~5 agent steps per battle, each step's reward signal must be strong.
    """

    def __init__(self, num_envs, config=None):
        self.num_envs = num_envs
        self.crew_kill_reward = 2.0       # R9A3: stronger per-kill reward (fewer steps to learn from)
        self.crew_lost_penalty = -0.5     # R9A3: stronger penalty
        self.win_reward = 8.0             # R9A3: stronger terminal signal
        self.loss_penalty = -2.0          # R9A3: stronger loss penalty
        self.step_penalty = -0.005        # R9A3: reduced step penalty (each step = 16 frames now)

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
