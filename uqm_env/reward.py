"""
Reward shaping for UQM Melee.

This file is MODIFIABLE by competing agents.
Agents can change the reward function to improve training efficiency.
"""


def compute_reward(prev_state: dict, curr_state: dict) -> float:
    """
    Compute per-frame reward for player 1.

    Args:
        prev_state: dict with p1_crew, p2_crew from previous frame
        curr_state: dict with p1_crew, p2_crew, done, winner from current frame

    Returns:
        float reward value
    """
    reward = 0.0

    # Crew damage dealt to opponent (normalized)
    p2_crew_lost = prev_state.get("p2_crew", 0) - curr_state.get("p2_crew", 0)
    if p2_crew_lost > 0:
        max_crew = curr_state.get("p2_max_crew", 1) or 1
        reward += 0.1 * (p2_crew_lost / max_crew)

    # Crew damage received (normalized)
    p1_crew_lost = prev_state.get("p1_crew", 0) - curr_state.get("p1_crew", 0)
    if p1_crew_lost > 0:
        max_crew = curr_state.get("p1_max_crew", 1) or 1
        reward -= 0.1 * (p1_crew_lost / max_crew)

    # Small time penalty to encourage aggression
    reward -= 0.001

    # Terminal rewards
    if curr_state.get("done", False):
        winner = curr_state.get("winner", -1)
        if winner == 0:
            reward += 1.0   # Player 1 won
        elif winner == 1:
            reward -= 1.0   # Player 1 lost

    return reward
