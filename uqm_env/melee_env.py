"""
Gymnasium environment for UQM Super Melee.

Wraps the cffi bridge to libmelee.so, providing a standard
Gymnasium interface for reinforcement learning.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from . import melee_ffi


class MeleeEnv(gym.Env):
    """
    UQM Super Melee environment.

    Observation: RGB screenshot (240, 320, 3) uint8
    Action: Discrete(32) - 5-bit mask of (left, right, thrust, weapon, special)

    Args:
        ship_p1: Ship index for player 1 (0-24)
        ship_p2: Ship index for player 2 (0-24)
        p2_cyborg: If True, player 2 is the built-in Cyborg AI
        frame_skip: Number of game frames per agent step (default 4)
        headless: If True, skip window presentation
        seed: RNG seed (0 = random)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 24}

    def __init__(self, ship_p1=5, ship_p2=5, p2_cyborg=True,
                 frame_skip=4, headless=True, seed=0,
                 render_mode="rgb_array"):
        super().__init__()

        self.ship_p1 = ship_p1
        self.ship_p2 = ship_p2
        self.p2_cyborg = p2_cyborg
        self.frame_skip = frame_skip
        self.headless = headless
        self._seed = seed
        self.render_mode = render_mode

        # 5 independent buttons = 32 action combos
        self.action_space = spaces.Discrete(32)

        # RGB screenshot
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(240, 320, 3), dtype=np.uint8
        )

        self._last_obs = None
        self._episode_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Close any existing match
        if melee_ffi.is_active():
            melee_ffi.close()

        # Use provided seed or generate one
        env_seed = seed if seed is not None else self._seed

        melee_ffi.init(
            self.ship_p1, self.ship_p2,
            p2_cyborg=self.p2_cyborg,
            headless=self.headless,
            seed=env_seed
        )

        # Step one frame to get initial observation
        result = melee_ffi.step(0, 0)
        self._last_obs = result["pixels"]
        self._episode_reward = 0.0

        return self._last_obs, {"frame_count": result["frame_count"]}

    def step(self, action):
        """
        Execute one agent step (frame_skip game frames).

        Args:
            action: int in [0, 31] representing 5-bit button mask

        Returns:
            observation, reward, terminated, truncated, info
        """
        total_reward = 0.0
        terminated = False
        info = {}

        for i in range(self.frame_skip):
            result = melee_ffi.step(action, 0)

            total_reward += result["reward_p1"]

            if result["done"]:
                terminated = True
                # Final reward based on outcome
                if result["winner"] == 0:
                    total_reward += 1.0  # Player 1 won
                elif result["winner"] == 1:
                    total_reward -= 1.0  # Player 1 lost
                break

        self._last_obs = result["pixels"]
        self._episode_reward += total_reward

        info = {
            "p1_crew": result["p1_crew"],
            "p2_crew": result["p2_crew"],
            "winner": result["winner"],
            "frame_count": result["frame_count"],
            "episode_reward": self._episode_reward,
        }

        return self._last_obs, total_reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._last_obs
        return None

    def close(self):
        if melee_ffi.is_active():
            melee_ffi.close()


# Register the environment
gym.register(
    id="UQMMelee-v0",
    entry_point="uqm_env.melee_env:MeleeEnv",
)
