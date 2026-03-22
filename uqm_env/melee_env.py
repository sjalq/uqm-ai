"""
Gymnasium environment for UQM Super Melee.

Round 9 Agent 1: In-process environment - no subprocess isolation.
Calls melee_ffi directly for 10-50x faster env steps by eliminating
pipe serialization, process startup, and SDL reinit overhead.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
import traceback

logger = logging.getLogger(__name__)

# Max consecutive failures before returning blank obs
MAX_CONSECUTIVE_FAILURES = 3


class MeleeEnv(gym.Env):
    """
    UQM Super Melee environment - direct in-process FFI calls.

    No subprocess isolation. Game runs in the same process for maximum
    throughput. Segfaults are caught by the SIGSEGV handler in melee_ffi.

    Observation: RGB screenshot (240, 320, 3) uint8
    Action: Discrete(32) - 5-bit mask of (left, right, thrust, weapon, special)
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

        self.action_space = spaces.Discrete(32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(240, 320, 3), dtype=np.uint8
        )

        self._last_obs = None
        self._episode_reward = 0.0
        self._active = False

        # Episode statistics
        self._episode_frames = 0
        self._episode_damage_dealt = 0.0
        self._episode_damage_taken = 0.0
        self._initial_p1_crew = 0
        self._initial_p2_crew = 0
        self._prev_p1_crew = 0
        self._prev_p2_crew = 0
        self._total_episodes = 0
        self._total_crashes = 0
        self._consecutive_failures = 0

        # Import ffi module once
        from uqm_env import melee_ffi
        self._ffi = melee_ffi

    def _close_game(self):
        """Close the current game session if active."""
        if self._active:
            try:
                self._ffi.close()
            except Exception:
                pass
            self._active = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        env_seed = seed if seed is not None else self._seed

        for attempt in range(MAX_CONSECUTIVE_FAILURES):
            try:
                self._close_game()

                # Restart library if corrupted
                if self._ffi.is_corrupted():
                    self._ffi.restart_library()

                self._ffi.init(
                    self.ship_p1, self.ship_p2,
                    p2_cyborg=self.p2_cyborg,
                    headless=self.headless,
                    seed=env_seed + attempt,
                )
                self._active = True

                # Step one frame to get initial observation
                result = self._ffi.step(0, 0)

                self._last_obs = result["pixels"]
                self._episode_reward = 0.0
                self._episode_frames = 0
                self._initial_p1_crew = result.get("p1_crew", 0)
                self._initial_p2_crew = result.get("p2_crew", 0)
                self._prev_p1_crew = self._initial_p1_crew
                self._prev_p2_crew = self._initial_p2_crew
                self._episode_damage_dealt = 0.0
                self._episode_damage_taken = 0.0
                self._consecutive_failures = 0
                self._total_episodes += 1

                return self._last_obs, {
                    "frame_count": result.get("frame_count", 0),
                    "p1_crew": self._initial_p1_crew,
                    "p2_crew": self._initial_p2_crew,
                }

            except Exception as e:
                logger.warning(f"Reset attempt {attempt+1} failed: {e}")
                self._total_crashes += 1
                self._active = False

        # All attempts failed
        self._consecutive_failures += 1
        logger.error(f"All {MAX_CONSECUTIVE_FAILURES} reset attempts failed, returning blank obs")
        self._last_obs = np.zeros((240, 320, 3), dtype=np.uint8)
        return self._last_obs, {"frame_count": 0, "p1_crew": 0, "p2_crew": 0}

    def step(self, action):
        """Execute one agent step (frame_skip game frames)."""
        try:
            total_reward = 0.0
            terminated = False
            result = None

            for i in range(self.frame_skip):
                result = self._ffi.step(int(action), 0)
                total_reward += result["reward_p1"]

                if result["done"]:
                    terminated = True
                    if result["winner"] == 0:
                        total_reward += 1.0
                    elif result["winner"] == 1:
                        total_reward -= 1.0
                    break

            self._last_obs = result["pixels"]
            self._episode_reward += total_reward
            self._episode_frames += 1

            # Track damage
            p1_crew = result.get("p1_crew", self._prev_p1_crew)
            p2_crew = result.get("p2_crew", self._prev_p2_crew)
            dmg_dealt = max(0, self._prev_p2_crew - p2_crew)
            dmg_taken = max(0, self._prev_p1_crew - p1_crew)
            self._episode_damage_dealt += dmg_dealt
            self._episode_damage_taken += dmg_taken
            self._prev_p1_crew = p1_crew
            self._prev_p2_crew = p2_crew

            self._consecutive_failures = 0

            info = {
                "p1_crew": p1_crew,
                "p2_crew": p2_crew,
                "winner": result.get("winner", -1),
                "frame_count": result.get("frame_count", 0),
                "episode_reward": self._episode_reward,
                "episode_frames": self._episode_frames,
                "episode_damage_dealt": self._episode_damage_dealt,
                "episode_damage_taken": self._episode_damage_taken,
            }

            if terminated:
                self._active = False

            return self._last_obs, total_reward, terminated, False, info

        except Exception as e:
            logger.warning(f"Step failed: {e}")
            self._consecutive_failures += 1
            self._total_crashes += 1
            self._active = False

            if self._last_obs is None:
                self._last_obs = np.zeros((240, 320, 3), dtype=np.uint8)
            return self._last_obs, 0.0, True, False, {
                "p1_crew": 0,
                "p2_crew": 0,
                "winner": -1,
                "frame_count": 0,
                "episode_reward": self._episode_reward,
                "crash": True,
            }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._last_obs
        return None

    def close(self):
        self._close_game()

    def get_stats(self):
        """Return cumulative environment statistics."""
        return {
            "total_episodes": self._total_episodes,
            "total_crashes": self._total_crashes,
            "consecutive_failures": self._consecutive_failures,
        }


# Register the environment
gym.register(
    id="UQMMelee-v0",
    entry_point="uqm_env.melee_env:MeleeEnv",
)
