"""
Gymnasium environment for UQM Super Melee.

Round 6 Agent 2: Subprocess-isolated environment wrapper.
The game runs in a child process so segfaults in libmelee.so
don't kill the training loop. Includes timeouts, auto-restart,
and episode statistics tracking.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import multiprocessing as mp
import logging
import time
import os
import traceback

logger = logging.getLogger(__name__)

# Timeout for individual step/reset calls (seconds)
STEP_TIMEOUT = 5.0
RESET_TIMEOUT = 10.0
# Max consecutive failures before giving up
MAX_CONSECUTIVE_FAILURES = 5


def _worker_loop(pipe, ship_p1, ship_p2, p2_cyborg, frame_skip, headless, seed):
    """
    Game worker process. Runs libmelee.so in isolation.
    Communicates via a multiprocessing Pipe.
    If this process segfaults, the parent can detect the broken pipe and restart.
    """
    # Import inside the subprocess to get a fresh library load
    from uqm_env import melee_ffi

    active = False

    try:
        while True:
            try:
                if not pipe.poll(30.0):  # 30s idle timeout
                    continue
                cmd = pipe.recv()
            except (EOFError, BrokenPipeError):
                break

            if cmd[0] == "reset":
                cmd_seed = cmd[1] if len(cmd) > 1 else seed
                try:
                    if active:
                        try:
                            melee_ffi.close()
                        except Exception:
                            pass
                        active = False

                    if melee_ffi.is_corrupted():
                        melee_ffi.restart_library()

                    melee_ffi.init(ship_p1, ship_p2,
                                   p2_cyborg=p2_cyborg,
                                   headless=headless,
                                   seed=cmd_seed)
                    active = True

                    # Step one frame to get initial observation
                    result = melee_ffi.step(0, 0)
                    pipe.send(("ok", result))
                except Exception as e:
                    pipe.send(("error", str(e)))

            elif cmd[0] == "step":
                action = cmd[1]
                try:
                    total_reward = 0.0
                    terminated = False
                    result = None

                    for i in range(frame_skip):
                        result = melee_ffi.step(action, 0)
                        total_reward += result["reward_p1"]

                        if result["done"]:
                            terminated = True
                            if result["winner"] == 0:
                                total_reward += 1.0
                            elif result["winner"] == 1:
                                total_reward -= 1.0
                            break

                    pipe.send(("ok", result, total_reward, terminated))
                except Exception as e:
                    pipe.send(("error", str(e)))

            elif cmd[0] == "close":
                try:
                    if active:
                        melee_ffi.close()
                        active = False
                except Exception:
                    pass
                pipe.send(("ok",))
                break

            else:
                pipe.send(("error", f"unknown command: {cmd[0]}"))

    except Exception as e:
        logger.error(f"Worker crashed: {e}\n{traceback.format_exc()}")
    finally:
        try:
            if active:
                melee_ffi.close()
        except Exception:
            pass
        try:
            pipe.close()
        except Exception:
            pass


class MeleeEnv(gym.Env):
    """
    UQM Super Melee environment with subprocess isolation.

    The game runs in a child process (multiprocessing spawn) so that
    segfaults in libmelee.so don't crash the training loop.
    Includes timeouts on all operations and automatic restart on failure.

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

        # 5 independent buttons = 32 action combos
        self.action_space = spaces.Discrete(32)

        # RGB screenshot
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(240, 320, 3), dtype=np.uint8
        )

        self._last_obs = None
        self._episode_reward = 0.0

        # Subprocess state
        self._process = None
        self._pipe = None
        self._consecutive_failures = 0

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

    def _start_worker(self):
        """Start (or restart) the game worker subprocess."""
        self._kill_worker()
        try:
            ctx = mp.get_context("spawn")
            parent_pipe, child_pipe = ctx.Pipe()
            self._pipe = parent_pipe
            self._process = ctx.Process(
                target=_worker_loop,
                args=(child_pipe, self.ship_p1, self.ship_p2,
                      self.p2_cyborg, self.frame_skip, self.headless, self._seed),
                daemon=True,
            )
            self._process.start()
            child_pipe.close()  # Parent doesn't need this end
            self._consecutive_failures = 0
            logger.debug(f"Worker started (pid={self._process.pid})")
        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            self._process = None
            self._pipe = None
            raise

    def _kill_worker(self):
        """Kill the worker process if it's running."""
        if self._process is not None:
            try:
                if self._process.is_alive():
                    self._process.kill()
                    self._process.join(timeout=2.0)
            except Exception:
                pass
            self._process = None
        if self._pipe is not None:
            try:
                self._pipe.close()
            except Exception:
                pass
            self._pipe = None

    def _send_cmd(self, cmd, timeout):
        """Send a command to the worker and wait for response with timeout."""
        if self._pipe is None or self._process is None or not self._process.is_alive():
            raise OSError("Worker process is not running")

        try:
            self._pipe.send(cmd)
        except (BrokenPipeError, OSError) as e:
            raise OSError(f"Worker pipe broken: {e}") from e

        if not self._pipe.poll(timeout):
            # Timeout - kill the worker
            logger.warning(f"Worker timed out on {cmd[0]} after {timeout}s, killing")
            self._kill_worker()
            raise TimeoutError(f"Worker timed out on {cmd[0]}")

        try:
            return self._pipe.recv()
        except (EOFError, BrokenPipeError, OSError) as e:
            raise OSError(f"Worker died during {cmd[0]}: {e}") from e

    def _ensure_worker(self):
        """Make sure a worker process is running."""
        if self._process is None or not self._process.is_alive():
            self._start_worker()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        env_seed = seed if seed is not None else self._seed

        for attempt in range(MAX_CONSECUTIVE_FAILURES):
            try:
                self._ensure_worker()
                response = self._send_cmd(("reset", env_seed + attempt), RESET_TIMEOUT)

                if response[0] == "ok":
                    result = response[1]
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
                else:
                    logger.warning(f"Reset attempt {attempt+1} failed: {response[1]}")
                    self._kill_worker()

            except (OSError, TimeoutError) as e:
                logger.warning(f"Reset attempt {attempt+1} exception: {e}")
                self._total_crashes += 1
                self._kill_worker()

        # All attempts failed - return a blank observation
        self._consecutive_failures += 1
        logger.error(f"All {MAX_CONSECUTIVE_FAILURES} reset attempts failed, returning blank obs")
        self._last_obs = np.zeros((240, 320, 3), dtype=np.uint8)
        return self._last_obs, {"frame_count": 0, "p1_crew": 0, "p2_crew": 0}

    def step(self, action):
        """
        Execute one agent step (frame_skip game frames).

        Returns:
            observation, reward, terminated, truncated, info
        """
        try:
            self._ensure_worker()
            response = self._send_cmd(("step", int(action)), STEP_TIMEOUT)

            if response[0] == "ok":
                result = response[1]
                total_reward = response[2]
                terminated = response[3]

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

                return self._last_obs, total_reward, terminated, False, info
            else:
                raise OSError(f"Step failed: {response[1]}")

        except (OSError, TimeoutError) as e:
            logger.warning(f"Step failed: {e}")
            self._consecutive_failures += 1
            self._total_crashes += 1
            self._kill_worker()

            # Return a "done" signal so the training loop resets this env
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
        if self._pipe is not None and self._process is not None and self._process.is_alive():
            try:
                self._send_cmd(("close",), 3.0)
            except Exception:
                pass
        self._kill_worker()

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
