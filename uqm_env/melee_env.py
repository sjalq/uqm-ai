"""
Gymnasium environment for UQM Super Melee.

Round 9 Agent 2: Shared memory optimization for subprocess communication.
The 240x320x3 pixel observation (~230KB) is written to shared memory by the
worker process. Only small control dicts (crew, energy, done, winner) are
sent over the pipe. This eliminates pickle serialization of the observation,
which was the main throughput bottleneck (~5 SPS).

The worker subprocess is persistent across episodes (no respawning).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import logging
import time
import os
import traceback

logger = logging.getLogger(__name__)

# Timeout for individual step/reset calls (seconds)
STEP_TIMEOUT = 10.0
RESET_TIMEOUT = 60.0  # First reset loads UQM content (~20-30s)
# Max consecutive failures before giving up
MAX_CONSECUTIVE_FAILURES = 5

# Observation dimensions
OBS_H, OBS_W, OBS_C = 240, 320, 3
OBS_BYTES = OBS_H * OBS_W * OBS_C  # 230400 bytes


def _worker_loop(pipe, shm_name, ship_p1, ship_p2, p2_cyborg, frame_skip, headless, seed):
    """
    Game worker process. Runs libmelee.so in isolation.
    Writes pixel observations directly to shared memory.
    Sends only small control dicts over the pipe.
    """
    from uqm_env import melee_ffi

    active = False
    # Attach to the shared memory block created by the parent
    shm = None
    shm_array = None
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        shm_array = np.ndarray((OBS_H, OBS_W, OBS_C), dtype=np.uint8, buffer=shm.buf)
    except Exception as e:
        try:
            pipe.send(("error", f"Failed to attach shared memory: {e}"))
        except Exception:
            pass
        return

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
                    # Write pixels to shared memory (zero-copy into buffer)
                    pixels = result["pixels"]
                    if pixels.shape == (OBS_H, OBS_W, OBS_C):
                        shm_array[:] = pixels
                    else:
                        shm_array[:] = 0
                    # Send only metadata over pipe (no pixels)
                    pipe.send(("ok", {
                        "p1_crew": result["p1_crew"],
                        "p2_crew": result["p2_crew"],
                        "p1_max_crew": result.get("p1_max_crew", 0),
                        "p2_max_crew": result.get("p2_max_crew", 0),
                        "p1_energy": result.get("p1_energy", 0),
                        "p2_energy": result.get("p2_energy", 0),
                        "frame_count": result["frame_count"],
                    }))
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

                    # Write pixels to shared memory
                    pixels = result["pixels"]
                    if pixels.shape == (OBS_H, OBS_W, OBS_C):
                        shm_array[:] = pixels
                    else:
                        shm_array[:] = 0

                    # Send only metadata + reward over pipe
                    pipe.send(("ok", {
                        "p1_crew": result["p1_crew"],
                        "p2_crew": result["p2_crew"],
                        "p1_max_crew": result.get("p1_max_crew", 0),
                        "p2_max_crew": result.get("p2_max_crew", 0),
                        "p1_energy": result.get("p1_energy", 0),
                        "p2_energy": result.get("p2_energy", 0),
                        "winner": result.get("winner", -1),
                        "frame_count": result["frame_count"],
                        "done": result["done"],
                    }, total_reward, terminated))
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
            if shm is not None:
                shm.close()  # Close but don't unlink (parent owns it)
        except Exception:
            pass
        try:
            pipe.close()
        except Exception:
            pass


class MeleeEnv(gym.Env):
    """
    UQM Super Melee environment with subprocess isolation and shared memory.

    The game runs in a child process (multiprocessing spawn) so that
    segfaults in libmelee.so don't crash the training loop.
    Pixel observations are transferred via shared memory (zero-copy).
    Only small control messages go through the pipe.

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
            low=0, high=255, shape=(OBS_H, OBS_W, OBS_C), dtype=np.uint8
        )

        self._last_obs = None
        self._episode_reward = 0.0

        # Subprocess state
        self._process = None
        self._pipe = None
        self._shm = None
        self._shm_array = None
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

    def _create_shared_memory(self):
        """Create a shared memory block for pixel observations."""
        if self._shm is not None:
            return  # Already created
        try:
            self._shm = shared_memory.SharedMemory(
                create=True, size=OBS_BYTES
            )
            self._shm_array = np.ndarray(
                (OBS_H, OBS_W, OBS_C), dtype=np.uint8, buffer=self._shm.buf
            )
            self._shm_array[:] = 0
            logger.debug(f"Shared memory created: {self._shm.name} ({OBS_BYTES} bytes)")
        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            self._shm = None
            self._shm_array = None
            raise

    def _destroy_shared_memory(self):
        """Clean up shared memory."""
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None
            self._shm_array = None

    def _start_worker(self):
        """Start (or restart) the game worker subprocess."""
        self._kill_worker()
        try:
            # Ensure shared memory exists
            self._create_shared_memory()

            ctx = mp.get_context("spawn")
            parent_pipe, child_pipe = ctx.Pipe()
            self._pipe = parent_pipe
            self._process = ctx.Process(
                target=_worker_loop,
                args=(child_pipe, self._shm.name,
                      self.ship_p1, self.ship_p2,
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

    def _read_obs_from_shm(self):
        """Read observation from shared memory (copy to avoid races)."""
        if self._shm_array is not None:
            return self._shm_array.copy()
        return np.zeros((OBS_H, OBS_W, OBS_C), dtype=np.uint8)

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
                    # Read pixels from shared memory instead of pipe
                    self._last_obs = self._read_obs_from_shm()
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
        self._last_obs = np.zeros((OBS_H, OBS_W, OBS_C), dtype=np.uint8)
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

                # Read pixels from shared memory instead of pipe
                self._last_obs = self._read_obs_from_shm()
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
                self._last_obs = np.zeros((OBS_H, OBS_W, OBS_C), dtype=np.uint8)
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
        self._destroy_shared_memory()

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
