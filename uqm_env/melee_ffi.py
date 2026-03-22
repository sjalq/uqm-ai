"""
cffi wrapper for libmelee.so - the UQM Super Melee shared library.

Provides a thin Python API over the C bridge functions.
Pixel data is exposed as numpy arrays via zero-copy ffi.buffer().
"""

import cffi
import numpy as np
from pathlib import Path

ffi = cffi.FFI()

# Load the cdef from the header file
_api_header = Path(__file__).parent / "libmelee_api.h"
ffi.cdef(_api_header.read_text())

# Load the shared library
_lib_path = Path(__file__).parent / "libmelee.so"
if not _lib_path.exists():
    # Try build directory
    _lib_path = Path(__file__).parent.parent / "uqm-megamod" / "libmelee.so"

lib = None

def _ensure_loaded():
    global lib
    if lib is None:
        if not _lib_path.exists():
            raise FileNotFoundError(
                f"libmelee.so not found at {_lib_path}. "
                "Run: python uqm_env/build_uqm.py"
            )
        lib = ffi.dlopen(str(_lib_path))
    return lib


# Action bits - match BATTLE_INPUT_STATE in controls.h
ACTION_LEFT    = 1 << 0
ACTION_RIGHT   = 1 << 1
ACTION_THRUST  = 1 << 2
ACTION_WEAPON  = 1 << 3
ACTION_SPECIAL = 1 << 4

# Number of discrete actions (5 bits = 32 combinations)
NUM_ACTIONS = 32


def lib_init() -> int:
    """Initialize UQM library subsystems. Must be called once before init().
    Returns 0 on success. Idempotent."""
    l = _ensure_loaded()
    return l.melee_lib_init()


def lib_shutdown():
    """Shut down UQM library subsystems."""
    l = _ensure_loaded()
    l.melee_lib_shutdown()


def init(ship_p1: int, ship_p2: int, p2_cyborg: bool = True,
         headless: bool = True, seed: int = 0) -> int:
    """Initialize a melee match. Returns 0 on success.
    Automatically calls lib_init() if not already done."""
    l = _ensure_loaded()
    # Auto-init library if needed
    ret = l.melee_lib_init()
    if ret != 0:
        raise RuntimeError(f"melee_lib_init() failed with code {ret}")
    return l.melee_init(ship_p1, ship_p2, int(p2_cyborg), int(headless), seed)


def step(p1_action: int, p2_action: int = 0) -> dict:
    """
    Advance one game frame.

    Args:
        p1_action: 5-bit action mask for player 1
        p2_action: 5-bit action mask for player 2 (ignored if p2 is cyborg)

    Returns:
        dict with keys: pixels (H,W,3 uint8 ndarray), reward_p1, reward_p2,
        done, p1_crew, p2_crew, winner, frame_count
    """
    l = _ensure_loaded()
    result = l.melee_step(p1_action, p2_action)

    # Zero-copy pixel access
    if result.pixels != ffi.NULL and result.width > 0 and result.height > 0:
        buf = ffi.buffer(result.pixels, result.width * result.height * 3)
        pixels = np.frombuffer(buf, dtype=np.uint8).reshape(
            result.height, result.width, 3
        ).copy()  # copy because buffer may be overwritten next step
    else:
        pixels = np.zeros((result.height or 240, result.width or 320, 3),
                          dtype=np.uint8)

    return {
        "pixels": pixels,
        "reward_p1": result.reward_p1,
        "reward_p2": result.reward_p2,
        "done": bool(result.done),
        "p1_crew": result.p1_crew,
        "p2_crew": result.p2_crew,
        "p1_max_crew": result.p1_max_crew,
        "p2_max_crew": result.p2_max_crew,
        "p1_energy": result.p1_energy,
        "p2_energy": result.p2_energy,
        "winner": result.winner,
        "frame_count": result.frame_count,
    }


def close():
    """Clean up and release resources."""
    l = _ensure_loaded()
    l.melee_close()


def get_ship_count() -> int:
    l = _ensure_loaded()
    return l.melee_get_ship_count()


def get_ship_name(index: int) -> str:
    l = _ensure_loaded()
    return ffi.string(l.melee_get_ship_name(index)).decode("utf-8")


def is_active() -> bool:
    l = _ensure_loaded()
    return bool(l.melee_is_active())
