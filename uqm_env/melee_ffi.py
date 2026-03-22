"""
cffi wrapper for libmelee.so - the UQM Super Melee shared library.

Provides a thin Python API over the C bridge functions.
Pixel data is exposed as numpy arrays via zero-copy ffi.buffer().

Round 6 Agent 2: Added robustness - signal handling, library restart,
wrapped all cffi calls in try/except for crash recovery.
"""

import cffi
import numpy as np
import signal
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

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
_lib_corrupted = False
_original_sigsegv_handler = None


# Action bits - match BATTLE_INPUT_STATE in controls.h
ACTION_LEFT    = 1 << 0
ACTION_RIGHT   = 1 << 1
ACTION_THRUST  = 1 << 2
ACTION_WEAPON  = 1 << 3
ACTION_SPECIAL = 1 << 4

# Number of discrete actions (5 bits = 32 combinations)
NUM_ACTIONS = 32


def _sigsegv_handler(signum, frame):
    """Handle SIGSEGV from libmelee.so. Mark library as corrupted."""
    global _lib_corrupted
    _lib_corrupted = True
    logger.error(f"SIGSEGV caught in melee_ffi (pid={os.getpid()}). Library marked corrupted.")
    # Re-raise as a Python exception so the caller can catch it
    raise OSError("Segmentation fault in libmelee.so")


def _install_signal_handler():
    """Install SIGSEGV handler. Only safe in the main thread."""
    global _original_sigsegv_handler
    try:
        _original_sigsegv_handler = signal.signal(signal.SIGSEGV, _sigsegv_handler)
    except (ValueError, OSError):
        # Can't set signal handler outside main thread - that's fine,
        # subprocess isolation handles this case
        pass


def _restore_signal_handler():
    """Restore original SIGSEGV handler."""
    global _original_sigsegv_handler
    if _original_sigsegv_handler is not None:
        try:
            signal.signal(signal.SIGSEGV, _original_sigsegv_handler)
            _original_sigsegv_handler = None
        except (ValueError, OSError):
            pass


def _ensure_loaded():
    global lib, _lib_corrupted
    if _lib_corrupted:
        raise OSError("libmelee.so is in a corrupted state. Call restart_library() first.")
    if lib is None:
        if not _lib_path.exists():
            raise FileNotFoundError(
                f"libmelee.so not found at {_lib_path}. "
                "Run: python uqm_env/build_uqm.py"
            )
        try:
            lib = ffi.dlopen(str(_lib_path))
        except Exception as e:
            raise OSError(f"Failed to load libmelee.so: {e}") from e
        _install_signal_handler()
    return lib


def restart_library():
    """Force-reload the library after corruption or crash.
    This closes the old handle and opens a fresh one.
    Should be called from a fresh process for best results."""
    global lib, _lib_corrupted
    _lib_corrupted = False
    if lib is not None:
        try:
            # Try to close cleanly first
            lib.melee_close()
        except Exception:
            pass
        lib = None
    logger.info("Library state reset. Next call will reload libmelee.so.")


def is_corrupted() -> bool:
    """Check if the library is in a corrupted state."""
    return _lib_corrupted


def lib_init() -> int:
    """Initialize UQM library subsystems. Must be called once before init().
    Returns 0 on success. Idempotent."""
    try:
        l = _ensure_loaded()
        return l.melee_lib_init()
    except OSError:
        raise
    except Exception as e:
        logger.error(f"lib_init() failed: {e}")
        raise OSError(f"lib_init() failed: {e}") from e


def lib_shutdown():
    """Shut down UQM library subsystems."""
    try:
        l = _ensure_loaded()
        l.melee_lib_shutdown()
    except Exception as e:
        logger.warning(f"lib_shutdown() error (non-fatal): {e}")


def init(ship_p1: int, ship_p2: int, p2_cyborg: bool = True,
         headless: bool = True, seed: int = 0) -> int:
    """Initialize a melee match. Returns 0 on success.
    Automatically calls lib_init() if not already done."""
    try:
        l = _ensure_loaded()
        # Auto-init library if needed
        ret = l.melee_lib_init()
        if ret != 0:
            raise RuntimeError(f"melee_lib_init() failed with code {ret}")
        return l.melee_init(ship_p1, ship_p2, int(p2_cyborg), int(headless), seed)
    except OSError:
        raise
    except Exception as e:
        raise OSError(f"melee init failed: {e}") from e


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
    try:
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
    except OSError:
        raise
    except Exception as e:
        global _lib_corrupted
        _lib_corrupted = True
        raise OSError(f"melee_step failed (library marked corrupted): {e}") from e


def close():
    """Clean up and release resources."""
    try:
        l = _ensure_loaded()
        l.melee_close()
    except Exception as e:
        logger.warning(f"melee_close() error (non-fatal): {e}")


def get_ship_count() -> int:
    try:
        l = _ensure_loaded()
        return l.melee_get_ship_count()
    except Exception as e:
        logger.warning(f"get_ship_count() failed: {e}")
        return 0


def get_ship_name(index: int) -> str:
    try:
        l = _ensure_loaded()
        return ffi.string(l.melee_get_ship_name(index)).decode("utf-8")
    except Exception as e:
        logger.warning(f"get_ship_name({index}) failed: {e}")
        return f"ship_{index}"


def is_active() -> bool:
    try:
        l = _ensure_loaded()
        return bool(l.melee_is_active())
    except Exception:
        return False
