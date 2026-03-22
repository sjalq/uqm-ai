#!/bin/bash
#
# Build UQM-MegaMod as libmelee.so (shared library for AI training)
#
# This script bypasses UQM's interactive build system and compiles
# the game directly into a shared library, exposing the ai_bridge API.
#
# Usage: ./scripts/build_libmelee.sh [-j N] [--clean]
#
# Dependencies: gcc, pkg-config, SDL2, libpng, zlib, libvorbis
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UQM_DIR="$PROJECT_ROOT/uqm-megamod"
SRC_DIR="$UQM_DIR/src"
BUILD_DIR="$UQM_DIR/obj/libmelee"
OUTPUT="$PROJECT_ROOT/uqm_env/libmelee.so"

# Default parallel jobs
JOBS=$(nproc 2>/dev/null || echo 4)
CLEAN=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j*)
            JOBS="${1#-j}"
            if [[ -z "$JOBS" ]]; then
                shift
                JOBS="$1"
            fi
            ;;
        --clean)
            CLEAN=1
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [-j N] [--clean]" >&2
            exit 1
            ;;
    esac
    shift
done

if [[ "$CLEAN" -eq 1 ]]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    rm -f "$OUTPUT"
    echo "Clean complete."
    exit 0
fi

echo "=== Building libmelee.so ==="
echo "UQM source: $SRC_DIR"
echo "Build dir:  $BUILD_DIR"
echo "Output:     $OUTPUT"
echo "Jobs:       $JOBS"
echo ""

# ------------------------------------------------------------------
# Step 1: Check dependencies
# ------------------------------------------------------------------
echo "--- Checking dependencies ---"

check_pkg() {
    if ! pkg-config --exists "$1" 2>/dev/null; then
        echo "ERROR: Missing dependency: $1" >&2
        echo "Install with your package manager (e.g., sudo pacman -S $2)" >&2
        return 1
    fi
    echo "  Found: $1 $(pkg-config --modversion "$1" 2>/dev/null || echo '(version unknown)')"
}

check_pkg sdl2 sdl2
check_pkg libpng libpng
check_pkg zlib zlib
# vorbisfile is optional but useful
if pkg-config --exists vorbisfile 2>/dev/null; then
    check_pkg vorbisfile libvorbis
    HAS_VORBIS=1
else
    echo "  Warning: vorbisfile not found, building without Ogg Vorbis support"
    HAS_VORBIS=0
fi

echo ""

# ------------------------------------------------------------------
# Step 2: Generate config_unix.h
# ------------------------------------------------------------------
echo "--- Generating config_unix.h ---"

mkdir -p "$BUILD_DIR"

# Check for common symbols/types to populate config header
check_symbol() {
    echo "int main(void) { extern int $1(); return $1(); }" | \
        gcc -x c - -o /dev/null 2>/dev/null && echo "#define $2" || echo "/* #undef $2 */"
}

check_type() {
    echo "#include <stddef.h>
#include <wchar.h>
#include <wctype.h>
#include <stdbool.h>
int main(void) { $1 x; (void)x; return 0; }" | \
        gcc -x c - -o /dev/null 2>/dev/null && echo "#define $2" || echo "/* #undef $2 */"
}

# We'll use a content directory relative to the binary for now
CONTENTDIR="$UQM_DIR/content"

cat > "$BUILD_DIR/config_unix.h" << CONFIGEOF
/* Auto-generated config_unix.h for libmelee.so build */
#ifndef CONFIG_UNIX_H_
#define CONFIG_UNIX_H_

#define CONTENTDIR "$CONTENTDIR"
#define USERDIR "~/.uqm-megamod/"
#define CONFIGDIR USERDIR
#define MELEEDIR "\${UQM_CONFIG_DIR}/teams/"
#define SAVEDIR "\${UQM_CONFIG_DIR}/save/"
#define SCRSHOTDIR "\${UQM_CONFIG_DIR}/screenshots/"

/* Endianness - assume little-endian (x86/ARM) */
/* #undef WORDS_BIGENDIAN */

/* POSIX features */
$(check_symbol readdir_r HAVE_READDIR_R)
$(check_symbol setenv HAVE_SETENV)
/* #undef HAVE_STRUPR */
#define HAVE_STRCASECMP_UQM
/* #undef HAVE_STRICMP */
#define HAVE_GETOPT_LONG
$(check_symbol iswgraph HAVE_ISWGRAPH)
$(check_type wchar_t HAVE_WCHAR_T)
$(check_type wint_t HAVE_WINT_T)
$(check_type _Bool HAVE__BOOL)

#endif /* CONFIG_UNIX_H_ */
CONFIGEOF

echo "  Generated $BUILD_DIR/config_unix.h"
echo ""

# ------------------------------------------------------------------
# Step 3: Collect source files
# ------------------------------------------------------------------
echo "--- Collecting source files ---"

# Find all .c files, excluding platform-specific and unnecessary ones
collect_sources() {
    find "$SRC_DIR" -name '*.c' -type f | sort | while read -r f; do
        local rel="${f#$SRC_DIR/}"

        # Exclude main() - we're building a library
        [[ "$rel" == "uqm.c" ]] && continue

        # Exclude platform-specific code (Darwin/macOS, Symbian)
        [[ "$rel" == darwin/* ]] && continue
        [[ "$rel" == symbian/* ]] && continue

        # Exclude getopt (system has getopt_long)
        [[ "$rel" == getopt/* ]] && continue

        # Exclude regex (system has regex.h)
        [[ "$rel" == regex/* ]] && continue

        # Exclude pthread threads (we use SDL threads)
        [[ "$rel" == libs/threads/pthread/* ]] && continue

        # Exclude OpenAL sound driver (we use MixSDL)
        [[ "$rel" == libs/sound/openal/* ]] && continue

        # Exclude network code (not needed for AI training)
        [[ "$rel" == libs/network/* ]] && continue
        [[ "$rel" == uqm/supermelee/netplay/* ]] && continue

        # Exclude cdp (not needed)
        [[ "$rel" == libs/cdp/* ]] && continue

        # Exclude Windows-only files
        [[ "$rel" == libs/log/msgbox_win.c ]] && continue

        # Exclude broken/debug-only files
        [[ "$rel" == libs/uio/memdebug.c ]] && continue
        [[ "$rel" == libs/uio/hashtable.c ]] && continue
        [[ "$rel" == abxadec/* ]] && continue

        # Exclude Lua standalone interpreter/compiler (have their own main())
        [[ "$rel" == libs/lua/lua.c ]] && continue
        [[ "$rel" == libs/lua/luac.c ]] && continue

        echo "$f"
    done
}

SOURCES=$(collect_sources)
SOURCE_COUNT=$(echo "$SOURCES" | wc -l)
echo "  Found $SOURCE_COUNT source files"
echo ""

# ------------------------------------------------------------------
# Step 4: Compile
# ------------------------------------------------------------------
echo "--- Compiling ($JOBS parallel jobs) ---"

# Build compiler flags
SDL2_CFLAGS=$(pkg-config --cflags sdl2)
SDL2_LIBS=$(pkg-config --libs sdl2)
PNG_CFLAGS=$(pkg-config --cflags libpng)
PNG_LIBS=$(pkg-config --libs libpng)
ZLIB_CFLAGS=$(pkg-config --cflags zlib)
ZLIB_LIBS=$(pkg-config --libs zlib)

VORBIS_CFLAGS=""
VORBIS_LIBS=""
VORBIS_DEFINES=""
if [[ "$HAS_VORBIS" -eq 1 ]]; then
    VORBIS_CFLAGS=$(pkg-config --cflags vorbisfile 2>/dev/null || echo "")
    VORBIS_LIBS=$(pkg-config --libs vorbisfile 2>/dev/null || echo "")
else
    VORBIS_DEFINES="-DOVCODEC_NONE"
fi

# Common flags
CFLAGS_COMMON=(
    -fPIC
    -O2
    -DNDEBUG
    --std=gnu99
    -DGFXMODULE_SDL
    -DSDL_DIR=SDL2
    -DHAVE_JOYSTICK
    -DHAVE_ZIP=1
    -DUSE_INTERNAL_MIKMOD
    -DUSE_INTERNAL_LUA
    -DTHREADLIB_SDL
    -DUSE_PLATFORM_ACCEL
    -I"$SRC_DIR"
    -I"$BUILD_DIR"
    $SDL2_CFLAGS
    $PNG_CFLAGS
    $ZLIB_CFLAGS
    $VORBIS_CFLAGS
    $VORBIS_DEFINES
)

# Link flags
LDFLAGS_ALL=(
    -shared
    -fPIC
    $SDL2_LIBS
    $PNG_LIBS
    $ZLIB_LIBS
    $VORBIS_LIBS
    -lm
    -lpthread
)

# Compile each source file to .o
COMPILED=0
FAILED=0
SKIPPED=0
OBJ_FILES=()

compile_one() {
    local src="$1"
    local rel="${src#$SRC_DIR/}"
    local obj="$BUILD_DIR/${rel}.o"
    local obj_dir
    obj_dir="$(dirname "$obj")"

    mkdir -p "$obj_dir"

    # Skip if object is newer than source
    if [[ -f "$obj" && "$obj" -nt "$src" ]]; then
        echo "$obj"
        return 0
    fi

    if gcc "${CFLAGS_COMMON[@]}" -c "$src" -o "$obj" 2>"$obj.err"; then
        rm -f "$obj.err"
        echo "$obj"
        return 0
    else
        echo "COMPILE_ERROR: $rel" >&2
        cat "$obj.err" >&2
        rm -f "$obj" "$obj.err"
        return 1
    fi
}

export -f compile_one
export SRC_DIR BUILD_DIR
export CFLAGS_COMMON_STR="${CFLAGS_COMMON[*]}"

# Use a temporary file list and xargs for parallel compilation
OBJLIST="$BUILD_DIR/objects.list"
ERRLOG="$BUILD_DIR/errors.log"
: > "$ERRLOG"

# Write a compilation script that each parallel job executes
COMPILE_SCRIPT="$BUILD_DIR/compile_one.sh"
cat > "$COMPILE_SCRIPT" << 'COMPEOF'
#!/bin/bash
set -e
src="$1"
SRC_DIR="$2"
BUILD_DIR="$3"
shift 3
CFLAGS=("$@")

rel="${src#$SRC_DIR/}"
obj="$BUILD_DIR/${rel}.o"
obj_dir="$(dirname "$obj")"
mkdir -p "$obj_dir"

# Skip if object is newer than source
if [[ -f "$obj" && "$obj" -nt "$src" ]]; then
    echo "$obj"
    exit 0
fi

if gcc "${CFLAGS[@]}" -c "$src" -o "$obj" 2>/tmp/libmelee_err_$$.log; then
    rm -f "/tmp/libmelee_err_$$.log"
    echo "$obj"
    exit 0
else
    echo "FAIL: $rel" >&2
    cat "/tmp/libmelee_err_$$.log" >&2
    rm -f "$obj" "/tmp/libmelee_err_$$.log"
    exit 1
fi
COMPEOF
chmod +x "$COMPILE_SCRIPT"

# Run parallel compilation, collecting results
echo "$SOURCES" | \
    xargs -P "$JOBS" -I{} bash "$COMPILE_SCRIPT" {} "$SRC_DIR" "$BUILD_DIR" "${CFLAGS_COMMON[@]}" \
    > "$OBJLIST" 2>"$ERRLOG" || true

OBJ_COUNT=$(wc -l < "$OBJLIST")
ERR_COUNT=$(grep -c "^FAIL:" "$ERRLOG" 2>/dev/null || true)
ERR_COUNT="${ERR_COUNT:-0}"

echo "  Compiled: $OBJ_COUNT objects"
if [[ "$ERR_COUNT" -gt 0 ]]; then
    echo "  Failed:   $ERR_COUNT files"
    echo ""
    echo "--- Compilation errors ---"
    cat "$ERRLOG"
    echo ""
    echo "Continuing with successful objects (some features may be missing)..."
fi
echo ""

if [[ "$OBJ_COUNT" -eq 0 ]]; then
    echo "ERROR: No object files produced. Cannot link." >&2
    exit 1
fi

# ------------------------------------------------------------------
# Step 5: Link into shared library
# ------------------------------------------------------------------
echo "--- Linking libmelee.so ---"

# Read object list
mapfile -t OBJ_ARRAY < "$OBJLIST"

mkdir -p "$(dirname "$OUTPUT")"

if gcc "${LDFLAGS_ALL[@]}" "${OBJ_ARRAY[@]}" -o "$OUTPUT" 2>"$BUILD_DIR/link.err"; then
    rm -f "$BUILD_DIR/link.err"
    echo "  Success: $OUTPUT"
    echo "  Size: $(du -h "$OUTPUT" | cut -f1)"
else
    echo "ERROR: Linking failed:" >&2
    cat "$BUILD_DIR/link.err" >&2
    echo ""
    echo "Attempting to identify undefined symbols..." >&2
    grep "undefined reference" "$BUILD_DIR/link.err" 2>/dev/null | head -20 >&2 || true
    exit 1
fi

echo ""

# ------------------------------------------------------------------
# Step 6: Verify exported symbols
# ------------------------------------------------------------------
echo "--- Verifying exported symbols ---"

# Cache nm output to file
NM_FILE="$BUILD_DIR/nm_output.txt"
nm -D "$OUTPUT" > "$NM_FILE" 2>/dev/null || true

for sym in melee_init melee_step melee_close melee_get_ship_count melee_get_ship_name melee_is_active; do
    if grep -q " T ${sym}$" "$NM_FILE" 2>/dev/null; then
        echo "  OK: $sym"
    else
        echo "  MISSING: $sym (may cause runtime errors)"
    fi
done

echo ""
echo "=== Build complete ==="
echo "Library: $OUTPUT"
echo ""
echo "Test with: python -c \"import ctypes; lib = ctypes.CDLL('$OUTPUT'); print('Loaded OK, ships:', lib.melee_get_ship_count())\""
