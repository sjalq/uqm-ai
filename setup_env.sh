#!/usr/bin/env bash
# One-shot environment setup for UQM Melee AI
set -euo pipefail

echo "=== UQM Melee AI Setup ==="

# System dependencies (Arch Linux)
echo "Installing system dependencies..."
sudo pacman -S --needed --noconfirm \
    sdl2 sdl2_image sdl2_mixer sdl2_net sdl2_ttf \
    libvorbis libogg libpng zlib base-devel \
    tmux python

# Python venv
echo "Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# PyTorch - try nightly for SM_120 support
echo "Installing PyTorch (nightly for RTX 5060 Ti SM_120 support)..."
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 || {
    echo "Nightly failed, trying stable..."
    pip install torch torchvision
}

# ML dependencies
echo "Installing ML dependencies..."
pip install open-clip-torch gymnasium tensorboard cffi numpy

# Git submodule
echo "Initializing UQM submodule..."
git submodule update --init --recursive

echo ""
echo "=== Setup complete ==="
echo "Activate the venv with: source .venv/bin/activate"
echo ""
echo "Next step: build UQM as a shared library (TODO: python uqm_env/build_uqm.py)"
