"""
Agent network: lightweight grayscale CNN encoder + trainable actor-critic MLP heads.

Round 3 Agent 3: Architecture improvements for faster convergence.
- LayerNorm in CNN encoder output for stable feature distribution
- Deeper MLP heads (2 hidden layers each) with LayerNorm
- Separate feature streams prevent actor-critic gradient interference
- All prior features preserved: frame stacking, action masking, torch.compile, GPU preprocess
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False


def _ortho_init(layer, gain=1.0):
    """Orthogonal initialization for better early training stability."""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class SigLIPEncoder(nn.Module):
    """Frozen SigLIP-B/16 vision encoder."""

    def __init__(self, model_name="ViT-B-16-SigLIP", pretrained="webli"):
        super().__init__()
        if not HAS_OPEN_CLIP:
            raise ImportError("open_clip is required: pip install open-clip-torch")
        model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.visual = model.visual
        self.output_dim = self.visual.output_dim
        for param in self.visual.parameters():
            param.requires_grad = False
        self.visual.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.visual(x)


class SimpleCNNEncoder(nn.Module):
    """Nature DQN-style CNN with LayerNorm output. Accepts frame_stack channels."""

    def __init__(self, input_channels=4):
        super().__init__()
        self.output_dim = 512
        self.conv1 = _ortho_init(nn.Conv2d(input_channels, 32, 8, stride=4), gain=np.sqrt(2))
        self.conv2 = _ortho_init(nn.Conv2d(32, 64, 4, stride=2), gain=np.sqrt(2))
        self.conv3 = _ortho_init(nn.Conv2d(64, 64, 3, stride=1), gain=np.sqrt(2))
        self.fc = _ortho_init(nn.Linear(3136, self.output_dim), gain=np.sqrt(2))
        # R3A3: LayerNorm on encoder output stabilizes feature distribution for MLP heads
        self.ln = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = self.ln(F.relu(self.fc(x)))
        return x


def preprocess_obs_gpu(obs_tensor, target_size=84):
    """
    GPU-side preprocessing: grayscale conversion + resize.
    Input: (B, H, W, 3) uint8 tensor on GPU
    Output: (B, 1, target_size, target_size) float32 tensor on GPU
    """
    obs_float = obs_tensor.float()
    gray = (0.2989 * obs_float[:, :, :, 0] + 0.5870 * obs_float[:, :, :, 1] + 0.1140 * obs_float[:, :, :, 2])
    gray = gray.unsqueeze(1) / 255.0
    if gray.shape[2] != target_size or gray.shape[3] != target_size:
        gray = F.interpolate(gray, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return gray


def preprocess_obs(obs, target_size=84, device="cpu"):
    """
    Preprocess game screenshot. Handles raw RGB and pre-processed inputs.
    Returns single-channel grayscale (B, 1, 84, 84) for CNN path,
    or passes through already-stacked frames (B, N, 84, 84).
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
    # Already preprocessed (B, C, H, W) float32 with correct spatial dims
    if obs.dim() == 4 and obs.dtype == torch.float32 and obs.shape[2] == target_size and obs.shape[3] == target_size:
        return obs.to(device)
    if obs.dim() == 3:
        obs = obs.unsqueeze(0)
    if target_size <= 84:
        obs_float = obs.float()
        gray = (0.2989 * obs_float[:, :, :, 0] + 0.5870 * obs_float[:, :, :, 1] + 0.1140 * obs_float[:, :, :, 2])
        gray = gray.unsqueeze(1) / 255.0
        gray = F.interpolate(gray, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return gray.to(device)
    else:
        obs = obs.permute(0, 3, 1, 2).float() / 255.0
        obs = F.interpolate(obs, size=(target_size, target_size), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=obs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=obs.device).view(1, 3, 1, 1)
        return ((obs - mean) / std).to(device)


class MeleeAgent(nn.Module):
    """
    Actor-critic agent with deeper MLP heads and LayerNorm for fast convergence.

    R3 Agent 3 architecture improvements:
    - LayerNorm on CNN encoder output for stable feature distribution
    - 2-layer actor and critic heads with LayerNorm in first hidden layer
    - Wider mid-layer (hidden_dim -> hidden_dim//2) for richer representations
    - Frame stacking + action masking + torch.compile preserved
    """

    def __init__(self, encoder_type="siglip", hidden_dim=256, action_dim=32,
                 encoder_name="ViT-B-16-SigLIP", encoder_pretrained="webli",
                 frame_stack=4):
        super().__init__()
        self.encoder_type = encoder_type
        self.action_dim = action_dim
        self.frame_stack = frame_stack

        if encoder_type == "siglip" and HAS_OPEN_CLIP:
            self.encoder = SigLIPEncoder(encoder_name, encoder_pretrained)
            self.input_size = 224
        else:
            self.encoder = SimpleCNNEncoder(input_channels=frame_stack)
            self.input_size = 84

        enc_dim = self.encoder.output_dim
        mid_dim = hidden_dim // 2

        # R3A3: Deeper actor head with LayerNorm for training stability
        self.actor = nn.Sequential(
            _ortho_init(nn.Linear(enc_dim, hidden_dim), gain=np.sqrt(2)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            _ortho_init(nn.Linear(hidden_dim, mid_dim), gain=np.sqrt(2)),
            nn.ReLU(),
            _ortho_init(nn.Linear(mid_dim, action_dim), gain=0.01),
        )

        # R3A3: Deeper critic head with LayerNorm
        self.critic = nn.Sequential(
            _ortho_init(nn.Linear(enc_dim, hidden_dim), gain=np.sqrt(2)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            _ortho_init(nn.Linear(hidden_dim, mid_dim), gain=np.sqrt(2)),
            nn.ReLU(),
            _ortho_init(nn.Linear(mid_dim, 1), gain=1.0),
        )

        # Internal frame buffer for eval/inference (not used during training minibatch)
        self._frame_buffer = None
        # Action mask: None means all actions allowed
        self._action_mask = None
        # Flag for whether torch.compile has been applied
        self._compiled = False

    def try_compile(self):
        """Apply torch.compile to encoder for kernel fusion speedup. Safe no-op on failure."""
        if self._compiled:
            return
        try:
            self.encoder = torch.compile(self.encoder, mode="reduce-overhead")
            self._compiled = True
        except Exception:
            pass

    def reset_frame_buffer(self):
        """Clear frame buffer. Call on episode reset."""
        self._frame_buffer = None

    def _update_frame_buffer(self, single_frame):
        """
        Update internal frame buffer with a new single-channel frame.
        single_frame: (B, 1, H, W) preprocessed grayscale
        Returns: (B, frame_stack, H, W) stacked frames
        """
        if self._frame_buffer is None or self._frame_buffer.shape[0] != single_frame.shape[0]:
            self._frame_buffer = single_frame.repeat(1, self.frame_stack, 1, 1)
        else:
            self._frame_buffer = torch.cat([
                self._frame_buffer[:, 1:, :, :],
                single_frame
            ], dim=1)
        return self._frame_buffer

    def set_action_mask(self, allowed_actions):
        """Set which actions are allowed. Pass None to allow all actions."""
        if allowed_actions is None or len(allowed_actions) >= self.action_dim:
            self._action_mask = None
        else:
            mask = torch.full((self.action_dim,), float("-inf"))
            for a in allowed_actions:
                mask[a] = 0.0
            self._action_mask = mask

    def _apply_mask(self, logits):
        """Apply action mask to logits. No-op if mask is None."""
        if self._action_mask is None:
            return logits
        return logits + self._action_mask.to(logits.device)

    def get_features(self, obs):
        """
        Extract features. Handles two input modes:
        1. Raw/single-frame obs -> preprocess, update frame buffer, stack
        2. Pre-stacked obs (B, frame_stack, 84, 84) -> pass directly to encoder
        """
        device = next(self.actor.parameters()).device
        processed = preprocess_obs(obs, self.input_size, device=device)

        # If single-channel (eval path or sequential inference), use frame buffer
        if processed.shape[1] == 1 and self.encoder_type != "siglip":
            processed = self._update_frame_buffer(processed)

        return self.encoder(processed)

    def get_value(self, obs):
        return self.critic(self.get_features(obs))

    def get_action_and_value(self, obs, action=None):
        features = self.get_features(obs)
        logits = self._apply_mask(self.actor(features))
        value = self.critic(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)
