"""
Agent network: frozen SigLIP-B/16 vision encoder + trainable actor-critic MLP heads.

Agent 3 - Round 1: Added action masking for curriculum learning,
deeper MLP heads, and optimized preprocessing.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False


class SigLIPEncoder(nn.Module):
    """Frozen SigLIP-B/16 vision encoder."""

    def __init__(self, model_name="ViT-B-16-SigLIP", pretrained="webli"):
        super().__init__()
        if not HAS_OPEN_CLIP:
            raise ImportError("open_clip is required: pip install open-clip-torch")

        model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = model.visual
        self.output_dim = self.visual.output_dim

        # Freeze all parameters
        for param in self.visual.parameters():
            param.requires_grad = False
        self.visual.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.visual(x)


class SimpleCNNEncoder(nn.Module):
    """Fallback CNN encoder if SigLIP doesn't work on SM_120."""

    def __init__(self, input_channels=3):
        super().__init__()
        self.output_dim = 512

        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # For 84x84 input: (84-8)/4+1=20, (20-4)/2+1=9, (9-3)/1+1=7 => 64*7*7=3136
        self.fc = nn.Linear(3136, self.output_dim)

    def forward(self, x):
        features = self.network(x)
        return torch.relu(self.fc(features))


# Pre-allocated normalization tensors (created once, reused)
_norm_mean = None
_norm_std = None


def _get_norm_tensors(device):
    """Get or create normalization tensors on the right device."""
    global _norm_mean, _norm_std
    if _norm_mean is None or _norm_mean.device != device:
        _norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        _norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return _norm_mean, _norm_std


def preprocess_obs(obs, target_size=224, device="cpu"):
    """
    Preprocess game screenshot for the encoder.
    Optimized: minimal allocations, direct device placement.
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)

    if obs.dim() == 3:
        obs = obs.unsqueeze(0)

    # Move to device first, then convert - avoids CPU float allocation
    obs = obs.to(device, non_blocking=True)
    obs = obs.permute(0, 3, 1, 2).float().div_(255.0)

    obs = torch.nn.functional.interpolate(
        obs, size=(target_size, target_size), mode="bilinear", align_corners=False
    )

    mean, std = _get_norm_tensors(device)
    obs = (obs - mean) / std

    return obs


class MeleeAgent(nn.Module):
    """
    Actor-critic agent with action masking support for curriculum learning.

    The action mask allows restricting the action space during early training
    to a subset of useful combat actions, then expanding to the full space.
    """

    def __init__(self, encoder_type="siglip", hidden_dim=256, action_dim=32,
                 encoder_name="ViT-B-16-SigLIP", encoder_pretrained="webli"):
        super().__init__()

        self.encoder_type = encoder_type
        self.action_dim = action_dim

        if encoder_type == "siglip" and HAS_OPEN_CLIP:
            self.encoder = SigLIPEncoder(encoder_name, encoder_pretrained)
            self.input_size = 224
        else:
            self.encoder = SimpleCNNEncoder()
            self.input_size = 84

        enc_dim = self.encoder.output_dim

        # Deeper actor head with LayerNorm for training stability
        self.actor = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Deeper critic head with LayerNorm
        self.critic = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Action mask: None means all actions enabled
        self._action_mask = None
        self._mask_value = float('-inf')

    def set_action_mask(self, allowed_actions):
        """
        Set which actions are allowed. Pass None to allow all actions.

        Args:
            allowed_actions: list of allowed action indices, or None for all
        """
        if allowed_actions is None:
            self._action_mask = None
        else:
            mask = torch.full((self.action_dim,), self._mask_value)
            for a in allowed_actions:
                mask[a] = 0.0
            self._action_mask = mask

    def get_features(self, obs):
        """Extract features from observation."""
        processed = preprocess_obs(
            obs, self.input_size,
            device=next(self.actor.parameters()).device
        )
        return self.encoder(processed)

    def get_value(self, obs):
        features = self.get_features(obs)
        return self.critic(features)

    def _apply_mask(self, logits):
        """Apply action mask to logits if set."""
        if self._action_mask is not None:
            mask = self._action_mask.to(logits.device)
            logits = logits + mask
        return logits

    def get_action_and_value(self, obs, action=None):
        """
        Get action, log_prob, entropy, and value from observation.
        Respects action mask for curriculum learning.
        """
        features = self.get_features(obs)
        logits = self.actor(features)
        logits = self._apply_mask(logits)
        value = self.critic(features)

        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)
