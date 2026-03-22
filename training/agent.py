"""
Agent network: frozen SigLIP-B/16 vision encoder + trainable actor-critic MLP heads.

This file is MODIFIABLE by competing agents.
Agents can change the architecture, add frame stacking, use different encoders, etc.
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
        """
        Args:
            x: (B, H, W, 3) uint8 tensor or (B, 3, 224, 224) float tensor

        Returns:
            (B, output_dim) feature vector
        """
        with torch.no_grad():
            return self.visual(x)


class SimpleCNNEncoder(nn.Module):
    """
    Fallback CNN encoder (Impala-style) if SigLIP doesn't work on SM_120.
    Much smaller but learns features from scratch.
    """

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

        # Calculate flattened size for 84x84 input
        # After conv layers: (84-8)/4+1=20, (20-4)/2+1=9, (9-3)/1+1=7
        # 64 * 7 * 7 = 3136
        self.fc = nn.Linear(3136, self.output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 3, 84, 84) float tensor (normalized to [0,1])
        """
        features = self.network(x)
        return torch.relu(self.fc(features))


def preprocess_obs(obs, target_size=224, device="cpu"):
    """
    Preprocess game screenshot for the encoder.

    Args:
        obs: (B, 240, 320, 3) uint8 numpy array or tensor
        target_size: resize target (224 for SigLIP, 84 for CNN)
        device: torch device

    Returns:
        (B, 3, target_size, target_size) float32 tensor, normalized
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)

    # (B, H, W, C) -> (B, C, H, W)
    if obs.dim() == 3:
        obs = obs.unsqueeze(0)
    obs = obs.permute(0, 3, 1, 2).float() / 255.0

    # Resize
    obs = torch.nn.functional.interpolate(
        obs, size=(target_size, target_size), mode="bilinear", align_corners=False
    )

    # Normalize for SigLIP (approximate ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406], device=obs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=obs.device).view(1, 3, 1, 1)
    obs = (obs - mean) / std

    return obs.to(device)


class MeleeAgent(nn.Module):
    """
    Actor-critic agent for UQM Super Melee.

    Uses a frozen vision encoder for feature extraction,
    with trainable actor and critic MLP heads.
    """

    def __init__(self, encoder_type="siglip", hidden_dim=256, action_dim=32,
                 encoder_name="ViT-B-16-SigLIP", encoder_pretrained="webli"):
        super().__init__()

        self.encoder_type = encoder_type

        if encoder_type == "siglip" and HAS_OPEN_CLIP:
            self.encoder = SigLIPEncoder(encoder_name, encoder_pretrained)
            self.input_size = 224
        else:
            self.encoder = SimpleCNNEncoder()
            self.input_size = 84

        enc_dim = self.encoder.output_dim

        self.actor = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def get_features(self, obs):
        """Extract features from observation."""
        processed = preprocess_obs(obs, self.input_size, device=next(self.actor.parameters()).device)
        return self.encoder(processed)

    def get_value(self, obs):
        features = self.get_features(obs)
        return self.critic(features)

    def get_action_and_value(self, obs, action=None):
        """
        Get action, log_prob, entropy, and value from observation.

        Args:
            obs: (B, 240, 320, 3) uint8
            action: optional pre-selected action

        Returns:
            action, log_prob, entropy, value
        """
        features = self.get_features(obs)
        logits = self.actor(features)
        value = self.critic(features)

        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)
