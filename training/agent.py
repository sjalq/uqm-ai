"""
Agent network: lightweight grayscale CNN encoder + trainable actor-critic MLP heads.

Agent 2 - Round 1: Architecture optimized for 5-minute training budget.
- Grayscale 84x84 input (1 channel) - 10x fewer pixels than SigLIP's 224x224x3
- Nature DQN CNN with orthogonal init for stable early learning
- All parameters trainable (no frozen encoder overhead)
- preprocess_obs handles both raw RGB and pre-processed inputs
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
    """Nature DQN-style CNN for grayscale 84x84 input. 1 input channel for max throughput."""

    def __init__(self, input_channels=1):
        super().__init__()
        self.output_dim = 512
        self.conv1 = _ortho_init(nn.Conv2d(input_channels, 32, 8, stride=4), gain=np.sqrt(2))
        self.conv2 = _ortho_init(nn.Conv2d(32, 64, 4, stride=2), gain=np.sqrt(2))
        self.conv3 = _ortho_init(nn.Conv2d(64, 64, 3, stride=1), gain=np.sqrt(2))
        self.fc = _ortho_init(nn.Linear(3136, self.output_dim), gain=np.sqrt(2))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))


def preprocess_obs(obs, target_size=84, device="cpu"):
    """
    Preprocess game screenshot. Handles raw RGB and pre-processed inputs.
    CNN path (target_size<=84): grayscale + resize -> (B, 1, 84, 84)
    SigLIP path (target_size>84): RGB + ImageNet norm -> (B, 3, size, size)
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
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
    """Actor-critic agent. Lightweight CNN + orthogonal MLP heads for fast 5-min training."""

    def __init__(self, encoder_type="siglip", hidden_dim=256, action_dim=32,
                 encoder_name="ViT-B-16-SigLIP", encoder_pretrained="webli"):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "siglip" and HAS_OPEN_CLIP:
            self.encoder = SigLIPEncoder(encoder_name, encoder_pretrained)
            self.input_size = 224
        else:
            self.encoder = SimpleCNNEncoder(input_channels=1)
            self.input_size = 84
        enc_dim = self.encoder.output_dim
        self.actor = nn.Sequential(
            _ortho_init(nn.Linear(enc_dim, hidden_dim), gain=np.sqrt(2)),
            nn.ReLU(),
            _ortho_init(nn.Linear(hidden_dim, action_dim), gain=0.01),
        )
        self.critic = nn.Sequential(
            _ortho_init(nn.Linear(enc_dim, hidden_dim), gain=np.sqrt(2)),
            nn.ReLU(),
            _ortho_init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def get_features(self, obs):
        processed = preprocess_obs(obs, self.input_size, device=next(self.actor.parameters()).device)
        return self.encoder(processed)

    def get_value(self, obs):
        return self.critic(self.get_features(obs))

    def get_action_and_value(self, obs, action=None):
        features = self.get_features(obs)
        logits = self.actor(features)
        value = self.critic(features)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)
