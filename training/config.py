"""
Training configuration - all hyperparameters in one place.

This file is MODIFIABLE by competing agents.
"""

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    # Environment
    ship_p1: int = 5                # Earthling Cruiser
    ship_p2: int = 5                # Earthling Cruiser
    p2_cyborg: bool = True          # Opponent is built-in AI
    frame_skip: int = 4             # Game frames per agent step
    num_envs: int = 8               # Parallel environments

    # Model architecture
    encoder_name: str = "ViT-B-16-SigLIP"
    encoder_pretrained: str = "webli"
    encoder_dim: int = 768          # SigLIP-B/16 output dimension
    hidden_dim: int = 256           # MLP hidden layer size
    action_dim: int = 32            # 5-bit action space

    # PPO hyperparameters
    learning_rate: float = 3e-4
    num_steps: int = 128            # Steps per rollout per env
    num_minibatches: int = 4
    update_epochs: int = 4
    gamma: float = 0.99             # Discount factor
    gae_lambda: float = 0.95        # GAE lambda
    clip_coef: float = 0.2          # PPO clip coefficient
    ent_coef: float = 0.01          # Entropy bonus coefficient
    vf_coef: float = 0.5            # Value function loss coefficient
    max_grad_norm: float = 0.5      # Gradient clipping

    # Training duration
    total_timesteps: int = 1_000_000
    eval_interval: int = 10_000     # Evaluate every N timesteps
    eval_episodes: int = 20         # Episodes per evaluation
    checkpoint_interval: int = 50_000

    # Evaluation threshold
    target_win_rate: float = 0.8    # Win rate to consider "competent"

    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
