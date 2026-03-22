"""
Training configuration - all hyperparameters in one place.

Agent 2 - Round 1: Tuned for maximum throughput with lightweight CNN.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    ship_p1: int = 5
    ship_p2: int = 5
    p2_cyborg: bool = True
    frame_skip: int = 4
    num_envs: int = 8

    encoder_name: str = "ViT-B-16-SigLIP"
    encoder_pretrained: str = "webli"
    encoder_dim: int = 512
    hidden_dim: int = 256
    action_dim: int = 32

    learning_rate: float = 5e-4     # Higher LR for small trainable CNN
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.02          # Higher entropy for exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    total_timesteps: int = 1_000_000
    wall_clock_budget: float = 290.0
    eval_interval: int = 20_000
    eval_episodes: int = 10
    checkpoint_interval: int = 50_000

    target_win_rate: float = 0.8

    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
