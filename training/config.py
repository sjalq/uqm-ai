"""
Training configuration - all hyperparameters in one place.

Round 4 Agent 1: Integrated best practices from R3 losers.
- RunningMeanStd reward normalization, clipped value loss (R3A1)
- LayerNorm, deeper MLP heads, cosine LR annealing (R3A3)
- LR warmup for training stability (R3A1)
"""

from dataclasses import dataclass, field
from typing import List


# 12 useful combat actions (indices into the 32-action space)
# Each action is a 5-bit mask: [LEFT, RIGHT, THRUST, WEAPON, SPECIAL]
# Bit 0=LEFT(1), 1=RIGHT(2), 2=THRUST(4), 3=WEAPON(8), 4=SPECIAL(16)
COMBAT_ACTIONS: List[int] = [
    0,   # idle
    1,   # left
    2,   # right
    4,   # thrust
    8,   # fire
    12,  # thrust + fire (4+8)
    5,   # left + thrust (1+4)
    6,   # right + thrust (2+4)
    9,   # left + fire (1+8)
    10,  # right + fire (2+8)
    13,  # left + thrust + fire (1+4+8)
    14,  # right + thrust + fire (2+4+8)
]


@dataclass
class TrainingConfig:
    ship_p1: int = 5
    ship_p2: int = 5
    p2_cyborg: bool = True
    frame_skip: int = 4
    num_envs: int = 1               # R8A2: single env, maximize update frequency

    encoder_name: str = "ViT-B-16-SigLIP"
    encoder_pretrained: str = "webli"
    encoder_dim: int = 512
    hidden_dim: int = 256
    action_dim: int = 32

    # Frame stacking: number of consecutive grayscale frames as CNN input
    frame_stack: int = 4

    learning_rate: float = 1e-3      # R8A2: higher LR for fast learning
    num_steps: int = 8               # R8A2: tiny rollouts, maximum update frequency
    num_minibatches: int = 1         # R8A2: single minibatch (batch_size=8)
    update_epochs: int = 8           # R8A2: more epochs per update
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Entropy annealing: linearly anneal from ent_coef to ent_coef_final
    ent_coef_final: float = 0.005

    # Curriculum learning: phase 1 uses restricted combat actions, phase 2 uses all 32
    curriculum_phase1_steps: int = 80_000
    combat_actions: List[int] = field(default_factory=lambda: list(COMBAT_ACTIONS))

    total_timesteps: int = 1_000_000
    wall_clock_budget: float = 290.0
    eval_interval: int = 999999      # R8A2: disable eval during training
    eval_episodes: int = 5
    checkpoint_interval: int = 999999  # R8A2: disable checkpoints during training

    target_win_rate: float = 0.8

    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Reward shaping parameters
    reward_damage_dealt_scale: float = 3.0
    reward_damage_taken_scale: float = 1.0
    reward_combo_bonus: float = 0.05
    reward_survival_bonus: float = 0.0005

    # R3A2: Throughput optimizations
    use_torch_compile: bool = False   # R8A2: disable, overhead not worth it for tiny batches
    pin_memory: bool = True           # pin rollout buffers for faster GPU transfer
    gpu_preprocess: bool = True       # do preprocessing on GPU

    # R4A1: Integrated best practices
    use_layernorm: bool = True        # LayerNorm on encoder output (R3A3)
    deep_heads: bool = True           # 2-layer MLP heads 512->128->out (R3A3)
    use_cosine_lr: bool = True        # Cosine LR annealing (R3A3, better for short budgets)
    use_lr_warmup: bool = True        # LR warmup (R3A1)
    lr_warmup_frac: float = 0.05     # Fraction of updates for warmup
    use_reward_normalization: bool = False  # R8A2: disable, raw rewards for tiny batches
    use_clipped_vloss: bool = True    # Clipped value loss (R3A1)
