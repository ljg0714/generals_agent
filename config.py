"""Global configuration for Generals.io PPO training."""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
VIDEO_DIR = os.path.join(BASE_DIR, "videos")

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
PAD_TO = 24                 # observations are zero-padded to PAD_TO x PAD_TO
TRUNCATION = 500            # max game steps before truncation

# ---------------------------------------------------------------------------
# Map generation (GridFactory)
#   mode="uniform":    generals placed at Manhattan distance >= max(max_dims)//2
#   mode="generalsio": generals.io-style maps, needs maps ~18+ and uses a BFS
#                      min-generals-distance of 20
# ---------------------------------------------------------------------------
GRID_MODE = "uniform"
MIN_GRID_DIMS = (12, 12)
MAX_GRID_DIMS = (14, 14)
MOUNTAIN_DENSITY = 0.20
CITY_DENSITY = 0.02

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
INPUT_CHANNELS = 15         # generals-bots observation has 15 channels
HIDDEN_CHANNELS = (32, 32, 32, 16)
VALUE_HIDDEN = 64

# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
PPO_EPOCHS = 4
MINIBATCH_SIZE = 256
GRAD_CLIP = 0.5

# ---------------------------------------------------------------------------
# Training setup
# ---------------------------------------------------------------------------
NUM_ENVS = 64
NUM_STEPS = 128
NUM_ITERATIONS = 5000
CHECKPOINT_INTERVAL = 100
LOG_INTERVAL = 10
EVAL_INTERVAL = 500
SEED = 42

# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------
POOL_SIZE = 10                # number of past policy snapshots kept
SELFPLAY_PROB = 0.5           # chance an env faces a pool opponent instead of a bot
NUM_OPPONENT_GROUPS = 8       # distinct opponents sampled per rollout (for batching)
SAVE_TO_POOL_INTERVAL = 100   # iterations before snapshotting the current policy

# ---------------------------------------------------------------------------
# Reward shaping (potential-based, ported from the generals-bots paper)
# ---------------------------------------------------------------------------
MAX_ARMY_RATIO = 1.6
MAX_LAND_RATIO = 1.3
RATIO_WEIGHT = 0.3
CASTLE_WEIGHT = 0.4
