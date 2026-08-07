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
TRUNCATION = 1000           # max game steps before truncation (was 500 — games
                            # on 12-14 maps rarely finished, causing ~90% draws)

# Army growth (patched via agents/game_patch.py to match real generals.io).
# Real generals.io: each owned cell produces +1 army every ~25 turns, and
# cities/the general produce +1 every turn.
ARMY_INCREMENT_RATE = 25    # per-cell +1 period (was 50 in generals-bots)
CITY_GROWTH_PERIOD = 1      # owned general/city cells produce +1 every N steps

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
# Curriculum learning (map-size stages)
#   Each stage is (min_grid, max_grid, advance_win_rate, max_stage_iterations).
#   - advance_win_rate: advance when the greedy eval win rate vs bots on the
#     current stage's maps is >= this value.
#   - max_stage_iterations: hard cap on iterations in the stage (0 = no cap);
#     advances even if the win-rate bar was not met. Prevents stalls.
#   The last stage is never advanced past. The policy network is sized for the
#   largest stage (PAD_TO = final grid + 2) so checkpoints stay compatible
#   across stages. Set CURRICULUM = None to disable (or use --no-curriculum).
CURRICULUM = [
    (8, 8,   0.7, 2000),  # learn basics: expansion, combat on tiny maps
    (10, 12, 0.6, 2000),  # transfer to medium maps
    (12, 14, 0.0, 0),     # final: standard size, no advance
]
CURRICULUM_EVAL_INTERVAL = 50
CURRICULUM_EVAL_GAMES = 20

# ---------------------------------------------------------------------------
# Reward shaping (potential-based, ported from the generals-bots paper)
# The potential combines RELATIVE (ratio) terms with ABSOLUTE (log) terms:
#   pot = RATIO_WEIGHT*(army_ratio + land_ratio)
#       + ABS_WEIGHT*(log1p(land) + log1p(army))   # dense signal per step
#       + CASTLE_WEIGHT*castles
# Ratio-only potentials are ~0 while both players grow together (log(1)=0),
# which starved learning in the 2000-iteration run. The absolute terms give
# every captured cell / gained army a positive signal.
# ---------------------------------------------------------------------------
MAX_ARMY_RATIO = 1.6
MAX_LAND_RATIO = 1.3
RATIO_WEIGHT = 0.2
ABS_WEIGHT = 0.1
CASTLE_WEIGHT = 0.4
