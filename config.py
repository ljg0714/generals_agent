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

# Army growth (patched via agents/game_patch.py) — FAITHFUL to real generals.io.
# In this env 2 steps = 1 real turn, so per-cell +1 every 50 steps = every 25
# turns (matches real floor(land/25)) and cities/the general produce +1 every
# 2 steps = every turn. Games take ~150-500 steps to resolve, which is
# realistic. For faster training you could set 25/1, but a policy trained on
# the faster rate does NOT transfer faithfully to the real game.
ARMY_INCREMENT_RATE = 50    # per-cell +1 period (= every 25 real turns)
CITY_GROWTH_PERIOD = 2      # owned general/city cells +1 every N steps (= every real turn)

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
# Memory-augmented observation (paper arXiv:2507.06825): 23 base/memory channels
# + 2 * MEMORY_HISTORY_SIZE per-side army-delta move-history stacks.
INPUT_CHANNELS = 31         # 23 + 2 * MEMORY_HISTORY_SIZE
MEMORY_HISTORY_SIZE = 4     # per-side history length in the augmented observation
HIDDEN_CHANNELS = (32, 32, 32, 16)
VALUE_HIDDEN = 64

# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE = 3e-4
# 0.995 (was 0.99): games on big maps run 600+ steps, and 0.99^600 ≈ 0.002
# diluted the terminal win/loss to nothing before the early "attack or expand"
# decisions. 0.995^600 ≈ 0.05 keeps a usable signal at that horizon.
GAMMA = 0.995
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
NUM_OPPONENT_GROUPS = 8       # distinct opponents sampled per rollout (for batching)
SAVE_TO_POOL_INTERVAL = 100   # iterations before snapshotting the current policy
# Self-play ratio ramps from START to END over RAMP_ITERS iterations. Early
# training keeps a high bot fraction (learn the basics); once the policy beats
# the built-in bots, ramp toward self-play so opponents grow with it. The
# residual bot fraction anchors against strategy drift.
SELFPLAY_PROB_START = 0.5
SELFPLAY_PROB_END = 0.75
SELFPLAY_RAMP_ITERS = 2000

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
    (8, 8,   0.8, 2000),  # learn basics: expansion, combat on tiny maps
    (10, 12, 0.7, 2000),  # transfer to medium maps
    (12, 14, 0.0, 0),     # final: standard size, no advance
]
CURRICULUM_EVAL_INTERVAL = 50
CURRICULUM_EVAL_GAMES = 20

# ---------------------------------------------------------------------------
# Reward shaping (potential-based, ported from the generals-bots paper)
# The potential is a RELATIVE-ADVANTAGE proxy:
#   pot = RATIO_WEIGHT*(army_ratio + land_ratio)
#       + ABS_WEIGHT*(log1p(my_land) - log1p(opp_land)
#                     + log1p(my_army) - log1p(opp_army))
#       + CASTLE_WEIGHT*(castles_mine - castles_opp)
# Pure log1p(my_*) saturates late (slope 1/x) and biases AGAINST attacking
# (army spent in combat lowers it) while the terminal +-1 is diluted to ~0.02
# by gamma over a long game. Subtracting the opponent's counts keeps a dense
# gradient at every stage: a successful strike — my land/army up OR the
# opponent's down — raises the potential, and net castles value city trades.
# ---------------------------------------------------------------------------
MAX_ARMY_RATIO = 1.6
MAX_LAND_RATIO = 1.3
RATIO_WEIGHT = 0.2
ABS_WEIGHT = 0.1
# Opponent-decline terms are weighted 2x the own-growth terms, so REDUCING the
# opponent (attacks, taking their land) pays more than endlessly expanding into
# neutral ground. On big maps the own-growth terms reward expansion forever,
# which stalls the game; the higher opp weight tilts the policy toward striking.
ABS_OPP_WEIGHT = 0.2
CASTLE_WEIGHT = 0.5

# Time-dependent terms — encourage attacking efficiency / shorter games.
# DRAW_PENALTY: reward for a truncation draw. Smaller than a loss (-1) so a
#   draw is bad but not as bad as losing.
DRAW_PENALTY = 0.7
# STEP_PENALTY: subtracted per step once an episode exceeds STEP_PENALTY_START
#   steps, and it GROWS every STEP_PENALTY_INTERVAL further steps by
#   STEP_PENALTY_INCREMENT — so a game that drags on gets increasingly
#   expensive, forcing the policy to finish instead of turtling. Short games
#   are unpenalized (no premature rushing).
STEP_PENALTY = 0.002
STEP_PENALTY_START = 100
STEP_PENALTY_INCREMENT = 0.002   # added per STEP_PENALTY_INTERVAL steps past the start
STEP_PENALTY_INTERVAL = 200
