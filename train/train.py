"""PPO training for Generals.io with self-play (CleanRL-style).

Single-process vectorized environment: `NUM_ENVS` games are stepped in a Python
loop each rollout step. Opponents are either built-in bots or frozen snapshots
of the policy from the self-play pool. Potential-based reward shaping from the
generals-bots paper shapes non-terminal steps; terminal steps get +1/-1.

Usage:
    python train/train.py                          # defaults from config.py
    python train/train.py --num-envs 16 --num-steps 64 --iterations 200 \
        --grid-min 8 --grid-max 8 --no-selfplay   # quick smoke test
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from agents.generals_env import GeneralsEnv
from agents.network import PolicyValueNetwork, decode_actions
from agents.ppo_agent import PPOAgent
from agents.rewards import composite_reward
from generals.agents.expander_agent import ExpanderAgent
from generals.agents.random_agent import RandomAgent
from generals.core.action import compute_valid_move_mask
from generals.core.grid import GridFactory
from train.eval import evaluate
from train.self_play import OpponentPool

# human_exe (vendored EklipZ bot) is optional: default training doesn't need it.
try:
    from agents.human_exe_agent import HumanExeAgent
    _HAS_HUMAN_EXE = True
except Exception:
    HumanExeAgent = None
    _HAS_HUMAN_EXE = False


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def make_grid_factory(cfg) -> GridFactory:
    return GridFactory(
        mode=cfg.GRID_MODE,
        min_grid_dims=cfg.MIN_GRID_DIMS,
        max_grid_dims=cfg.MAX_GRID_DIMS,
        mountain_density=cfg.MOUNTAIN_DENSITY,
        city_density=cfg.CITY_DENSITY,
    )


def make_env(cfg, opponent=None) -> GeneralsEnv:
    return GeneralsEnv(
        opponent=opponent,
        grid_factory=make_grid_factory(cfg),
        pad_observations_to=cfg.PAD_TO,
        truncation=cfg.TRUNCATION,
    )


def rebuild_envs(cfg, num_envs, pool, bot_pool, seed, selfplay_prob):
    """Create fresh envs (used at start and on curriculum advance)."""
    envs = [make_env(cfg) for _ in range(num_envs)]
    opp_agents = [None] * num_envs
    rng = np.random.default_rng(seed)
    for i, env in enumerate(envs):
        env.reset(seed=seed * 10_000 + i)
        opp_agents[i] = sample_opponent(pool, bot_pool, selfplay_prob, rng)
        env.set_opponent(opp_agents[i])
    return envs, opp_agents


def curriculum_stage_dims(cfg, stage_idx):
    """Return (min_grid, max_grid) for a curriculum stage."""
    min_g, max_g, _, _ = cfg.CURRICULUM[stage_idx]
    return (min_g, min_g), (max_g, max_g)


# ---------------------------------------------------------------------------
# Opponent handling
# ---------------------------------------------------------------------------
def sample_opponent(pool, bot_pool, selfplay_prob, rng):
    if len(pool) > 0 and rng.random() < selfplay_prob:
        return pool.sample()
    chosen = random.choice(bot_pool)
    # HumanExeAgent is stateful per game -> a fresh instance per env.
    if _HAS_HUMAN_EXE and isinstance(chosen, HumanExeAgent):
        return HumanExeAgent(player_id=chosen.player_id)
    return chosen


def selfplay_prob_at(cfg, iteration):
    """Ramp the self-play fraction from START to END over RAMP_ITERS iters."""
    ramp = max(cfg.SELFPLAY_RAMP_ITERS, 1)
    frac = min(1.0, iteration / ramp)
    return cfg.SELFPLAY_PROB_START + (cfg.SELFPLAY_PROB_END - cfg.SELFPLAY_PROB_START) * frac


def build_groups(opp_agents):
    """Group env indices by the identity of their shared opponent object."""
    groups: dict[int, list[int]] = {}
    for i, agent in enumerate(opp_agents):
        groups.setdefault(id(agent), []).append(i)
    id_to_agent = {id(a): a for a in opp_agents}
    return [(id_to_agent[k], v) for k, v in groups.items()]


def compute_opponent_actions(envs, groups, pad_to):
    """Compute one action per env's opponent. (N, 5) int32 actions."""
    N = len(envs)
    opp_actions = np.zeros((N, 5), dtype=np.int32)
    for agent, indices in groups:
        if isinstance(agent, PPOAgent):
            obs_list, mask_list = [], []
            for i in indices:
                o = envs[i].opponent_observation()
                # NOTE: as_tensor(pad_to=...) pads `o` IN PLACE, so it must run
                # before compute_valid_move_mask(o) for the mask to be (P,P,4).
                obs_list.append(np.asarray(o.as_tensor(pad_to=pad_to), dtype=np.float32))
                mask_list.append(np.asarray(compute_valid_move_mask(o), dtype=bool))
            # env_keys=indices isolates each env's memory inside the shared opponent.
            acts = agent.act_batched(np.stack(obs_list), np.stack(mask_list), env_keys=list(indices))
            opp_actions[np.asarray(indices)] = acts
        elif _HAS_HUMAN_EXE and isinstance(agent, HumanExeAgent):
            for i in indices:
                opp_actions[i] = np.asarray(
                    agent.act_on_game(envs[i]._env.game, envs[i].opponent_id), dtype=np.int32)
        else:
            for i in indices:
                o = envs[i].opponent_observation()
                opp_actions[i] = np.asarray(agent.act(o), dtype=np.int32)
    return opp_actions


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def collect_rollout(envs, opp_agents, network, pool, bot_pool, device, cfg, selfplay_prob):
    """Collect `cfg.NUM_STEPS` steps across all envs. Returns buffers and stats."""
    N, P, T = len(envs), cfg.PAD_TO, cfg.NUM_STEPS
    rng = np.random.default_rng()

    obs = np.stack([e.last_obs for e in envs]).astype(np.float32)
    masks = np.stack([e.last_mask for e in envs]).astype(bool)

    buf_obs = np.zeros((T, N, config.INPUT_CHANNELS, P, P), dtype=np.float32)
    buf_masks = np.zeros((T, N, P, P, 4), dtype=bool)
    buf_actions = np.zeros((T, N), dtype=np.int64)
    buf_logprobs = np.zeros((T, N), dtype=np.float32)
    buf_values = np.zeros((T, N), dtype=np.float32)
    buf_rewards = np.zeros((T, N), dtype=np.float32)
    buf_dones = np.zeros((T, N), dtype=bool)

    stats = {"episodes": 0, "wins": 0, "losses": 0, "drawn": 0,
             "length_sum": 0.0, "reward_sum": 0.0, "steps": 0}
    steps_since_reset = [0] * N

    for t in range(T):
        buf_masks[t] = masks

        obs_t = torch.from_numpy(obs).to(device)
        mask_t = torch.from_numpy(masks).to(device)
        with torch.no_grad():
            # In-network memory augmentation: fold the raw obs into the 31-channel
            # input, updating the network's per-env memory (auto-resets on timestep 0).
            obs_aug = network.augment_observation(obs_t)
            logits, values = network(obs_aug, mask_t)
            dist = Categorical(logits=logits)
            action_idx = dist.sample()
            logprob = dist.log_prob(action_idx)
            action_idx = action_idx.cpu().numpy()
            logprob = logprob.cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()

        buf_obs[t] = obs_aug.cpu().numpy()
        buf_actions[t] = action_idx
        buf_logprobs[t] = logprob
        buf_values[t] = values_np

        actions_5 = decode_actions(action_idx, P)  # (N, 5)

        # Opponent actions (batched by shared opponent object)
        opp_actions = compute_opponent_actions(envs, build_groups(opp_agents), P)

        # Step every environment
        obs_next = np.empty_like(obs)
        dones_t = np.zeros(N, dtype=bool)
        term_reward = np.zeros(N, dtype=np.float32)
        for i, env in enumerate(envs):
            o, _, term, trunc, info = env.step(actions_5[i], opp_actions[i])
            obs_next[i] = o
            done = term or trunc
            dones_t[i] = done
            steps_since_reset[i] += 1
            if done:
                stats["episodes"] += 1
                stats["length_sum"] += steps_since_reset[i]
                if term:
                    won = bool(info["winner"])
                    term_reward[i] = 1.0 if won else -1.0
                    stats["wins" if won else "losses"] += 1
                else:
                    # Truncation draw: penalize, but less than a loss.
                    term_reward[i] = -cfg.DRAW_PENALTY
                    stats["drawn"] += 1

        shaped = composite_reward(obs, obs_next, dones_t)
        # Time cost only applies to dragged-out episodes (steps > threshold),
        # and ramps up every STEP_PENALTY_INTERVAL further steps so an endless
        # turtle becomes increasingly expensive instead of a flat small tax.
        steps = np.asarray(steps_since_reset)
        over = np.maximum(steps - cfg.STEP_PENALTY_START, 0)
        step_penalty = np.where(
            steps > cfg.STEP_PENALTY_START,
            cfg.STEP_PENALTY + cfg.STEP_PENALTY_INCREMENT * (over // cfg.STEP_PENALTY_INTERVAL),
            0.0,
        )
        buf_rewards[t] = shaped + term_reward - step_penalty
        buf_dones[t] = dones_t
        stats["reward_sum"] += float(shaped.sum())
        stats["steps"] += N

        # Reset finished games with a fresh grid and a fresh opponent
        for i in np.flatnonzero(dones_t):
            opp_agents[i] = sample_opponent(pool, bot_pool, selfplay_prob, rng)
            envs[i].set_opponent(opp_agents[i])
            envs[i].reset(seed=None)
            obs_next[i] = envs[i].last_obs
            steps_since_reset[i] = 0

        obs = obs_next
        masks = np.stack([e.last_mask for e in envs]).astype(bool)

    # Bootstrap values for the final observation (augmented the same way as the loop)
    with torch.no_grad():
        obs_final = network.augment_observation(torch.from_numpy(obs).to(device))
        _, next_values = network(obs_final, torch.from_numpy(masks).to(device))
    next_values = next_values.squeeze(-1).cpu().numpy()

    return (buf_obs, buf_masks, buf_actions, buf_logprobs, buf_values,
            buf_rewards, buf_dones, next_values), stats


# ---------------------------------------------------------------------------
# GAE + PPO update
# ---------------------------------------------------------------------------
def compute_gae(rewards, values, dones, next_values, gamma, lam):
    T, N = rewards.shape
    values_ext = np.concatenate([values, next_values[None]], axis=0)
    advantages = np.zeros_like(rewards)
    last_adv = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_ext[t + 1] * nonterm - values[t]
        advantages[t] = delta + gamma * lam * nonterm * last_adv
        last_adv = advantages[t]
    return advantages


def ppo_update(network, optimizer, buffers, advantages, returns, device, cfg):
    obs, masks, actions, old_logprobs, _, _, _, _ = buffers
    T, N = obs.shape[:2]
    B = T * N

    obs_flat = obs.reshape(B, *obs.shape[2:])
    masks_flat = masks.reshape(B, *masks.shape[2:])
    actions_flat = actions.reshape(B)
    old_logprobs_flat = old_logprobs.reshape(B)

    obs_t = torch.from_numpy(obs_flat).to(device)
    masks_t = torch.from_numpy(masks_flat).to(device)
    actions_t = torch.from_numpy(actions_flat).to(device)
    old_logprobs_t = torch.from_numpy(old_logprobs_flat).to(device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages_t = torch.from_numpy(advantages.reshape(B)).to(device)
    returns_t = torch.from_numpy(returns.reshape(B)).to(device)

    total = {"pg": 0.0, "v": 0.0, "ent": 0.0}
    n_updates = 0
    for _ in range(cfg.PPO_EPOCHS):
        perm = torch.randperm(B, device=device)
        for start in range(0, B, cfg.MINIBATCH_SIZE):
            idx = perm[start:start + cfg.MINIBATCH_SIZE]
            logits, values = network(obs_t[idx], masks_t[idx])
            dist = Categorical(logits=logits)
            logprob = dist.log_prob(actions_t[idx])
            entropy = dist.entropy().mean()
            ratio = torch.exp(logprob - old_logprobs_t[idx])
            adv = advantages_t[idx]
            pg_loss = -torch.min(ratio * adv,
                                 torch.clamp(ratio, 1 - cfg.CLIP_EPS, 1 + cfg.CLIP_EPS) * adv).mean()
            v_loss = 0.5 * ((values.squeeze(-1) - returns_t[idx]) ** 2).mean()
            loss = pg_loss + cfg.VF_COEF * v_loss - cfg.ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

            total["pg"] += pg_loss.item()
            total["v"] += v_loss.item()
            total["ent"] += entropy.item()
            n_updates += 1

    return {k: v / max(n_updates, 1) for k, v in total.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--grid-min", type=int, default=None, help="min grid size (square)")
    p.add_argument("--grid-max", type=int, default=None, help="max grid size (square)")
    p.add_argument("--truncation", type=int, default=None)
    p.add_argument("--no-selfplay", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--save-to-pool-interval", type=int, default=None)
    p.add_argument("--eval-games", type=int, default=20)
    p.add_argument("--no-curriculum", action="store_true", help="disable map-size curriculum")
    p.add_argument("--use-human-exe-opponent", action="store_true",
                   help="add human_exe (vendored EklipZ bot) to the self-play opponent pool")
    p.add_argument("--curriculum-eval-games", type=int, default=None)
    p.add_argument("--init-from-pretrained", type=str, default=None,
                   help="warm-start from a behavior-cloned policy (.pt from train/pretrain.py)")
    p.add_argument("--ent-coef", type=float, default=None, help="override entropy coefficient")
    p.add_argument("--lr", type=float, default=None, help="override learning rate")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def apply_args(cfg, args):
    if args.num_envs: cfg.NUM_ENVS = args.num_envs
    if args.num_steps: cfg.NUM_STEPS = args.num_steps
    if args.iterations: cfg.NUM_ITERATIONS = args.iterations
    if args.grid_min or args.grid_max:
        gmin = args.grid_min if args.grid_min is not None else args.grid_max
        gmax = args.grid_max if args.grid_max is not None else args.grid_min
        cfg.MIN_GRID_DIMS = (gmin, gmin)
        cfg.MAX_GRID_DIMS = (gmax, gmax)
        # Pad observations to just above the largest grid instead of the full 24
        # (cheaper visibility filters, smaller network, faster stepping).
        cfg.PAD_TO = max(cfg.PAD_TO, gmax + 2)
    if args.truncation: cfg.TRUNCATION = args.truncation
    if args.no_selfplay:
        cfg.SELFPLAY_PROB_START = 0.0
        cfg.SELFPLAY_PROB_END = 0.0
    if args.checkpoint_interval: cfg.CHECKPOINT_INTERVAL = args.checkpoint_interval
    if args.eval_interval: cfg.EVAL_INTERVAL = args.eval_interval
    if args.save_to_pool_interval: cfg.SAVE_TO_POOL_INTERVAL = args.save_to_pool_interval
    if args.curriculum_eval_games: cfg.CURRICULUM_EVAL_GAMES = args.curriculum_eval_games
    if args.ent_coef is not None: cfg.ENT_COEF = args.ent_coef
    if args.lr is not None: cfg.LEARNING_RATE = args.lr
    if args.seed: cfg.SEED = args.seed
    # Warm-started (BC) policies are already near-deterministic; the default
    # entropy bonus (0.01) and LR (3e-4) destroy them during fine-tuning.
    # Lower both unless the user overrides explicitly.
    if args.init_from_pretrained and args.ent_coef is None:
        cfg.ENT_COEF = min(cfg.ENT_COEF, 0.001)
    if args.init_from_pretrained and args.lr is None:
        cfg.LEARNING_RATE = min(cfg.LEARNING_RATE, 1e-4)
    # Curriculum is disabled by explicit flag or by a fixed-grid CLI override.
    if args.no_curriculum or args.grid_min or args.grid_max:
        cfg.CURRICULUM = None
    if cfg.CURRICULUM is not None:
        final_max = max(s[1] for s in cfg.CURRICULUM)
        cfg.PAD_TO = max(cfg.PAD_TO, final_max + 2)


def main():
    args = parse_args()
    apply_args(config, args)

    # Map-size curriculum: start on the first stage's grid
    curriculum_enabled = config.CURRICULUM is not None
    cur_stage = 0
    stage_start_iter = 0
    if curriculum_enabled:
        config.MIN_GRID_DIMS, config.MAX_GRID_DIMS = curriculum_stage_dims(config, 0)
        print(f"Curriculum ON: {len(config.CURRICULUM)} stages "
              f"({config.CURRICULUM[0][0]} -> {config.CURRICULUM[-1][1]}), "
              f"starting on {config.MIN_GRID_DIMS[0]}x{config.MIN_GRID_DIMS[0]}")

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Device: {device}  |  grid {config.MIN_GRID_DIMS}-{config.MAX_GRID_DIMS}  "
          f"|  envs={config.NUM_ENVS} steps={config.NUM_STEPS} iter={config.NUM_ITERATIONS}")

    run_name = args.run_name or f"grid{config.MAX_GRID_DIMS[0]}"
    writer = SummaryWriter(log_dir=f"runs/generals_{run_name}")
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Policy + optimizer
    network = PolicyValueNetwork(
        input_channels=config.INPUT_CHANNELS,
        hidden_channels=config.HIDDEN_CHANNELS,
        grid_size=config.PAD_TO,
        value_hidden=config.VALUE_HIDDEN,
    ).to(device)
    n_params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {n_params:,}")
    optimizer = torch.optim.Adam(network.parameters(), lr=config.LEARNING_RATE, eps=1e-5)

    # Optional behavior-cloning warm-start
    if args.init_from_pretrained:
        state = torch.load(args.init_from_pretrained, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        net_state = network.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in net_state and net_state[k].shape == v.shape:
                net_state[k] = v
                loaded += 1
        network.load_state_dict(net_state)
        print(f"Warm-started from {args.init_from_pretrained}: loaded {loaded}/{len(net_state)} params")

    # Self-play pool + bots
    pool = OpponentPool(network, device=device, pad_to=config.PAD_TO, pool_size=config.POOL_SIZE)
    bot_pool = [RandomAgent(), ExpanderAgent(), RandomAgent(), ExpanderAgent()]
    if args.use_human_exe_opponent:
        if not _HAS_HUMAN_EXE:
            sys.exit("--use-human-exe-opponent requires the vendored human_exe bot deps "
                     "(logbook, ortools, disjoint-set, unionfind); see README.")
        bot_pool.append(HumanExeAgent(player_id="agent_1"))
        print("[human_exe] added to opponent pool (slow; sampled at 1/(len(bot_pool)) "
              "of non-self-play choices)")

    # Environments
    envs, opp_agents = rebuild_envs(config, config.NUM_ENVS, pool, bot_pool, config.SEED,
                                    selfplay_prob_at(config, 0))

    global_step = 0
    best_eval = -1.0
    t_start = time.time()

    for iteration in range(config.NUM_ITERATIONS):
        t0 = time.time()
        sp_prob = selfplay_prob_at(config, iteration)
        buffers, stats = collect_rollout(envs, opp_agents, network, pool, bot_pool, device,
                                         config, sp_prob)
        buf_obs, buf_masks, buf_actions, buf_logprobs, buf_values, buf_rewards, buf_dones, next_values = buffers

        advantages = compute_gae(buf_rewards, buf_values, buf_dones, next_values,
                                 config.GAMMA, config.GAE_LAMBDA)
        returns = advantages + buf_values
        losses = ppo_update(network, optimizer, buffers, advantages, returns, device, config)
        global_step += config.NUM_ENVS * config.NUM_STEPS

        elapsed = time.time() - t0
        sps = (config.NUM_ENVS * config.NUM_STEPS) / max(elapsed, 1e-6)

        # Logging
        eps = max(stats["episodes"], 1)
        win_rate = stats["wins"] / eps
        avg_len = stats["length_sum"] / eps
        avg_shaped = stats["reward_sum"] / max(stats["steps"], 1)

        if iteration % config.LOG_INTERVAL == 0 or iteration == 0:
            print(
                f"iter {iteration:5d} | steps {global_step:>9d} | loss {losses['pg']:.4f} "
                f"v {losses['v']:.3f} ent {losses['ent']:.3f} | win {win_rate:.2f} "
                f"({stats['wins']}/{stats['episodes']}) | avg_len {avg_len:5.0f} | "
                f"shaped {avg_shaped:+.4f} | {sps:6.0f} sps | {elapsed:.1f}s"
            )
            writer.add_scalar("train/win_rate", win_rate, global_step)
            writer.add_scalar("train/episodes", stats["episodes"], global_step)
            writer.add_scalar("train/avg_episode_length", avg_len, global_step)
            writer.add_scalar("train/avg_shaped_reward", avg_shaped, global_step)
            writer.add_scalar("train/policy_loss", losses["pg"], global_step)
            writer.add_scalar("train/value_loss", losses["v"], global_step)
            writer.add_scalar("train/entropy", losses["ent"], global_step)
            writer.add_scalar("train/sps", sps, global_step)

        # Snapshot into self-play pool (FIFO; OpponentPool evicts the oldest)
        if (iteration + 1) % config.SAVE_TO_POOL_INTERVAL == 0:
            pool.add(network)

        # Checkpoint
        if (iteration + 1) % config.CHECKPOINT_INTERVAL == 0:
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"model_{iteration + 1}.pt")
            torch.save({"model_state": network.state_dict(), "iteration": iteration + 1,
                        "global_step": global_step}, ckpt_path)
            torch.save({"model_state": network.state_dict(), "iteration": iteration + 1,
                        "global_step": global_step}, os.path.join(config.CHECKPOINT_DIR, "latest.pt"))

        # Evaluate vs bots
        if config.EVAL_INTERVAL and (iteration + 1) % config.EVAL_INTERVAL == 0:
            results = evaluate(network, config, num_games=args.eval_games,
                               device=device, greedy=False, seed=config.SEED + iteration)
            for opp_name, r in results.items():
                print(f"  eval vs {opp_name}: win_rate {r['win_rate']:.2f} ({r['wins']}/{r['wins'] + r['losses'] + r['draws']})")
                writer.add_scalar(f"eval/win_rate_{opp_name}", r["win_rate"], global_step)
            avg_win = float(np.mean([r["win_rate"] for r in results.values()]))
            if avg_win > best_eval:
                best_eval = avg_win
                torch.save({"model_state": network.state_dict(), "iteration": iteration + 1,
                            "global_step": global_step},
                           os.path.join(config.CHECKPOINT_DIR, "best.pt"))

        # Curriculum advance: bigger maps once the current stage is mastered
        if curriculum_enabled and cur_stage < len(config.CURRICULUM) - 1 and \
                (iteration + 1) % config.CURRICULUM_EVAL_INTERVAL == 0:
            results = evaluate(network, config, num_games=config.CURRICULUM_EVAL_GAMES,
                               device=device, greedy=True, seed=config.SEED + iteration)
            win_rate = float(np.mean([r["win_rate"] for r in results.values()]))
            _, _, bar, cap = config.CURRICULUM[cur_stage]
            iterations_in_stage = (iteration + 1) - stage_start_iter
            if win_rate >= bar or (cap and iterations_in_stage >= cap):
                cur_stage += 1
                if cur_stage < len(config.CURRICULUM):
                    config.MIN_GRID_DIMS, config.MAX_GRID_DIMS = curriculum_stage_dims(config, cur_stage)
                    envs, opp_agents = rebuild_envs(config, config.NUM_ENVS, pool, bot_pool,
                                                    config.SEED + cur_stage,
                                                    selfplay_prob_at(config, iteration + 1))
                    stage_start_iter = iteration + 1
                    print(f"  [curriculum] advanced to stage {cur_stage}: "
                          f"{config.MIN_GRID_DIMS[0]}x{config.MIN_GRID_DIMS[0]}-"
                          f"{config.MAX_GRID_DIMS[1]}x{config.MAX_GRID_DIMS[1]} "
                          f"(eval win_rate {win_rate:.2f} vs bar {bar})")
                    writer.add_scalar("curriculum/stage", cur_stage, global_step)
                else:
                    print("  [curriculum] reached final stage")

    writer.close()
    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time / 60:.1f} min. Checkpoints in {config.CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
