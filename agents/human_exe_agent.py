"""Bridge the EklipZ bot ("human_exe") into the gymnasium generals environment.

The EklipZ bot (EklipZgit/generals-bot, vendored under `agents/human_exe`) is a
purely algorithmic 1v1 generals.io bot that runs on its own `MapBase` world model.
This adapter makes it act as a gymnasium player:

  1. Build a full-truth `MapBase` from the gymnasium `Game` each turn.
  2. Build the bot's fog-restricted player map on the first turn via
     `Sim/GameSimulator.generate_player_map`, then apply a fog-correct in-place
     sync each turn driven by `game.channels.get_visibility(player)`. We do NOT
     reuse the bot's `GamePlayer.set_map_vision`, which leaks hidden armies and
     enemy-general positions into fog tiles.
  3. Ask `EklipZBot.find_move()` and convert the returned Move to a gymnasium
     action.

The bot is stateful per game, so each environment / game needs its own instance
(or a call to `reset()`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HX_DIR = Path(__file__).resolve().parent / "human_exe"
if str(_HX_DIR) not in sys.path:
    sys.path.insert(0, str(_HX_DIR))

# Headless: neutralize the EklipZ bot's file-logging (it also reads run_config.txt
# from a repo-parent path that doesn't exist here). Must run before EklipZBot is used.
from BotModules import BotLifecycle  # noqa: E402

_original_initialize_logging = BotLifecycle.initialize_logging
BotLifecycle.initialize_logging = lambda bot: None  # type: ignore[assignment]

# Disable the bot's debug/unit-test assertions (some dependency imports `unittest`,
# flipping IS_DEBUG_OR_UNIT_TEST_MODE on and making the bot's self-checks throw on
# maps the gymnasium env generates). Must be set before EklipZBot runs.
import DebugHelper  # noqa: E402

DebugHelper.IS_DEBUGGING = False
DebugHelper.IS_RUNNING_UNIT_TESTS = False
DebugHelper.IS_DEBUG_OR_UNIT_TEST_MODE = False

from base.client.map import (  # noqa: E402
    MapBase,
    Player,
    Score,
    Tile,
    TILE_EMPTY,
    TILE_MOUNTAIN,
    TILE_FOG,
    TILE_OBSTACLE,
)
from bot_ek0x45 import EklipZBot  # noqa: E402
from generals.core.action import Action  # noqa: E402
from Sim.GameSimulator import generate_player_map  # noqa: E402

# Direction of a Move destination relative to its source: (dy, dx) -> DIRECTIONS index.
_DIRMAP = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}


def _build_full_map(game) -> MapBase:
    """Build a full-truth MapBase snapshot from a gymnasium Game."""
    H, W = game.grid_dims
    armies = np.asarray(game.channels.armies)
    mountains = np.asarray(game.channels.mountains, dtype=bool)
    cities = np.asarray(game.channels.cities, dtype=bool)
    generals = np.asarray(game.channels.generals, dtype=bool)
    agents = list(game.agents)

    # Owner index per cell: 0/1 for players, -1 for neutral.
    owner_idx = np.full((H, W), -1, dtype=np.int32)
    for p, agent in enumerate(agents):
        owner_idx[np.asarray(game.channels.ownership[agent], dtype=bool)] = p

    infos = game.get_infos()
    map_grid: list[list[Tile]] = []
    for y in range(H):
        row: list[Tile] = []
        for x in range(W):
            owner = int(owner_idx[y, x])
            is_mountain = bool(mountains[y, x])
            is_city = bool(cities[y, x])
            is_general = bool(generals[y, x])
            tile_type = TILE_MOUNTAIN if is_mountain else (owner if is_city else TILE_EMPTY)
            t = Tile(
                x, y,
                tile=tile_type,
                army=int(armies[y, x]),
                isCity=is_city,
                isGeneral=is_general,
                player=owner,
                isMountain=is_mountain,
                tileIndex=y * W + x,
            )
            row.append(t)
        map_grid.append(row)

    map_raw = MapBase(
        player_index=0,
        teams=[0, 1],
        user_names=agents,
        turn=int(game.time),
        map_grid_y_x=map_grid,
        replay_url="gymnasium",
        modifiers=[],
    )
    map_raw.is_custom_map = False

    scores = []
    for p, agent in enumerate(agents):
        info = infos[agent]
        total = int(info["army"])
        land = int(info["land"])
        dead = bool(info["is_done"] and not info["is_winner"])
        gtile = map_grid[game.general_positions[agent][0]][game.general_positions[agent][1]]
        map_raw.players[p].general = gtile
        map_raw.generals[p] = gtile
        scores.append(Score(p, total, land, dead))
    map_raw.update_scores(scores)
    map_raw.update_turn(int(game.time))
    map_raw.init_grid_movable()
    map_raw.update(bypassDeltas=True)
    return map_raw


class HumanExeAgent:
    """Wraps one EklipZBot and exposes `act_on_game(game, agent_id) -> Action`.

    Not a generals-bots `Agent` (it needs the full `Game`, not just the fogged
    observation), so call sites special-case it.
    """

    def __init__(self, player_id: str = "agent_1", pad_to: int = 24, max_tracebacks: int = 3):
        self.player_id = player_id
        self.pad_to = pad_to
        self.max_tracebacks = max_tracebacks
        self.bot = EklipZBot()
        self.bot.no_file_logging = True
        self._player_map: MapBase | None = None
        self._started = False

    def reset(self):
        """Start a fresh game (fresh EklipZBot + empty player map)."""
        self.bot = EklipZBot()
        self.bot.no_file_logging = True
        self._player_map = None
        self._started = False

    def act_on_game(self, game, agent_id: str | None = None):
        """Compute a gymnasium action from the full game state.

        Returns a numpy (5,) action [pass, row, col, direction, split].
        """
        agent_id = agent_id or self.player_id
        player_index = game.agents.index(agent_id)
        map_raw = _build_full_map(game)

        if not self._started:
            self._player_map = generate_player_map(player_index, map_raw)
            # Register generals / correct visible tiles BEFORE the bot initializes,
            # since initialize_from_map_for_first_time reads self._map.generals.
            self._sync_player_map(game, map_raw, agent_id)
            self.bot.initialize_from_map_for_first_time(self._player_map)
            self._started = True
        else:
            self._sync_player_map(game, map_raw, agent_id)

        assert self.bot._map is self._player_map, "player map object identity lost"

        # The bot's init_turn/find_move expect a move timer to be open.
        self.bot.perf_timer.begin_move(int(game.time))
        try:
            self.bot.init_turn()
            move = self.bot.find_move()
        except Exception:
            # The EklipZ bot has known latent bugs; a failed turn must degrade to a
            # pass rather than crash the game. Print only the first few tracebacks
            # (for diagnosis) so a long run isn't flooded with identical errors.
            if getattr(self, "_error_count", 0) < self.max_tracebacks:
                import traceback
                traceback.print_exc()
            self._error_count = getattr(self, "_error_count", 0) + 1
            move = None
        return self._move_to_action(move)

    def _sync_player_map(self, game, map_raw: MapBase, agent_id: str):
        """Fog-correct in-place update of the bot's player map from game truth."""
        player_map = self._player_map
        player_index = game.agents.index(agent_id)
        rows, cols = player_map.rows, player_map.cols
        visible = np.asarray(game.channels.get_visibility(agent_id), dtype=bool)
        for y in range(rows):
            for x in range(cols):
                real = map_raw.tiles_by_index[y * cols + x]
                pt = player_map.tiles_by_index[y * cols + x]
                if visible[y, x]:
                    pt.army = real.army
                    pt.player = real.player
                    pt.isCity = real.isCity
                    pt.isMountain = real.isMountain
                    pt.isGeneral = real.isGeneral
                    pt.tile = real.tile
                    pt.visible = True
                    pt.discovered = True
                    if real.isGeneral and real.player is not None and real.player >= 0:
                        player_map.generals[real.player] = pt
                        player_map.players[real.player].general = pt
                else:
                    # generals.io reveals a once-seen general permanently (crown marker),
                    # so keep remembered general tiles visible even behind fog.
                    pt.visible = bool(pt.isGeneral)
                    # generals.io keeps owned territory visible; a tile behind fog is
                    # never ours. A previously-owned tile that was captured and re-fogged
                    # must not stay movable (else the bot moves from a tile it doesn't own).
                    if pt.player == player_index:
                        pt.player = -1
                    if not pt.discovered:
                        # Undiscovered fog. Structure POSITIONS are public (the env's
                        # structures_in_fog channel marks every fogged mountain/city), but
                        # the type is unknown -> a plain obstacle. Non-structures are fog.
                        pt.army = 0
                        pt.player = -1
                        pt.isCity = False
                        pt.isMountain = False
                        pt.isGeneral = False
                        pt.tile = TILE_OBSTACLE if (real.isMountain or real.isCity) else TILE_FOG

        player_map.update_turn(int(game.time))
        infos = game.get_infos()
        scores = []
        for p, agent in enumerate(game.agents):
            info = infos[agent]
            player_map.players[p].score = int(info["army"])
            player_map.players[p].tileCount = int(info["land"])
            player_map.players[p].dead = bool(info["is_done"] and not info["is_winner"])
            scores.append(Score(p, player_map.players[p].score,
                                player_map.players[p].tileCount, player_map.players[p].dead))
        player_map.update_scores(scores)
        player_map.update(bypassDeltas=True)

    def _move_to_action(self, move):
        if move is None:
            return Action(1, 0, 0, 0, 0)  # pass
        dy = move.dest.y - move.source.y
        dx = move.dest.x - move.source.x
        direction = _DIRMAP.get((dy, dx))
        if direction is None:
            return Action(1, 0, 0, 0, 0)  # invalid -> pass
        return Action(0, int(move.source.y), int(move.source.x),
                      direction, int(bool(move.move_half)))

    def __str__(self):
        return "human_exe"
