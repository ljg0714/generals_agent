"""Single-agent Gymnasium wrapper around the two-player Generals game.

The wrapper exposes only player-0's view (the learner). Player 1 (the opponent)
is controlled externally: the training loop fetches the opponent's raw
observation via `opponent_observation()` before each step, computes the
opponent's action, and passes both actions into `step()`.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from generals.core.action import Action
from generals.core.grid import GridFactory
from generals.envs.gymnasium_generals import GymnasiumGenerals

from agents.game_patch import patch_env_growth


class GeneralsEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        opponent=None,
        grid_factory: GridFactory | None = None,
        pad_observations_to: int = 24,
        truncation: int | None = 500,
        render_mode: str | None = None,
        agent_id: str = "agent_0",
        opponent_id: str = "agent_1",
    ):
        super().__init__()
        self.agent_id = agent_id
        self.opponent_id = opponent_id
        self.opponent = opponent
        self.pad_to = pad_observations_to
        # Use generals.io-accurate army growth (config-driven, idempotent).
        patch_env_growth()
        self._env = GymnasiumGenerals(
            agents=[agent_id, opponent_id],
            grid_factory=grid_factory or GridFactory(),
            pad_observations_to=pad_observations_to,
            truncation=truncation,
            render_mode=render_mode,
        )
        dim = pad_observations_to
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(15, dim, dim), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([2, dim, dim, 4, 2])

        # Cache of the opponent's raw observation for the *next* decision point.
        self._opp_raw_obs = None

    # ------------------------------------------------------------------
    # Opponent handling
    # ------------------------------------------------------------------
    def set_opponent(self, opponent):
        self.opponent = opponent

    def opponent_observation(self):
        """Raw Observation for the opponent at the current pre-decision state."""
        return self._opp_raw_obs

    def _refresh_opponent_observation(self):
        self._opp_raw_obs = self._env.game.agent_observation(self.opponent_id)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._refresh_opponent_observation()  # observation before the first action
        self.last_obs = np.asarray(obs[0], dtype=np.float32)
        self.last_mask = np.asarray(info[self.agent_id]["masks"], dtype=bool)
        return self.last_obs, info[self.agent_id]

    def step(self, action, opponent_action):
        """Both `action` (player 0) and `opponent_action` (player 1) are actions."""
        if isinstance(action, np.ndarray):
            action = Action(int(action[0]), int(action[1]), int(action[2]), int(action[3]), int(action[4]))
        if isinstance(opponent_action, np.ndarray):
            opponent_action = Action(
                int(opponent_action[0]), int(opponent_action[1]), int(opponent_action[2]),
                int(opponent_action[3]), int(opponent_action[4]),
            )
        obs, _reward, terminated, truncated, infos = self._env.step([action, opponent_action])
        self.last_obs = np.asarray(obs[0], dtype=np.float32)
        self.last_mask = np.asarray(infos[self.agent_id]["masks"], dtype=bool)
        self._refresh_opponent_observation()  # observation for the next decision point
        return self.last_obs, 0.0, bool(terminated), bool(truncated), infos[self.agent_id]

    def render(self):
        if self._env.render_mode == "human":
            self._env.render()

    def close(self):
        self._env.close()
