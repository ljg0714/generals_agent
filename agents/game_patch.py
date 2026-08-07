"""Patch the generals-bots numpy env to use generals.io-accurate army growth.

The default Game grows armies very slowly (+1 per owned cell every 50 steps,
+1 on generals/cities every 2 steps), so games on 12-14 maps almost never
finish before truncation — terminal rewards were essentially unreachable.

Real generals.io: every owned cell produces ~+1 army every 25 turns, and
cities / the general produce +1 every turn. `FastGame` implements that.

The env constructs `Game` in both `GymnasiumGenerals.__init__` and `reset()`;
`patch_env_growth()` swaps the `Game` name in the env module for `FastGame`
(a subclass), so both construction points pick it up.
"""
from __future__ import annotations

import config as _cfg
from generals.core.game import Game


class FastGame(Game):
    """Game with generals.io-accurate growth (values read from config)."""

    def __init__(self, grid, agents):
        super().__init__(grid, agents)
        self.increment_rate = _cfg.ARMY_INCREMENT_RATE
        self.city_growth_period = _cfg.CITY_GROWTH_PERIOD

    def _global_game_update(self):
        owners = self.agents
        # Every `increment_rate` steps each owned cell produces +1 army.
        if self.time % self.increment_rate == 0:
            for owner in owners:
                self.channels.armies += self.channels.ownership[owner]
        # Owned general/city cells produce +1 army every `city_growth_period` steps.
        if self.time % self.city_growth_period == 0 and self.time > 0:
            update_mask = self.channels.generals + self.channels.cities
            for owner in owners:
                self.channels.armies += update_mask * self.channels.ownership[owner]


def patch_env_growth():
    """Make GymnasiumGenerals construct FastGame instead of Game (idempotent)."""
    import generals.envs.gymnasium_generals as gg

    gg.Game = FastGame
