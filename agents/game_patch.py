"""Runtime patches that fix bugs in the installed generals-bots package.

1. Army growth (config-driven, currently FAITHFUL to real generals.io).
   In this env 2 steps = 1 real turn. Faithful growth = +1 per owned cell every
   50 steps (= every 25 turns, matches real floor(land/25)) and +1 on
   generals/cities every 2 steps (= every real turn). `FastGame` reads
   ARMY_INCREMENT_RATE / CITY_GROWTH_PERIOD from config, so you can also set a
   faster 25/1 for training speed, but policies trained on the faster rate do
   NOT transfer faithfully to the real game.
   The env constructs `Game` in both `GymnasiumGenerals.__init__` and
   `reset()`; `patch_env_growth()` swaps the `Game` name in the env module for
   `FastGame` (a subclass), so both construction points pick it up.

2. GUI clock. `generals.gui.properties.Properties` declares
   `__clock: Clock = Clock()` as a dataclass *class-level* default, so every
   `GUI` shares ONE Clock. `env.close()` calls `pygame.quit()`, which resets
   SDL's tick timebase to ~0 while the shared Clock's `last_tick` still holds the
   previous episode's timestamp. The next episode's first `clock.tick(fps)` then
   computes a huge negative frame time and `SDL_Delay()`s for ~the previous
   episode's duration — play.py appears frozen at episode 2. `patch_gui_clock()`
   gives each `Properties` instance its own fresh Clock.

NOTE: The base env's fog re-hides explored cells every step (visibility = 3x3 of
current owned cells). Real generals.io reveals explored cells permanently. That
memory is now provided by the in-network `MemoryAugmenter` (agents/memory_aug.py)
— channels for seen generals/cities/mountains persist inside the network — so
`FastGame` no longer injects procedural memory into the observation channels.
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


def patch_gui_clock():
    """Give each GUI its own fresh Clock instead of the shared class-level one.

    Idempotent. `Properties`' dataclass field `__clock: Clock = Clock()` is
    evaluated once at class-definition time, so every Properties instance reads
    the SAME Clock. After `pygame.quit()` (from `env.close()`), SDL resets its
    tick timebase but the shared Clock's `last_tick` keeps the previous episode's
    timestamp; the next `clock.tick(fps)` then waits `1/fps - (tick_now - stale)`,
    which is huge, and the window appears frozen for ~the previous episode's
    duration. Replacing `__init__` gives each instance its own fresh Clock.
    """
    from pygame.time import Clock
    from generals.gui import properties as _props

    if getattr(_props.Properties, "_clock_patched", False):
        return

    _orig_init = _props.Properties.__init__

    def _init(self, game, agent_data, mode, game_speed=1.0):
        _orig_init(self, game, agent_data, mode, game_speed)
        # Dataclass name-mangles `__clock` to `_Properties__clock`; replace the
        # shared class-level Clock with one that starts fresh for this GUI.
        self._Properties__clock = Clock()

    _props.Properties.__init__ = _init
    _props.Properties._clock_patched = True
