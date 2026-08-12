"""
BotModules Centralized Imports

This module provides centralized access to all BotModules classes
while handling circular dependencies through lazy imports.
"""
from __future__ import annotations


from BotModules import __getattr__

class _ModuleProxy:
    def __getattr__(self, name):
        return __getattr__(name)

modules = _ModuleProxy()

__all__ = ['modules']
