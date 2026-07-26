"""**Names of actions.** They can be used as MACTIONS only.
**WARNING!** Many named actions are not a GameAction object"""
__all__ = ["named_actions"]

from ._abc import GameAction
from .presets import (
                      GameAction_run,
                      GameAction_keyboard,
                      GameAction_mouse
                     )

# Named action can be used as MACTION only.
named_actions: dict[str, type[GameAction]] = {
    "run": GameAction_run,
    "kb": GameAction_keyboard, "keyboard": GameAction_keyboard,
    "mouse": GameAction_mouse
}