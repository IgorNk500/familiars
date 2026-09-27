"""**Names of actions.** They can be used as MACTIONS only.
**WARNING!** Many named actions are not a GameAction object"""
__all__ = ["named_actions"]

from ._abc import GameAction
from .presets import (
    GameAction_Run,
    GameAction_Keyboard,
    GameAction_Mouse
                     )

# Named action can be used as MACTION only.
named_actions: dict[str, type[GameAction]] = {
    "run": GameAction_Run,
    "kb": GameAction_Keyboard, "keyboard": GameAction_Keyboard,
    "mouse": GameAction_Mouse
}