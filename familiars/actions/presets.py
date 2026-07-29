"""Presets of game actions and triggers

**List of presets:**\n
+ **GameAction_run** (run): runs cmd/bash commands with subprocess
+ **GameAction_keyboard**"""

__all__ = [
    "ACTION_CLICK",
    "ACTION_PRESS",
    "ACTION_RELEASE",
    "ACTION_MOUSE_LEFT",
    "ACTION_MOUSE_RIGHT",
    "ACTION_MOUSE_DOUBLE_CLICK",
    "GameAction_run",
    "GameAction_keyboard",
    "GameAction_mouse"
]

import keyboard as kb
import mouse as m
from subprocess import run as sp_run
from typing import Literal, Optional

from ._abc import GameAction, GameTrigger

# Constants
ACTION_PRESS = "press"
ACTION_RELEASE = "release"
ACTION_CLICK = "click"
ACTION_MOUSE_DOUBLE_CLICK = "dclick"

ACTION_MOUSE_LEFT = "left"
ACTION_MOUSE_RIGHT = "right"

# Classes

class GameAction_run(GameAction):
    """Runs cmd/bash commands with subprocess"""
    def __init__(self, **kwargs):
        self.command = kwargs

    def activate(self):
        return sp_run(self.command)

class GameAction_keyboard(GameAction):
    """**Click/Press/Release button on keyboard.**

    Action argument must be "click", "press" or "release". Check ACTION_... constants.

    The hotkey must be a key (Ctrl; A; 1; f4) or a keyboard shortcut (Ctrl+A; Alt+Shift).
    """

    def __init__(self, action: Literal["press", "release", "click"], hotkey: str):
        self.action = action
        self.hotkey = hotkey

    def activate(self):
        act = self.action
        if act == "click":
            kb.send(self.hotkey)
        elif act == "press":
            kb.press(self.hotkey)
        elif act == "release":
            kb.release(self.hotkey)
        else:
            raise ValueError('Action argument must be "click", "press" or "release". Check ACTION_... constants.')

class GameAction_mouse(GameAction):
    """**Click/Press/Release/Double-click left/right mouse button**

    Action argument must be "click", "press", "release" or "dclick". Check ACTION_... constants.

    Button argument must be "left" or "right". Check ACTION_MOUSE_... constants
    """

    def __init__(self, action: Literal["press", "release", "click"], button: Literal["left", "right"], px: Optional[int] = None, py: Optional[int] = None):
        self.action = action

        if button == "left":
            self.button = False
        elif button == "right":
            self.button = True
        else:
            raise ValueError('Mouse button must be "left" or "right". Check ACTION_MOUSE_... constants')

        self.px = px
        self.py = py

    def activate(self):
        act = self.action
        if self.px and self.py:
            m.move(self.px, self.py, True)

        if act == "click":
            if self.button:
                m.click(m.RIGHT)
            else: m.click()
        elif act == "press":
            if self.button:
                m.press(m.RIGHT)
            else:
                m.press()
        elif act == "release":
            if self.button:
                m.release(m.RIGHT)
            else:
                m.release()
        elif act == "dclick":
            if self.button:
                m.double_click(m.RIGHT)
            else: m.double_click()
        else:
            raise ValueError('Action argument must be "click", "press", "release" or "dclick". Check ACTION_... constants.')
