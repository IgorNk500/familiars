"""GameAction and GameTrigger class"""
import abc

class GameAction(abc.ABC):
    """Base of game action.
    **Abstract class.**

    **Functions:**\n
    + **__init__:** Abstract constructor, supports any action args
    + **activate:** Main activate function"""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs): pass

    @abc.abstractmethod
    def activate(self, *args, **kwargs): pass

    __call__ = activate

class GameTrigger(abc.ABC):
    """Base of game trigger.
    **Abstract class.**

    **Functions:**\n
    + **__init__:** Abstract constructor, supports any trigger args
    + **check -> bool:** Main check function"""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs): pass

    @abc.abstractmethod
    def check(self) -> bool:  pass

    __call__ = check
