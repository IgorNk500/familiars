"""
This is the GameActions package, which is designed to provide a convenient way to control the game using Familiars.
This is a short guide on how to use it, and you can read the rest in the documentation.

**Classes:**
+ **GameActions:** The main class for managing actions
+ **GameAction*:** Action class. Can be used for inheritance.
+ **GameTrigger:** Trigger class.
"""
__all__ = ["GameActions", "GameAction", "GameTrigger", "named_actions"]

from threading import Event

from ._abc import GameAction, GameTrigger
from ._named import named_actions
from ._loader import GameActionsDC as _GameActionsDC

class GameActions:
    """**GameActions is the main class for managing actions, events and triggers.**

     It is he who combines your *familiar* with the game and allows you to track training progress.

     **Read more in the docs**"""
    def __init__(self, mactions: list[GameAction] = None, aliases: dict[str | GameTrigger, str | GameAction] = None):
        if mactions is None:
            mactions = []
        if aliases is None:
            aliases = {}

        # MACTIONS - The model's actions
        self._mactions = mactions

        self._aliases: dict[str | GameTrigger, str | GameAction] = aliases

        # Setup all overloaded events
        # These events are controlled by GameActions, and sent to Trainer.
        self.stop_train_event = Event()
        self.start_train_event = Event()
        self.end_event = Event()
        self.save_event = Event()
        self.death_event = Event() # death_event - called by the algorithm or alias when the bot dies

        # Setup all overloaded triggers
        # These events are controlled by Trainer, and sent to GameActions.
        self.start_trigger = Event()
        self.pre_init_trigger = Event()
        self.post_init_trigger = Event()
        self.pre_cycle_trigger = Event()
        self.pre_train_trigger = Event()
        self.post_train_trigger = Event()
        self.pre_using_trigger = Event()
        self.post_using_trigger = Event()
        self.ending_trigger = Event()

    def add_alias(self, trg: str | GameTrigger, act: str | GameAction):
        """**Adds alias:** IF *trigger* THEN *action*"""

        # Process trigger
        if isinstance(trg, str):
            _trg = self._find_str_trigger(trg)
        else: _trg = trg

        # Process action
        if isinstance(act, str):
            _act = self._find_str_action(act)
        else: _act = act

        if not _trg:
            raise ValueError("Trigger %s not found" % act)
        elif not _act:
            raise ValueError("Action %s not found" % act)

        # Add alias
        self._aliases[_trg] = _act

    def add_maction(self, name: str | type[GameAction], *args, **kwargs):
        """**Adds maction at the end of the list.**

        **MACTIONS - The model's actions.**

        Maction cannot be overloaded action.

        :arg name: Name of the action
        :arg args: Arguments of the action
        :arg kwargs: Pos. arguments of the action"""
        if isinstance(name, str):
            if name in named_actions.keys():
                # Create an action
                self._mactions.append(
                    named_actions[name](*args, **kwargs)
                )
            else: raise ValueError("MACTION not found in named_actions.")
        else:
            self._mactions.append(
                name(*args, **kwargs)
            )

    def run_maction(self, aid: int, *args, **kwargs):
        self._mactions[aid].activate(*args, **kwargs)

    def trigger(self, name: str):
        name = name + "_trigger"
        if hasattr(self, name):
            trg = getattr(self, name)
            trg.set()
        else: raise ValueError(name + " not found")
        self.update()

    def untrigger(self, name: str):
        name = name + "_trigger"
        if hasattr(self, name):
            trg = getattr(self, name)
            trg.clear()
        else: raise ValueError(name + " not found")
        self.update()


    def update(self):
        """Updates aliases"""
        for trg, act in self._aliases:
            if isinstance(trg, str):
                _trg = getattr(self, trg)
                if _trg and _trg.is_set():
                    self._run_action(act)
                    _trg.clear()
            elif trg.check():
                self._run_action(act)

    def check_action(self, name: str):
        """Checks the status of the action"""
        _name = self._find_str_action(name)
        if not name: raise ValueError("Action %s not found" % name)
        act = getattr(self, _name)
        if act and act.is_set():
            act.clear()
            return True
        else: return False

    @property
    def num_actions(self) -> int:
        """Returns num of the **MACTIONS**"""
        return len(self._mactions)

    def to_pickle(self, fp: str):
        """You can export game actions to pickle file"""
        dc = _GameActionsDC(
            self._mactions,
            self._aliases
        )
        dc.to_pickle(fp)

    def to_pretrained(self, path: str):
        """You can save your game actions to pretrained model on your computer.
        Actions will be saved in `actions.bin` pickle file in the model root."""
        dc = _GameActionsDC(
            self._mactions,
            self._aliases
        )
        dc.to_pretrained(path)

    # System methods
    def _find_str_action(self, name: str) -> str | None:
        name = name + "_action"
        if hasattr(self, name):
            return name
        else: return None

    def _find_str_trigger(self, name: str) -> str | None:
        name = name + "_trigger"
        if hasattr(self, name):
            return name
        else:
            return None

    def _run_action(self, act: str | GameAction):
        if isinstance(act, str):
            getattr(self, act).set()
        else:
            act.activate()

    @classmethod
    def from_pickle(cls, fp: str) -> GameActions:
        """You can load game actions from pickle file
        **!!!WARNING!!! Load pickle files only from trusted sources! Pickle files can contain viruses! !!!WARNING!!!**"""
        dc = _GameActionsDC.from_pickle(fp)
        return cls(
            dc.mactions,
            dc.aliases
        )

    @classmethod
    def from_pretrained(cls, name_or_path: str, *args, **kwargs):
        """You can load game actions from pretrained model on *HuggingFace* or on local computer.
        Model files must contain `actions.bin` pickle file created with `actions.save_pretrained("name_of_your_model")`"""
        dc = _GameActionsDC.from_pretrained(name_or_path, *args, **kwargs)
        return cls(
            dc.mactions,
            dc.aliases
        )



