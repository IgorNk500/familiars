"""Familiars Tools. BaseIO and FamiliarIO.

Copyright (c) 2026 IgorNk500"""
import abc
from typing import Callable

from .actions import GameActions
from .models import FamiliarModel, NormalDataFamiliarModel
from .exceptions import NumActionsError, ModelTypeError, ModelBrokenError

class BaseIO(abc.ABC):
    """ABC Class. BaseIO - the basis for any Familiars' IO"""
    supports_evol_training: bool = True

    def __init__(self,
                 model: FamiliarModel,
                 actions: GameActions):
        self.model = model
        self.actions = actions

        self._model_num_args = model.config.num_args
        self._model_num_actions = model.config.num_actions

    @abc.abstractmethod
    def play(self, *args, **kwargs): pass

    @property
    @abc.abstractmethod
    def is_dead(self): pass


class FamiliarIO(BaseIO):
    """**DataFamiliarIO - the basis of the model's communication with the game**

        Uses *GameActions* to handle the model's actions and work with training events.

        Read more in the docs.

        :arg model: Familiar pretrained model
        :arg actions: GameActions object
        :arg get_input_func: A function for receiving input data. `get_input_func -> list[float]`; `len(get_input_func()) == `"""

    def __init__(self,
                 model: FamiliarModel,
                 actions: GameActions,
                 get_input_func: Callable):
        super().__init__(model, actions)
        self.get_input_func = get_input_func

        if not isinstance(model, NormalDataFamiliarModel): raise ModelTypeError("Model for FamiliarIO must be DataFamiliar only."
                                                                                "For a different model use another IO"
                                                                                "For example, screen-familiars.ScreenFamiliarIO")

        # Check the actions num
        if model.config.num_actions != actions.num_actions:
            raise NumActionsError(
                f"Model num_actions ({model.config.num_actions}) != GameActions num_actions ({actions.num_actions})"
            )

    def play(self):
        """Playing cycle. WHILE NOT DEAD"""
        i = 0 # Iteration counter for custom play_once function
        while not self.is_dead:
            self._play_once(i=i)
            self.actions.update()
            i += 1
        return i


    def _play_once(self, *args, **kwargs):
        """**MAIN PLAYING FUNCTION**"""
        # STEP 1: GET INPUT FROM get_input_func
        inp = self.get_input_func()

        # STEP 2: GENERATE RESULT
        action, args = self.model(inp)

        # STEP 3: Check if the model is broken
        if action > self._model_num_actions or len(args) != self._model_num_args:
            raise ModelBrokenError(f"{self.model.__class__.__name__} is broken."
                                   f"If you think otherwise, please open issue on GitHub.")

        # STEP 4: Run action with args
        self.actions.run_maction(action, *args)

    @property
    def is_dead(self):
        return self.actions.death_event() and self.actions.death_event.is_set()

