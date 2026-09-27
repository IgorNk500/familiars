"""Familiars Tools. EvalManager.

**EvalManager contains model populations and their properties.**
Used together with *FamiliarsIO* *(or screen-familiars.ScreenIO)* and *EvalTrainer*

Copyright (c) 2026 IgorNk500"""
import copy

from ..io import BaseIO
from ..exceptions import ModelTypeError, IOTypeError

class EvalManager:
    """**EvalManager contains model populations and their properties.**
    Used together with `FamiliarsIO` *(or `screen-familiars.ScreenIO`)* and `EvalTrainer`"""

    def __init__(self, io: BaseIO, population_size: int):
        self.io = io
        self.model = io.model # Model link
        self.actions = io.actions # GameActions link

        # Is IO supports eval training?
        if not io.supports_evol_training: raise IOTypeError("{} doesn't support eval training.".format(io.__class__.__name__))

        # Is it an eval model?
        if not self.model.is_evol: raise ModelTypeError("{} isn't eval model.".format(self.model.__class__.__name__))

        self.population_size = population_size

        # First best model - main model
        self._best_state = self.model.state_dict()

        # Initialize population
        self.population = []
        self.reset_population()

    def reset_population(self):
        """Resets the population from loaded model"""
        base_state = copy.deepcopy(self.model.state_dict())
        self.population = [copy.deepcopy(base_state) for _ in range(self.population_size)]

    def set_best(self, state: dict):
        """Sets the best population from trainer"""
        self._best_state = state

    @property
    def best(self):
        """Returns the best model"""
        return copy.deepcopy(self.model).load_state_from_dict(self._best_state)