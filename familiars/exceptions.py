"""Familiars exceptions module"""
from typing import Optional


class FamiliarsException(Exception):
    """Main exception class for familiars library"""

    def __init__(self, msg: Optional[str]):
        self.message = msg

    def __str__(self):
        if self.message:
            return self.message
        else:
            return f'{self.__class__.__name__} has been raised'


##########     EXCEPTIONS     ##########

class NumActionsError(FamiliarsException):
    """The `num_actions` of the model != the `num_actions` specified in the `GameActions` class."""
    pass


class ModelTypeError(FamiliarsException):
    """**You tried to connect an unsupported model to IO.**

    For example, when you try to connect a `NormalScreenFamiliarModel` or an `EvolScreenFamliarModel` to `FamiliarIO`.
    There are `screen-familiars.ScreenIO` for these models."""
    pass

class IOTypeError(FamiliarsException):
    """You tried to connect an unsupported IO"""
    pass


class ModelBrokenError(FamiliarsException):
    """Your model is broken.
    *If you think otherwise, please open issue on GitHub.*"""
    pass
