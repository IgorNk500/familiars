"""Familiars. Neural AI models for playing games
-----

*A familiar is a magical spirit or creature that,
according to European folklore and witchcraft traditions,
serves a witch, sorcerer, or mage, assisting them in sorcery,
protecting them from enemies, and often acting as a loyal companion.*

Links:
-----

+ **GitHub:** https://github.com/IgorNk500/familiars
+ **Documentation:** https://github.com/IgorNk500/familiars/wiki
+ **Issues:** https://github.com/IgorNk500/familiars/issues
+ **Changelog:** https://github.com/IgorNk500/familiars/blob/main/CHANGELOG.md

Copyright (c) 2026 IgorNk500
Licensed MIT
"""
__version__ = "0.0.1"
__author__ = "IgorNk500"
__all__ = [
    "pipeline", "FamiliarIO", "EvolManager", "EvolTrainer", "GameActions", "GameAction", "GameTrigger",
    "FamiliarModel", "FamiliarConfig",
    "NormalDataFamiliarModel", "NormalDataFamiliarConfig",
    "NormalScreenFamiliarModel", "NormalScreenFamiliarConfig",
    "EvolDataFamiliarModel", "EvolDataFamiliarConfig",
    "EvolScreenFamiliarModel", "EvolScreenFamiliarConfig"
]

from .pipeline import pipeline

from .io import FamiliarIO
from .evol import EvolManager, EvolTrainer
from .actions import GameActions, GameAction, GameTrigger

from .models import FamiliarModel, FamiliarConfig
from .models import NormalDataFamiliarModel, NormalDataFamiliarConfig
from .models import NormalScreenFamiliarModel, NormalScreenFamiliarConfig
from .models import EvolDataFamiliarModel, EvolDataFamiliarConfig
from .models import EvolScreenFamiliarModel, EvolScreenFamiliarConfig
