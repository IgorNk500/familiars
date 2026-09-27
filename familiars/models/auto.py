"""Familiars Models. AutoFamiliarConfig and AutoFamiliarModel.

Copyright (c) 2026 IgorNk500"""
__all__ = ["AutoFamiliarModel"]
#__all__ = ["AutoFamiliarModel", "AutoFamiliarConfig"]

#from transformers import AutoConfig, AutoModel
# noinspection PyProtectedMember
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
#from transformers.models.auto.configuration_auto import _LazyConfigMapping
from collections import OrderedDict

##########  MAPPINGS  ##########

FAMILIAR_MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Normal models
        ("data_normal_familiar", "NormalDataFamiliarModel"),
        ("screen_normal_familiar", "NormalScreenFamiliarModel"),

        # Evol models
        ("data_evol_familiar", "EvolDataFamiliarModel"),
        ("screen_evol_familiar", "EvolScreenFamiliarModel")
    ]
)

FAMILIAR_CONFIG_MAPPING_NAMES = OrderedDict(
    [
        # Configs for normal models
        ("data_normal_familiar", "NormalDataFamiliarConfig"),
        ("screen_normal_familiar", "NormalScreenFamiliarConfig"),

        # Configs for evol models
        ("data_evol_familiar", "EvolDataFamiliarConfig"),
        ("screen_evol_familiar", "EvolScreenFamiliarConfig")
    ]
)

#####

FAMILIAR_MODEL_MAPPING = _LazyAutoMapping(FAMILIAR_CONFIG_MAPPING_NAMES, FAMILIAR_MODEL_MAPPING_NAMES)

#FAMILIAR_CONFIG_MAPPING = _LazyConfigMapping(FAMILIAR_CONFIG_MAPPING_NAMES)

################################

class AutoFamiliarModel(_BaseAutoModelClass):
    """This class works in the same way as `transformers.AutoModel`, but for Familiars models.

    **This class automatically detects the type of Familiars model**"""
    _model_mapping = FAMILIAR_MODEL_MAPPING

'''
class AutoFamiliarConfig(AutoConfig):
    """This class works in the same way as `transformers.AutoConfig`, but for Familiars configs.

    **This class automatically detects the type of Familiars config**"""

    # We are rewriting the static method register.
    # The code is taken from transformers.models.auto.configuration_auto
    @staticmethod
    def register(model_type, config, exist_ok=False) -> None:
        pass
'''


########## WARNING! ###############################################################################################################
# We cannot register Familiars models and configurations in the AutoModel and AutoConfig class, as they require a separate class. #
###################################################################################################################################