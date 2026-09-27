"""Familiars Models. AutoFamiliarConfig and AutoFamiliarModel.

Copyright (c) 2026 IgorNk500"""
__all__ = ["AutoFamiliarModel"]

import os
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping
from collections import OrderedDict

from ..models import FamiliarModel, FamiliarConfig

##########  MAPPINGS  ##########

FAMILIAR_MODEL_MAPPING_NAMES: OrderedDict = ...

FAMILIAR_CONFIG_MAPPING_NAMES: OrderedDict = ...

#####

FAMILIAR_MODEL_MAPPING: _LazyAutoMapping = ...


#FAMILIAR_CONFIG_MAPPING: _LazyAutoMapping = ...

class AutoFamiliarModel(_BaseAutoModelClass):
    """This class works in the same way as `transformers.AutoModel`, but for Familiars models.

    **This class automatically detects the type of Familiars model**"""
    _model_mapping = FAMILIAR_MODEL_MAPPING

    @classmethod
    def from_config(cls, config: FamiliarConfig, **kwargs) -> FamiliarModel:
        ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], *model_args,
                        **kwargs) -> FamiliarModel:
        ...

    ...
