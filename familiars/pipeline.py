"""`pipeline()` allows you to quickly set up and use your familiars models.

**!!! IT IS NOT RECOMMENDED FOR PROFESSIONAL USE! SOME IMPORTANT SETTINGS ARE MISSING!!!**"""
__all__ = ["pipeline"]

import torch, logging
from .io import FamiliarIO, BaseIO
from .models import AutoFamiliarModel
from .actions import GameActions

from typing import Optional

def pipeline(
    model_name: str,
    device: torch.device | str = "cuda",
    dtype: torch.dtype | str = torch.bfloat16,
    gactions_download: bool = True,
    gactions: Optional[GameActions] = None,
    io_class: type[BaseIO] = FamiliarIO,
    **kwargs
):
    """**PIPELINE** allows you to quickly set up and use your familiars models.

    **!!! IT IS NOT RECOMMENDED FOR PROFESSIONAL USE! SOME IMPORTANT SETTINGS ARE MISSING!!!**

    :arg model_name: Name of the familiars model on *Hugging Face* or local path to model
    :arg device: torch.device or device name on which you plan to run the model
    :arg dtype: Data type of the model
    :arg gactions_download: Do download gactions.bin from Hugging Face
    :arg gactions:
    :arg io_class:"""
    pass