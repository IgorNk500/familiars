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
    device: torch.device | str | None = None,
    dtype: torch.dtype | str = torch.bfloat16,
    actions: Optional[GameActions] = None,
    io_class: type[BaseIO] = FamiliarIO,
    **kwargs
):
    """**PIPELINE** allows you to quickly set up and use your familiars models.

    **!!! IT IS NOT RECOMMENDED FOR PROFESSIONAL USE! SOME IMPORTANT SETTINGS ARE MISSING!!!**

    :arg model_name: Name of the familiars model on *Hugging Face* or local path to model
    :arg device: torch.device or device name on which you plan to run the model
    :arg dtype: Data type of the model
    :arg actions: Priority GameActions class. If not stated - actions will be downloaded from Hugging Face
    :arg io_class: Custom io class"""
    raise Exception("Pipeline not ready yet")
    logger = logging.getLogger(__name__)

    if device is None:
        cur_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        cur_device = torch.device(device)
    else:
        cur_device = device
    logger.info(f"using {cur_device} device")
    logger.info(f"using {dtype} dtype")

    logger.info("initializing actions...")
    if actions is None:
        logger.info("downloading actions...")
        logger.warning("ACTIONS WILL BE DOWNLOADED IN PICKLE .bin FORMAT!!!")
        logger.warning("Load pickle files only from trusted sources! Pickle files can contain viruses!")
        actions = GameActions.from_pretrained(model_name)
        logger.info("download complete.")
    else:
        logger.info("using stated actions")

    logger.info("initializing model...")
    model = AutoFamiliarModel.from_pretrained(
        model_name, device_map="auto"
    )
    model = model.to(device=cur_device, dtype=dtype)

    logger.info("initializing IO...")
    io = io_class(
        model=model,
        actions=actions
    )
