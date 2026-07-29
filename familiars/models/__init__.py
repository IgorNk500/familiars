"""Familiars Models. NormalDataFamiliarConfig and NormalDataFamiliarModel; NormalScreenFamiliarConfig and NormalScreenFamiliarModel; EvolDataFamiliarConfig and EvolDataFamiliarModel; EvolScreenFamiliarConfig and EvolScreenFamiliarModel;
AutoFamiliarModel;
DATA_FAMILIAR_CONFIGS; SCREEN_FAMILIAR_CONFIGS.

Copyright (c) 2026 IgorNk500"""
__all__ = ["FamiliarModel", "FamiliarConfig",
           "NormalDataFamiliarModel", "NormalDataFamiliarConfig",
           "NormalScreenFamiliarModel", "NormalScreenFamiliarConfig",
           "EvolDataFamiliarModel", "EvolDataFamiliarConfig",
           "EvolScreenFamiliarModel", "EvolScreenFamiliarConfig",
           "AutoFamiliarModel",
           "DATA_FAMILIAR_CONFIGS", "SCREEN_FAMILIAR_CONFIGS"]

import torch, abc
import torch.nn as nn
from numpy import ceil as np_ceil
from transformers import (PreTrainedModel, PreTrainedConfig,
                          ViTConfig, ViTModel)   # Screen Familiar uses ViT

from typing import Literal, Optional

from .auto import AutoFamiliarModel
from .head import GameActionHead
from .std_configs import DATA_FAMILIAR_CONFIGS, SCREEN_FAMILIAR_CONFIGS

##########  ABSTRACT CLASSES  ##########

class FamiliarModel(PreTrainedModel, abc.ABC):
    """Familiar Model"""
    is_evol = False

    @abc.abstractmethod
    def generate(self, *args, **kwargs): pass

    @abc.abstractmethod
    def forward(self, state_seq: torch.Tensor, return_dict: bool = False): pass

    __call__ = generate

class FamiliarConfig(PreTrainedConfig, abc.ABC):
    """Familiar Config"""
    num_actions: int = 5  # Example num of actions
    num_args: int = 2  # Standard num of arguments (X and Y)
    hidden_dim: int = 256


########################################

class NormalDataFamiliarConfig(FamiliarConfig):
    """Configuration for **NormalDataFamiliarModel.**

    **Read more in the docs.**"""

    model_type = "data_normal_familiar"

    state_dim: int = 10
    seq_len: int = 4
    d_model: int = 128
    nhead: int = 4
    dim_feedforward: int = 2048
    num_layers: int = 2  # Example num of layers


class NormalDataFamiliarModel(FamiliarModel):
    """**A Familiars' Model for processing game data and transforming it into action in a game.**

    The game states *(distances to enemies, coordinates, speed, etc.)* are sent to the input *(forward function)* in **torch.Tensor** format. In the case of the more convenient generate function, an array consisting of game states is provided as input.

    At the output *(generate function)*, the model returns:\n
    + Action number *(from 0 to num_actions; float)*
    + List of arguments *(in the number of num_args)*

    **You can specify a custom number of actions and args when creating the model (num_actions, num_args).**"""

    config_class = NormalDataFamiliarConfig

    def __init__(self, config: config_class | Literal["tiny", "small", "medium", "big", "very_big"],
                 num_actions: Optional[int], num_args: Optional[int]):
        if isinstance(config, str):
            # Convert size to model config
            if not config in DATA_FAMILIAR_CONFIGS:
                raise ValueError(
                    f"'{config}' not found in Data familiar configs. Use custom config or set config to 'tiny', 'small', 'medium', 'big' or 'very_big'.\n Read more in the docs.")
            config = self.config_class.from_dict(DATA_FAMILIAR_CONFIGS[config])

        if num_actions: config.num_actions = num_actions
        if num_args: config.num_args = num_args

        super().__init__(config)
        self.state_proj = nn.Linear(config.state_dim, config.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, config.seq_len, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = GameActionHead(config.d_model, config.num_actions, config.num_args, config.hidden_dim)
        self.init_weights()

    def forward(self, state_seq: torch.Tensor, return_dict: bool = False):
        """**Low-level forward function**

        The game states *(distances to enemies, coordinates, speed, etc.)* are sent to the input *(forward function)* in **torch.Tensor** format.

        At the output, the model returns:\n
        + Action number *(from 0 to num_actions; float)*
        + Arguments of action *(in the number of num_args)*

        **It is recommended to use generate instead of forward**
        """
        # state_seq: (batch, seq_len, state_dim)
        x = self.state_proj(state_seq) + self.pos_embed[:, :state_seq.size(1), :]
        x = self.transformer(x)
        features = x[:, -1, :]  # The last time step
        action_logits, args = self.head(features)
        if return_dict:
            return {"action_logits": action_logits, "args": args}
        return action_logits, args

    def generate(self, states: list | torch.Tensor, return_dict: bool = False) -> dict | tuple[int, list]:
        """**A more convenient generation function.**

        It automatically converts *list* to *torch.Tensor*, and also automatically converts logits into probabilities."""

        # Convert list to torch.Tensor
        if isinstance(states, list):
            states = torch.tensor(states, self.dtype, self.device)

        # Run forward
        action_logits, args = self.forward(states)

        # Convert logits into probabilities
        probs = torch.softmax(action_logits, dim=1)  # [1, num_actions]
        action = torch.multinomial(probs, 1).item()  # The number from 0 to num_actions-1

        action = np_ceil(action)

        # Convert args
        args_norm = args[0].tolist()

        if return_dict:
            return {"action": action, "args": args}

        return action, args_norm

    #__call__ = generate


########################################################################################################################

class NormalScreenFamiliarConfig(FamiliarConfig):
    """Configuration for **NormalScreenFamiliarModel.**\n
    **Screen familiar uses ViT Models.**
    Set the model name to *vit_name* or leave it blank to use an empty ViT model.

    **Read more in the docs.**"""

    model_type = "screen_normal_familiar"

    '''
    num_actions: int = 5  # Example num of actions
    num_args: int = 2  # Standard num of arguments (X and Y)
    hidden_dim: int = 256
    '''

    # Example for tiny (~10M) model
    vit_name: Optional[str] = None
    vit_hidden_size: int = 256
    vit_num_hidden_layers: int = 12
    vit_num_attention_heads: int = 8
    vit_intermediate_size: int = 1024

    vit_image_size: int = 256
    vit_patch_size: int = 16


class NormalScreenFamiliarModel(FamiliarModel):
    """A Familiars' Model for processing game screen and transforming it into action in a game.**

    **!!!WARNING!!! For use this model, you need install `screen-familiars` library! Standard FamiliarIO does not support ScreenFamiliar! Use screen-familiars.ScreenIO !!!WARNING!!!**

    The pixel values are sent to the input *(forward function)* in **torch.Tensor** format.

    At the output *(generate function)*, the model returns:\n
    + Action number *(from 0 to num_actions; float)*
    + List of arguments *(in the number of num_args)*

    **You can specify a custom number of actions and args when creating the model (num_actions, num_args).**"""

    config_class = NormalScreenFamiliarConfig

    def __init__(self, config: config_class | Literal["tiny", "small", "medium", "big", "very_big"],
                 num_actions: Optional[int], num_args: Optional[int], vit_name: Optional[str]):
        if isinstance(config, str):
            # Convert size to model config
            if not config in SCREEN_FAMILIAR_CONFIGS:
                raise ValueError(
                    f"'{config}' not found in Screen familiar configs. Use custom config or set config to 'tiny', 'small', 'medium', 'big' or 'very_big'.\n Read more in the docs.")

            config = self.config_class.from_dict(SCREEN_FAMILIAR_CONFIGS[config])

        if num_actions: config.num_actions = num_actions
        if num_args: config.num_args = num_args
        if vit_name: config.vit_name = vit_name

        super().__init__(config)

        if config.vit_name:
            print("{} will be downloaded for NormalScreenFamiliarModel.".format(config.vit_name))
            self.vit = ViTModel.from_pretrained(config.vit_name)
        else:
            # Custom ViT
            vit_config = ViTConfig(
                hidden_size=config.vit_hidden_size,
                num_hidden_layers=config.vit_num_hidden_layers,
                num_attention_heads=config.vit_num_attention_heads,
                intermediate_size=config.vit_intermediate_size,
                image_size=config.vit_image_size,
                patch_size=config.vit_patch_size,
                num_channels=3
            )
            self.vit = ViTModel(vit_config)

        self.head = GameActionHead(self.vit.config.hidden_size, config.num_actions, config.hidden_dim)
        self.init_weights()

    def forward(self, pixel_values: torch.Tensor, return_dict: bool = False):
        """**Low-level forward function**

        The pixel values are sent to the input in **torch.Tensor** format.

        At the output, the model returns:\n
        + Action number *(from 0 to num_actions; float)*
        + Arguments of action *(in the number of num_args)*

        **It is recommended to use generate instead of forward**"""
        # pixel_values: (batch, channels, height, width)
        outputs = self.vit(pixel_values=pixel_values)
        features = outputs.last_hidden_state[:, 0, :]  # CLS token
        action_logits, args = self.head(features)
        if return_dict:
            return {"action_logits": action_logits, "args": args}
        return action_logits, args

    def generate(self, pixel_values: torch.Tensor, return_dict: bool = False):
        """**A more convenient generation function.**

         It automatically converts logits into probabilities.

         **RECOMMENDED TO USE WITH screen-familiars.ScreenIO**"""
        # Run forward
        action_logits, args = self.forward(pixel_values)

        # Convert logits into probabilities
        probs = torch.softmax(action_logits, dim=1)  # [1, num_actions]
        action = torch.multinomial(probs, 1).item()  # The number from 0 to num_actions-1

        action = np_ceil(action)

        # Convert args
        args_norm = args[0].tolist()

        if return_dict:
            return {"action": action, "args": args}

        return action, args_norm

#####  EVOL FAMILIAR MODEL  #####

class EvolDataFamiliarConfig(NormalDataFamiliarModel):
    """Configuration for **EvolDataFamiliarModel**.

    **Read more in the docs.**"""

    model_type = "data_evol_familiar"


class EvolDataFamiliarModel(NormalDataFamiliarModel):
    """**A Familiars' Model for processing game data and transforming it into action in a game.**

    The game states *(distances to enemies, coordinates, speed, etc.)* are sent to the input *(forward function)* in **torch.Tensor** format. In the case of the more convenient generate function, an array consisting of game states is provided as input.

    At the output *(generate function)*, the model returns:\n
    + Action number *(from 0 to num_actions; float)*
    + List of arguments *(in the number of num_args)*

    **You can specify a custom number of actions and args when creating the model (num_actions, num_args).**

    ======

    **This is an evolutionary model.**
    This means that it can learn on its own and does not need training data.

    **Use it with `evol.EvolTrainer`.**"""

    config_class = EvolDataFamiliarConfig
    is_evol = True

##########

class EvolScreenFamiliarConfig(NormalScreenFamiliarConfig):
    """Configuration for **EvolScreenFamiliarModel.**\n
    **Screen familiar uses ViT Models.**
    Set the model name to *vit_name* or leave it blank to use an empty ViT model.

    **Read more in the docs.**"""

    model_type = "screen_evol_familiar"

class EvolScreenFamiliarModel(NormalScreenFamiliarModel):
    """A Familiars' Model for processing game screen and transforming it into action in a game.**

    **!!!WARNING!!! For use this model, you need install `screen-familiars` library! Standard FamiliarIO does not support ScreenFamiliar! Use screen-familiars.ScreenIO !!!WARNING!!!**

    The pixel values are sent to the input *(forward function)* in **torch.Tensor** format.

    At the output *(generate function)*, the model returns:\n
    + Action number *(from 0 to num_actions; float)*
    + List of arguments *(in the number of num_args)*

    **You can specify a custom number of actions and args when creating the model (num_actions, num_args).**

    ======

    **This is an evolutionary model.**
    This means that it can learn on its own and does not need training data.

    **Use it with `evol.EvolTrainer`.**"""

    config_class = EvolScreenFamiliarConfig
    is_evol = True