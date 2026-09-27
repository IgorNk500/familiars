"""Adds class methods *'from_pretrained'* and *'from_dict'* in GameActions class.

The loader users the *GameActionsDC* dataclass to save/load *GameActions* as a pickle file\n
**!!!WARNING!!! Load pickle files only from trusted sources! Pickle files can contain viruses! !!!WARNING!!!**"""
from dataclasses import dataclass, field
import os
import pickle as p

import huggingface_hub as hf

from ._abc import GameAction, GameTrigger

STANDARD_GADC_FILENAME = "actions.bin"

@dataclass(repr=False)
class GameActionsDC:  # DC - DataClass
    mactions: list[GameAction] = field(default_factory=list)
    aliases: dict[GameTrigger | str, GameAction | str] = field(default_factory=dict)


    @classmethod
    def from_pickle(cls, fp: str):
        with open(fp, "rb") as f:
            dc = p.load(f)
        if not isinstance(dc, GameActionsDC):
            raise p.UnpicklingError("The loaded object is not a GameActionsDC")
        return dc

    def to_pickle(self, fp: str):
        with open(fp, "wb") as f:
            p.dump(self, f)

    @classmethod
    def _from_dir(cls, path: str):
        path1 = os.path.join(path, STANDARD_GADC_FILENAME)
        if os.path.isfile(path1):
            return cls.from_pickle(path1)
        else:
            raise FileNotFoundError("%s file not found in %s" % (STANDARD_GADC_FILENAME, path))

    @classmethod
    def _from_hub(cls, name: str, *args, **kwargs):
        path = hf.hf_hub_download(
            repo_id=name,
            filename=STANDARD_GADC_FILENAME,
            *args,
            **kwargs
        )
        return cls.from_pickle(path)

    @classmethod
    def from_pretrained(cls, path: str, *args, **kwargs):
        if os.path.isdir(path):
            # Load from the dir
            return cls._from_dir(path)
        else:
            # Load from the hub
            return cls._from_hub(path, *args, **kwargs)

    def to_pretrained(self, path: str):
        path = os.path.join(path, STANDARD_GADC_FILENAME)
        self.to_pickle(path)

#####  PUSHING TO HUB  #####
# Not ready yet