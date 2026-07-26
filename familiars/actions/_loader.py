"""Adds class methods *'from_pretrained'* and *'from_dict'* in GameActions class.

The loader users the *GameActionsDC* dataclass to save/load *GameActions* as a pickle file\n
**!!!WARNING!!! Load pickle file only from trusted sources! Pickle file can contain viruses! !!!WARNING!!!**"""
from dataclasses import dataclass, field
import os, tqdm
import pickle as p
from pickle import UnpicklingError

import huggingface_hub as hf

from ._abc import GameAction, GameTrigger

STANDARD_GADC_FILENAME = "actions.bin"

@dataclass(repr=False)
class GameActionsDC:  # DC - DataClass
    mactions: list[GameAction] = field(default_factory=list)
    aliases: dict[GameTrigger | str, GameAction | str] = field(default_factory=dict)

    def to_pickle(self, fp: str):
        with open(fp, "wb") as f:
            p.dump(self, f)


def load_from_pickle(fp: str):
    with open(fp, "rb") as f:
        dc = p.load(f)
    if not isinstance(dc, GameActionsDC):
        raise p.UnpicklingError("The loaded object is not a GameActionsDC")
    return dc

def save_to_pickle(dc: GameActionsDC, fp: str): dc.to_pickle(fp)



def _load_from_dir(path: str):
    path1 = os.path.join(path, STANDARD_GADC_FILENAME)
    if os.path.isfile(path1):
        return load_from_pickle(path1)
    else: raise FileNotFoundError("%s file not found in %s" % (STANDARD_GADC_FILENAME, path))

def _load_from_hub(name: str, *args, **kwargs):
    path = hf.hf_hub_download(
        repo_id=name,
        filename=STANDARD_GADC_FILENAME,
        *args,
        **kwargs
    )
    return load_from_pickle(path)

def load_from_pretrained(name: str, *args, **kwargs):
    path = os.path.join(os.getcwd(), name)
    if os.path.isdir(path):
        # Load from the dir
        return _load_from_dir(path)
    else:
        # Load from the hub
        return  _load_from_hub(name, *args, **kwargs)

def save_to_pretrained(dc: GameActionsDC, path: str):
    path = os.path.join(path, STANDARD_GADC_FILENAME)
    save_to_pickle(dc, path)

#####  PUSHING TO HUB  #####