"""Familiars Tools. EvolTrainer and EvolManager.

**This module is a very important part of familiars.**
It creates evolutionary neural networks compatible with the *Trainer API* and *GameActions*.

Copyright (c) 2026 IgorNk500"""
__all__ = ["EvolTrainer", "EvolManager"]

from .trainer import EvolTrainer
from .manager import EvolManager