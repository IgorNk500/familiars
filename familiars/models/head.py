"""Familiars Models. GameActionHead.

Copyright (c) 2026 IgorNk500"""
__all__ = ["GameActionHead"]

import torch
import torch.nn as nn


class GameActionHead(nn.Module):
    """Familiars' Head Model for issuing a discrete action and several continuous arguments"""
    def __init__(self, input_dim: int, num_actions: int, num_args: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        self.num_args = num_args
        self.args = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_args),
            nn.Tanh()
        )

    def forward(self, features: torch.Tensor):
        action_logits = self.actor(features)
        args = self.args(features)
        return action_logits, args
