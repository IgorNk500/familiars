"""Different sizes of familiars models

+ *tiny*: ~10M
+ *small*: ~100M
+ *medium*: ~4B
+ *big*: ~7B
+ *very_big*: ~10B"""

DATA_FAMILIAR_CONFIGS = { # Data Familiar Configs
    "tiny": {
        "state_dim": 10,
        "num_actions": 5,
        "num_args": 2,
        "seq_len": 4,
        "d_model": 512,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 2048,
        "hidden_dim": 512
    },
    "small": {
        "state_dim": 10,
        "num_actions": 5,
        "num_args": 2,
        "seq_len": 4,
        "d_model": 1024,
        "nhead": 16,
        "num_layers": 8,
        "dim_feedforward": 4096,
        "hidden_dim": 1024
    },
    "medium": {
        "state_dim": 10,
        "num_actions": 5,
        "num_args": 2,
        "seq_len": 4,
        "d_model": 4096,
        "nhead": 64,  # 4096 / 64 = 64
        "num_layers": 20,
        "dim_feedforward": 16384,
        "hidden_dim": 4096
    },
    "big": {
        "state_dim": 10,
        "num_actions": 5,
        "num_args": 2,
        "seq_len": 4,
        "d_model": 4992,
        "nhead": 64,  # 4992 / 64 = 78
        "num_layers": 24,
        "dim_feedforward": 19968,
        "hidden_dim": 4992
    },
    "very_big": {
        "state_dim": 10,
        "num_actions": 5,
        "num_args": 2,
        "seq_len": 4,
        "d_model": 5504,
        "nhead": 64,  # 5504 / 64 = 86
        "num_layers": 28,
        "dim_feedforward": 22016,
        "hidden_dim": 5504
    }
}

SCREEN_FAMILIAR_CONFIGS = { # Screen Familiar Configs
    "tiny": {
        "num_actions": 5,
        "num_args": 2,
        "hidden_dim": 256,            # Head
        "vit_name": None,             # Custom ViT
        "vit_hidden_size": 256,
        "vit_num_hidden_layers": 12,
        "vit_num_attention_heads": 8,
        "vit_intermediate_size": 1024
    },
    "small": {
        "num_actions": 5,
        "num_args": 2,
        "hidden_dim": 512,
        "vit_name": None,
        "vit_hidden_size": 512,
        "vit_num_hidden_layers": 32,
        "vit_num_attention_heads": 16,
        "vit_intermediate_size": 2048
    },
    "medium": {
        "num_actions": 5,
        "num_args": 2,
        "hidden_dim": 3200,
        "vit_name": None,
        "vit_hidden_size": 3200,
        "vit_num_hidden_layers": 32,
        "vit_num_attention_heads": 64,  # 3200 / 64 = 50
        "vit_intermediate_size": 12800
    },
    "big": {
        "num_actions": 5,
        "num_args": 2,
        "hidden_dim": 3840,
        "vit_name": None,
        "vit_hidden_size": 3840,
        "vit_num_hidden_layers": 40,
        "vit_num_attention_heads": 64,  # 3840 / 64 = 60
        "vit_intermediate_size": 15360
    },
    "very_big": {
        "num_actions": 5,
        "num_args": 2,
        "hidden_dim": 4096,
        "vit_name": None,
        "vit_hidden_size": 4096,
        "vit_num_hidden_layers": 50,
        "vit_num_attention_heads": 64,  # 4096 / 64 = 64
        "vit_intermediate_size": 16384
    }
}