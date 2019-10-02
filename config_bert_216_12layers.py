pretrained_model_name = None

dim = 216
hidden_size = dim

embed = {
    "dim": dim,
}

segment_embed = {
    "dim": dim
}

type_vocab_size = 1

position_embed = {
    'dim': dim
}
position_size = 1024

encoder = {
    "dim": dim,
    "num_blocks": 12,
    "use_bert_config": False,
    'embedding_dropout': 0.0,
    'residual_dropout': 0.0,
    "multihead_attention": {
        "use_bias": True,
        "num_units": dim,
        "num_heads": 12,
        "dropout_rate": 0.0,
        "output_dim": dim,
    },
    "poswise_feedforward": {
        "layers": [
            {
                "type": "Linear",
                "kwargs": {
                    "in_features": dim,
                    "out_features": dim * 4,
                    "bias": True,
                }
            },
            {
                "type": "BertGELU",
                "kwargs": {
                }
            },
            {
                "type": "Linear",
                "kwargs": {
                    "in_features": dim * 4,
                    "out_features": dim,
                    "bias": True,
                }
            }
        ],
    },
}

initializer = None
