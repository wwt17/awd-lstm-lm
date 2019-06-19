import torch
import torch.nn as nn

from weight_drop import WeightDrop


class MLP_Approximator(nn.Module):
    """Simple MLP model approximating hidden states of a trained model."""

    def __init__(
            self, context_size, input_size, hidden_size, output_size,
            input_dropout=0., hidden_dropout=0., output_dropout=0.,
            weight_dropout=0.):
        super(MLP_Approximator, self).__init__()
        layers = [
            nn.Linear(context_size * input_size, hidden_size),
            nn.Linear(hidden_size, output_size),
        ]
        if weight_dropout:
            layers = [WeightDrop(layer, ['weight'], dropout=weight_dropout)
                      for layer in layers]

        self.mlp = nn.Sequential(
            nn.Dropout(input_dropout),
            layers[0],
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            layers[1],
            nn.ReLU(),
            nn.Dropout(output_dropout),
        )

    def forward(self, input):
        return self.mlp(input.contiguous().view(input.size(0), -1))
