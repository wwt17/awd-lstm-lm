import torch
import torch.nn as nn
import torch.nn.functional as F

import texar as tx
from texar.modules import TransformerEncoder, SinusoidsPositionEmbedder

from weight_drop import WeightDrop

import copy


def get_output_layer(output_layer_type, output_size, sequence_length, hidden_size):
    if output_layer_type == 'fc':
        return nn.Linear(sequence_length * hidden_size, output_size)
    else:
        raise Exception('output_layer_type={} is not supported'.format(output_layer_type))

def perform_output_layer(h, output_layer_type, output_layer):
    if output_layer_type == 'fc':
        return output_layer(h.reshape((h.size(0), -1)))


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


class CNN_Approximator(nn.Module):
    def __init__(
            self, sequence_length, input_size, n_layers, channels, kernel_size,
            output_size, variational=False, padding=True, residual=False,
            activation=F.relu, output_layer_type='fc',
            input_dropout=0., hidden_dropout=0., output_dropout=0.):
        super(CNN_Approximator, self).__init__()
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout else None
        self.n_layers = n_layers
        self.activation = activation
        if isinstance(channels, int):
            self.channels = [channels] * n_layers
        else:
            self.channels = copy.copy(channels)
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * n_layers
        else:
            self.kernel_size = copy.copy(kernel_size)
        self.padding = padding
        if self.padding:
            self.pads = [nn.ConstantPad1d((self.kernel_size[0] - 1, 0), 0.)]
        else:
            sequence_length -= self.kernel_size[0] - 1
            assert sequence_length > 0
        self.residual = residual
        self.convs = [nn.Conv1d(input_size, self.channels[0], self.kernel_size[0])]
        self.hidden_dropouts = [nn.Dropout(hidden_dropout)] if hidden_dropout else None
        for l in range(1, n_layers):
            if self.padding:
                self.pads.append(nn.ConstantPad1d((self.kernel_size[l] - 1, 0), 0.))
            else:
                sequence_length -= self.kernel_size[l] - 1
                assert sequence_length > 0
            if variational and l > 1:
                conv = self.convs[-1]
            else:
                conv = nn.Conv1d(self.channels[l-1], self.channels[l], self.kernel_size[l])
            self.convs.append(conv)
            if hidden_dropout:
                self.hidden_dropouts.append(nn.Dropout(hidden_dropout))
        if self.pads is not None:
            for l, pad in enumerate(self.pads):
                self.add_module('pad_{}'.format(l), pad)
        for l, conv in enumerate(self.convs):
            self.add_module('conv_{}'.format(l), conv)
        if self.hidden_dropouts is not None:
            for l, hidden_dropout in enumerate(self.hidden_dropouts):
                self.add_module('hidden_dropout_{}'.format(l), hidden_dropout)
        self.output_dropout = nn.Dropout(output_dropout) if output_dropout else None
        self.output_layer_type = output_layer_type
        self.output_layer = get_output_layer(
            output_layer_type, sequence_length, self.channels[-1], output_size)

    def forward(self, input): # input: (batch_size, sequence_length, embedding_size)
        h = input.transpose(1, 2) # (batch_size, embedding_size, sequence_length)
        if self.input_dropout is not None:
            h = self.input_dropout(h)
        for l in range(self.n_layers):
            h_ = h
            if self.padding:
                h_ = self.pads[l](h_)
            h_ = self.convs[l](h_)
            if self.hidden_dropouts is not None:
                h_ = self.hidden_dropouts[l](h_)
            h_ = self.activation(h_)
            if self.residual and l > 0:
                h = h + h_
            else:
                h = h_
        if self.output_dropout is not None:
            h = self.output_dropout(h)
        output = perform_output_layer(h, self.output_layer_type, self.output_layer)
        return output


class LSTM_Approximator(nn.Module):
    def __init__(
            self, sequence_length, input_size, hidden_size, n_layers,
            output_size=None, input_dropout=0., hidden_dropout=0., output_dropout=0.):
        super(LSTM_Approximator, self).__init__()
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size if output_size is not None else hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=hidden_dropout, batch_first=True)
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout else None
        self.output_dropout = nn.Dropout(output_dropout) if output_dropout else None
        self.output_layer = nn.Linear(hidden_size, output_size) if output_size else None

    def forward(self, input): # input: (batch_size, sequence_length, embedding_size)
        if self.input_dropout is not None:
            input = self.input_dropout(input)
        output, (h, c) = self.lstm(input)
        output = output[:, -1]
        if self.output_dropout is not None:
            output = self.output_dropout(output)
        if self.output_layer is not None:
            output = self.output_layer(output)
        return output


class Transformer_Approximator(nn.Module):
    def __init__(
            self, sequence_length, input_size, hidden_size, n_blocks, n_heads,
            output_size, output_layer_type='fc',
            embedding_dropout=0.1, residual_dropout=0.1, multihead_dropout=0.1):
        super(Transformer_Approximator, self).__init__()
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.output_size = output_size
        self.pos_embedder = SinusoidsPositionEmbedder(
            sequence_length,
            hparams={"dim": input_size},
        )
        self.encoder = TransformerEncoder(hparams={
            "dim": hidden_size,
            "num_blocks": n_blocks,
            "embedding_dropout": embedding_dropout,
            "residual_dropout": residual_dropout,
            "poswise_feedforward": tx.modules.encoders.transformer_encoder
                .default_transformer_poswise_net_hparams(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
            ),
            "multihead_attention": {
                "num_units": hidden_size,
                "num_heads": n_heads,
                "dropout_rate": multihead_dropout,
                "output_dim": hidden_size,
                "use_bias": False,
            },
        })
        self.output_layer_type = output_layer_type
        self.output_layer = get_output_layer(
            output_layer_type, output_size, sequence_length, hidden_size)

    def forward(self, input): # input: (batch_size, sequence_length, embedding_size)
        input = input * self.hidden_size ** .5
        seq_len = torch.full(input.size()[:1], self.sequence_length, dtype=torch.int32, device=input.device)
        pos_embeds = self.pos_embedder(sequence_length=seq_len)
        input = input + pos_embeds
        h = self.encoder(inputs=input, sequence_length=seq_len)
        output = perform_output_layer(h, self.output_layer_type, self.output_layer)
        return output
