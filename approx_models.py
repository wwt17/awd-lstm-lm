import torch
import torch.nn as nn
import torch.nn.functional as F

import texar as tx
from texar.torch.modules import TransformerEncoder, SinusoidsPositionEmbedder

from locked_dropout import VariationalDropout
from weight_drop import WeightDrop
from utils import get_model_fn
from texar.torch.modules import GPT2Decoder

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
            output_size, variational=False, padding=True, skip_link=None,
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
        self.skip_link = skip_link
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
            if self.skip_link == 'res' and l > 0:
                h = h + h_
            else:
                h = h_
        if self.output_dropout is not None:
            h = self.output_dropout(h)
        output = perform_output_layer(h, self.output_layer_type, self.output_layer)
        return output


class LSTM_Approximator(nn.Module):
    def __init__(
            self, input_size, hidden_size, n_layers, explicit_stack=False, skip_link=None, normalization=None,
            output_size=None, input_transform=False,
            input_dropout=0., hidden_dropout=0., output_dropout=0.):
        super(LSTM_Approximator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.skip_link = skip_link
        if skip_link:
            assert explicit_stack
        self.explicit_stack = explicit_stack
        self.input_transform = input_transform
        self.output_size = output_size if output_size is not None else hidden_size
        if explicit_stack:
            assert self.skip_link is None or input_size == hidden_size or self.input_transform
            self.lstm = [nn.LSTM(input_size if l == 0 and not self.input_transform else hidden_size, hidden_size, batch_first=True) for l in range(n_layers)]
            self.lstm = nn.ModuleList(self.lstm)
            if normalization:
                self.normalizations = [nn.LayerNorm(hidden_size) for l in range(n_layers-1)]
                self.normalizations = nn.ModuleList(self.normalizations)
            self.hidden_dropout = VariationalDropout(hidden_dropout) if hidden_dropout else None
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, n_layers, dropout=hidden_dropout, batch_first=True)
        if self.input_transform:
            self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_dropout = VariationalDropout(input_dropout) if input_dropout else None
        self.output_dropout = VariationalDropout(output_dropout) if output_dropout else None
        self.output_layer = nn.Linear(hidden_size, output_size) if output_size else None

    def forward(self, input): # input: (batch_size, sequence_length, embedding_size)
        if self.input_dropout is not None:
            input = self.input_dropout(input)
        if hasattr(self, 'input_layer'):
            input = self.input_layer(input)
        if isinstance(self.lstm, nn.ModuleList):
            x = input
            for l, rnn in enumerate(self.lstm):
                out, _ = rnn(x)
                if hasattr(self, 'normalizations') and l != len(self.lstm)-1:
                    out = self.normalizations[l](out)
                if self.hidden_dropout is not None and l != len(self.lstm)-1:
                    out = self.hidden_dropout(out)
                if self.skip_link == 'res':
                    x = x + out
                else:
                    x = out
            output = x
        else:
            output, (h, c) = self.lstm(input)
        if self.output_dropout is not None:
            output = self.output_dropout(output)
        if self.output_layer is not None:
            output = self.output_layer(output)
        return output


class Transformer_Approximator(nn.Module):
    def __init__(self, hparams, output_size=None):
        super(Transformer_Approximator, self).__init__()
        self.model = GPT2Decoder(hparams=hparams)
        self.model.word_embedder = tx.torch.core.layers.Identity()
        self.model._output_layer = tx.torch.core.layers.Identity() if output_size is None else nn.Linear(hparams['dim'], output_size)

    def forward(self, input): # input: (batch_size, sequence_length, embedding_size)
        model_fn = get_model_fn(self.model)
        output = model_fn(input, batch_first=True)
        output = self.model.output_layer(output)
        return output
