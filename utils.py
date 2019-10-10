import torch
import torch.nn.functional as F
import math
import texar as tx
from torch.nn.utils.rnn import PackedSequence
from typing import NamedTuple

def map_structure(f, *s):
    if isinstance(s[0], (list, tuple)) and not isinstance(s[0], PackedSequence):
        res = (map_structure(f, *c) for c in zip(*s))
        if type(s[0]) in (list, tuple):
            return type(s[0])(res)
        else:
            return type(s[0])(*res)
    else:
        return f(*s)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    return map_structure(torch.Tensor.detach, h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len]
    return data, target


def get_config_model(config_model, vocab_size):
    import importlib
    config_model = importlib.import_module(config_model)
    config_model = {
        k: v for k, v in config_model.__dict__.items()
        if not k.startswith('__')}
    try:
        config_model.pop('dim')
    except KeyError:
        pass
    config_model['vocab_size'] = vocab_size
    return config_model


def get_splits(ntokens):
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    else:
        splits = []
    print('Using', splits)
    return splits


def loss_repr(loss):
    return 'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
        loss, math.exp(loss), loss / math.log(2))


def set_all_requires_grad(params, requires_grad):
    for param in params:
        param.requires_grad = requires_grad


def cross_entropy(logits, targets):
    logits_dim = logits.dim()
    return F.cross_entropy(logits.permute(*([0, logits_dim - 1] + list(range(1, logits_dim - 1)))), targets)


def get_model_fn(model):
    def model_fn(data, batch_first=False):
        if not batch_first:
            data = data.transpose(0, 1)
        output_layer = model.decoder._output_layer
        model.decoder._output_layer = tx.core.layers.Identity()
        output = model(
            decoding_strategy='train_greedy',
            inputs=data)
        model.decoder._output_layer = output_layer
        out = output.logits
        if not batch_first:
            out = out.transpose(0, 1)
        return out
    return model_fn


def get_embedder(model, is_GPT2):
    return model.word_embedder if is_GPT2 else model.encoder

def get_embedding_weight(embedder):
    if isinstance(embedder, torch.nn.Embedding):
        weight = embedder.weight
    else:
        weight = embedder.embedding
    return weight

def get_embedding_size(embedder):
    return get_embedding_weight(embedder).size(1)


def get_output_layer(model, is_GPT2):
    return model.decoder.output_layer if is_GPT2 else model.decoder


def get_criterion_fn(model, criterion, is_GPT2):
    output_layer = get_output_layer(model, is_GPT2)
    def criterion_fn(output, targets):
        return criterion(output_layer.weight, output_layer.bias, output.reshape(-1, output.size(-1)), targets.reshape(-1))
    return criterion_fn


def get_perplexities_entropies(p, log_p, target):
    perplexities = -torch.gather(log_p, -1, target.unsqueeze(-1)).squeeze(-1)
    entropies = -(p * log_p).sum(-1)
    return perplexities, entropies


def convert_data_tuple(data_tuple):
    x, y = data_tuple
    return x.transpose(0, 1), y.transpose(0, 1)


def force_reduce_lr(lr_scheduler):
    lr_scheduler._reduce_lr(lr_scheduler.last_epoch)
    lr_scheduler.cooldown_counter = lr_scheduler.cooldown
    lr_scheduler.num_bad_epochs = 0


def set_lr(lr_scheduler, lr):
    print('set lr to {}'.format(lr))
    for param_group in lr_scheduler.optimizer.param_groups:
        param_group['lr'] = lr
