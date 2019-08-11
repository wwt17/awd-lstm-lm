import torch
import torch.nn.functional as F
import math
import texar as tx


def map_structure(f, *s):
    if isinstance(s[0], (list, tuple)):
        return type(s[0])(map_structure(f, *c) for c in zip(*s))
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


def get_model_fn(model):
    def model_fn(data):
        output_layer = model.decoder._output_layer
        model.decoder._output_layer = tx.core.layers.Identity()
        output = model(
            decoding_strategy='train_greedy',
            inputs=data.transpose(0, 1))
        model.decoder._output_layer = output_layer
        return output.raw_output.transpose(0, 1)
    return model_fn


def get_embedder(model, is_GPT2):
    return model.word_embedder if is_GPT2 else model.encoder


def get_output_layer(model, is_GPT2):
    return model.decoder.output_layer if is_GPT2 else model.decoder


def get_criterion_fn(model, criterion, is_GPT2):
    output_layer = get_output_layer(model, is_GPT2)
    def criterion_fn(output, targets):
        return criterion(output_layer.weight, output_layer.bias, output.reshape(-1, output.size(-1)), targets.reshape(-1))
    return criterion_fn


def get_perplexities_entropies(logits, target):
    log_softmaxed = F.log_softmax(logits, -1)
    softmaxed = F.softmax(logits, -1)
    perplexities = -torch.gather(log_softmaxed, -1, target.unsqueeze(-1)).squeeze(-1)
    entropies = -(softmaxed * log_softmaxed).sum(-1)
    return perplexities, entropies


def convert_data_tuple(data_tuple):
    x, y = data_tuple
    return x.transpose(0, 1), y.transpose(0, 1)
