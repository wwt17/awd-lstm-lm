import torch
import math


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


def loss_repr(loss):
    return 'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
        loss, math.exp(loss), loss / math.log(2))
