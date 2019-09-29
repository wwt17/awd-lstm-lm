"""Evaluate a language model on perturbed input in the restricted context setting."""

import argparse
import time, os, sys
import math
import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data
import model

from data import FixedLengthContextDataset, get_vocab_all_pos
from utils import map_structure, get_model_fn, get_criterion_fn, get_embedder, get_output_layer, get_perplexities_entropies, convert_data_tuple
import pointer
from gpt2_decoder import GPT2Decoder
# this is not a good practice, but making an exception in this case
from perturbations import *

parser = argparse.ArgumentParser(description='Test language model on perturbed inputs in the restricted context setting')
parser.add_argument('--data', type=str, default='data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='model checkpoint to use')
parser.add_argument('--approx_model', type=str, default=None,
                    help='approximator model to use')
# Consistent with Language Modeling code from Merity et al.
parser.add_argument('--cuda', action='store_false',
                    help='Using this flag turns off CUDA, default value set to True')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--logdir', type=str, default='./',
                    help='location to write per token log loss')
parser.add_argument('--stage', default='valid', choices=['train', 'valid', 'test'],
                    help='Run on valid/test set')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=300,
                    help='sequence length')
parser.add_argument('--max_seq_len', type=int, default=1000,
                    help='Maximum possible sequence length to make all experiments start at the same example i.e. skip the same number of tokens at the start')
parser.add_argument('--sample_gap', type=int, default=1)
parser.add_argument('--freq', type=int, default=108,
                    help='Frequent words cut-off, all words with corpus count > 800')
parser.add_argument('--rare', type=int, default=1986,
                    help='Rare words cut-off, all words with corpus count < 50')
parser.add_argument('--func', type=str, default='',
                    help='random_drop_many, drop_pos, keep_pos, shuffle, shuffle_within_spans, reverse, reverse_within_spans, replace_target, replace_target_with_nearby_token, drop_target, replace_with_rand_seq',
                    choices=['random_drop_many', 'drop_pos', 'keep_pos', 'shuffle', 'shuffle_within_spans', 'reverse', 'reverse_within_spans', 'replace_target', 'replace_target_with_nearby_token', 'drop_target', 'replace_with_rand_seq'])
parser.add_argument('--span', type=int, default=20,
                    help='For shuffle and reverse within spans')
parser.add_argument('--drop_or_replace_target_window', type=int, default=300,
                    help='window for drop or replace target experiments')
parser.add_argument('--n', type=float,
                    help='Fraction of tokens to drop, between 0 and 1')
# Specify a list
parser.add_argument('--pos', action='append', default=None,
                    help='Pos tags to drop. Sample usage is --pos NN --pos VB --pos JJ')
parser.add_argument('--use_range', nargs='+', type=lambda s: list(map(int, s)),
                    default=[1, 5, 10, 15, 20, 30, 50, 100, 200],
                    help='Use these values for the boundary loop, but first convert to ints. Sample usage is --use_range 5 20 100')
parser.add_argument('--entropy', action='store_true',
                    help='Get entropy')
parser.add_argument('--pointer', action='store_true',
                    help='add cache pointer')
parser.add_argument('--theta', type=float, default=0.6625523432485668)
parser.add_argument('--lambdasm', type=float, default=0.12785920428335693)
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=4,
                    help='Number of workers to load dataset')
args = parser.parse_args()
print(args)

# Set the random seed manually for reproducibility.
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run without --cuda')
    else:
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)

if args.checkpoint is not None:
    print('Load model from {}'.format(args.checkpoint))
    model, criterion, optimizer = torch.load(args.checkpoint)
else:
    model = None
if args.approx_model is not None:
    print('Loading approximator model ...')
    loaded = torch.load(args.approx_model)
    approx_model = loaded[0]
else:
    approx_model = None

is_GPT2 = isinstance(model, GPT2Decoder)

device = torch.device('cuda' if args.cuda else 'cpu')
if model is not None:
    model.to(device)
if approx_model is not None:
    approx_model.to(device)

if is_GPT2:
    model_fn = get_model_fn(model)

embedder = get_embedder(model, is_GPT2)
output_layer = get_output_layer(model, is_GPT2)
criterion_fn = get_criterion_fn(model, criterion, is_GPT2)

corpus = data.prepare_corpus(args.data, data.get_holistic_text)

pos_datafile = os.path.dirname(args.data if args.data.endswith('/') else args.data+'/')+'_pos/'
print('try POS data file: {}'.format(pos_datafile))
if os.path.exists(pos_datafile):
    pos_corpus = data.prepare_corpus(pos_datafile, data.get_holistic_text, with_pos=True)
    print('Built pos corpus')
else:
    print('POS file does not exist.')
    pos_corpus = None

print('On {} set!!!'.format(args.stage))
data_ = getattr(corpus, args.stage)
if pos_corpus is not None:
    try:
        pos_data = getattr(pos_corpus, args.stage)
    except AttributeError:
        pos_data = None
    pos_dict = pos_corpus.vocab
else:
    pos_data = None
    pos_dict = None

print('Made batches.')

def evaluate(data_source, pdata_source, perturb_fn, args):
    use_pointer = args.pointer
    if use_pointer:
        theta = args.theta
        lambdah = args.lambdasm
    eval_entropy = args.entropy or use_pointer
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if approx_model is not None:
        approx_model.eval()

    n = 0
    total_loss, total_entropy = 0., 0.
    all_losses, all_entropies, all_topk = [], [], []

    # Number of batches excludes the last seq_len tokens as start of context, since the target index would lie out of bounds.
    # Skip examples at the beginning to ensure the first target is the same for each experiment.
    # All experiments then operate on the same examples.
    # Select the max_seq_len as the largest context size you'd like to test.
    examples_to_ignore = args.max_seq_len - args.seq_len
    print('Number of examples to ignore: {}'.format(examples_to_ignore))
    data_source = FixedLengthContextDataset(data_source[examples_to_ignore:], args.seq_len)
    n_ = len(data_source)
    sample_indices = list(range(-1, n_, args.sample_gap))
    if sample_indices[-1] != n_ - 1:
        sample_indices.append(n_ - 1)
    sample_indices_p = 0
    data_source = torch.utils.data.Subset(data_source, sample_indices[1:])
    if pdata_source is not None:
        pdata_source = FixedLengthContextDataset(pdata_source[examples_to_ignore:], args.seq_len)
        assert len(pdata_source) == n_
        pdata_source = torch.utils.data.Subset(pdata_source, sample_indices[1:])

    with torch.no_grad():
        data_loader = DataLoader(
            data_source,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        )
        pdata_loader = DataLoader(
            pdata_source,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        ) if pdata_source is not None else None
        t = tqdm(data_loader)
        for data_item in (t if pdata_loader is None else zip(t, pdata_loader)):
            if args.cuda:
                data_item = map_structure(lambda t: t.cuda(), data_item)
            if pdata_loader is None:
                data_tuple = data_item
            else:
                data_tuple, pdata_tuple = data_item
            data, targets = convert_data_tuple(data_tuple)
            if n == 0:
                print('First word: {}'.format(corpus.vocab.id_to_token_map_py[int(data[-1, 0])]))
            pdata, ptargets = convert_data_tuple(pdata_tuple) if pdata_loader is not None else (None, None)

            batch_size = data.size(1)

            gaps = []
            for i in range(batch_size):
                gaps.append(sample_indices[sample_indices_p + 1] - sample_indices[sample_indices_p])
                sample_indices_p += 1
            max_gap = max(gaps)

            data = perturb_fn(data, pdata, targets, ptargets, args)

            if approx_model is not None:
                output = approx_model(embedder(data.transpose(0, 1))).transpose(0, 1)
            elif is_GPT2:
                output = model_fn(data)
            else:
                # when using your own LM, ensure the init hidden function exists!
                init_hidden = model.init_hidden(batch_size)
                if use_pointer:
                    output, _, rnn_outs, _ = model(data, init_hidden, return_h=True)
                    rnn_out = rnn_outs[-1]
                else:
                    output, _ = model(data, init_hidden)
            output_ = output[-max_gap:]
            targets_ = targets[-max_gap:]
            if eval_entropy:
                logits = output_layer(output_)
                p = logits.softmax(-1)
                if use_pointer:
                    init_history = pointer.init_history(batch_size, p.size(-1), rnn_out.size(-1), device=p.device)
                    ptr_p, _ = pointer.get_p(targets, rnn_out, data.size(0), theta, init_history)
                    p = lambdah * ptr_p[-max_gap:] + (1. - lambdah) * p
                    log_p = p.log()
                else:
                    log_p = logits.log_softmax(-1)
                losses_batch, entropies_batch = get_perplexities_entropies(p, log_p, targets_)
                topk_batch = p.topk(args.topk, dim=-1)
                topk_batch = map_structure(lambda t: t.cpu().numpy(), topk_batch)
                for i, gap in enumerate(gaps):
                    losses, entropies, topk = map_structure(lambda t: t[-gap:, i], (losses_batch, entropies_batch, topk_batch))
                    all_losses += losses.tolist()
                    all_entropies += entropies.tolist()
                    all_topk.extend(zip(*topk))
                    total_loss += losses.sum().item()
                    total_entropy += entropies.sum().item()
                loss = losses.mean()
                entropy = entropies.mean()
            else:
                loss = criterion_fn(output_, targets_)
                entropy = 0.
                total_loss += batch_size * loss.item()
            n += sum(gaps)
            t.set_postfix_str('loss={:9.6f} ppl={:11.3f} entropy={:7.5f}'.format(loss, math.exp(loss), entropy))

        print('Last word: {}'.format(corpus.vocab.id_to_token_map_py[int(data[-1, -1])]))
        print('Total: {}'.format(n))

    return total_loss / n, total_entropy / n, all_losses, all_entropies, all_topk

def print_results(prefix, loss, entropy):
    print('{}: loss: {}, ppl: {}, entropy: {}'.format(prefix, loss, math.exp(loss), entropy))

def save_results(file_name, items):
    with open(os.path.join(args.logdir, file_name), 'wb') as f:
        for item in items:
            pickle.dump(item, f)

if not args.func:
    loss, entropy, all_losses, all_entropies, all_topk = evaluate(
        data_, pos_data,
        lambda data, pdata, targets, ptargets, args: data,
        args)
    print_results('seq_len={}'.format(args.seq_len), loss, entropy)
    save_results(
        'per_token_scores.{}.none.{}'.format(args.stage, args.seq_len),
        [args.seq_len, loss, entropy, all_losses, all_entropies, all_topk],
    )

    sys.exit(0)

def perturb_data(data, pdata, boundary, func, args, pos_dict, targets, ptargets, pos2vocab):
    if 'pos' in args.func:
        ret = func(data, pdata, boundary, args, pos_dict)
    elif args.func in ['replace_target', 'drop_target', 'replace_target_with_nearby_token']:
        ret = func(data, ptargets, args, targets, pos2vocab, corpus, pos_dict)
    elif args.func == 'replace_with_rand_seq':
        ret = func(data, boundary, args, corpus)
    else:
        ret = func(data, boundary, args)
    return ret


# The length of each sequence is altered differently, so examples cannot be batched.
if args.func in ['drop_pos', 'keep_pos', 'drop_target']:
    assert args.batch_size == 1

# Running with different boundary values to extract a trend.
loop_range = args.use_range

# For logging per token scores
prefix = 'per_token_scores.{}.{}'.format(args.stage, args.func)

# pos2vocab is a map containing all words with the given POS tag in a list sorted by frequency
if args.func in ['replace_target', 'replace_target_with_nearby_token', 'drop_target']:
    assert pos_corpus is not None, "No POS file is found"
    pos2vocab = get_vocab_all_pos(
        os.path.join(pos_datafile, '{}.txt'.format(args.stage)),
        corpus.vocab)
    loop_range = [-1]
    prefix += '.{}'.format(args.drop_or_replace_target_window)
else:
    pos2vocab = None

loop_range = list(filter(lambda x: x != 0, loop_range))

print(loop_range)

for boundary in loop_range:
    log_msg, cfunc, res_label = pick_perturbation(args, boundary)
    print(log_msg)
    loss, entropy, all_losses, all_entropies, all_topk = evaluate(
        data_, pos_data,
        lambda data, pdata, targets, ptargets, args: perturb_data(
            data.transpose(1, 0), pdata.transpose(1, 0) if pdata is not None else None,
            boundary, cfunc, args, pos_dict,
            targets[-1], ptargets[-1] if ptargets is not None else None,
            pos2vocab),
        args)
    print_results(res_label, loss, entropy)
    save_results(
        '{}.{}'.format(prefix, boundary),
        [res_label, loss, entropy, all_losses, all_entropies, all_topk],
    )
