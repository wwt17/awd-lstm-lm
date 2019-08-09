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
from utils import map_structure, get_model_fn, get_criterion_fn, get_output_layer, get_perplexities_entropies
from gpt2_decoder import GPT2Decoder
# this is not a good practice, but making an exception in this case
from perturbations import *

parser = argparse.ArgumentParser(description='Test language model on perturbed inputs in the restricted context setting')
parser.add_argument('--data', type=str, default='data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./WT2.pt',
                    help='model checkpoint to use')
# Consistent with Language Modeling code from Merity et al.
parser.add_argument('--cuda', action='store_false',
                    help='Using this flag turns off CUDA, default value set to True')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--logdir', type=str, default='./',
                    help='location to write per token log loss')
parser.add_argument('--stage', default='valid', choices=['valid', 'test'],
                    help='Run on valid/test set')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=300,
                    help='sequence length')
parser.add_argument('--max_seq_len', type=int, default=1000,
                    help='Maximum possible sequence length to make all experiments start at the same example i.e. skip the same number of tokens at the start')
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
parser.add_argument('--num_workers', type=int, default=12,
                    help='Number of workers to load dataset')
args = parser.parse_args()
print(args)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run without --cuda')
    else:
        torch.cuda.manual_seed(args.seed)

print('Load model from {}'.format(args.checkpoint))
start = time.time()
model, criterion, optimizer = torch.load(args.checkpoint)
print('[{:.1f} s]'.format(time.time() - start))

is_GPT2 = isinstance(model, GPT2Decoder)

if args.cuda:
    model.cuda()
else:
    model.cpu()

if is_GPT2:
    model_fn = get_model_fn(model)

output_layer = get_output_layer(model, is_GPT2)
criterion_fn = get_criterion_fn(model, criterion, is_GPT2)

corpus = data.prepare_corpus(args.data)

pos_datafile = os.path.dirname(args.data if args.data.endswith('/') else args.data+'/')+'_pos/'
print('try POS data file: {}'.format(pos_datafile))
if os.path.exists(pos_datafile):
    pos_corpus = data.Corpus(pos_datafile)
    print('Built pos corpus')
else:
    print('POS file does not exist.')
    pos_corpus = None

print('On {} set!!!'.format(args.stage))
data_ = getattr(corpus, args.stage)
if pos_corpus is not None:
    pos_data = getattr(pos_corpus, args.stage)
    pos_dict = pos_corpus.vocab
else:
    pos_data = None
    pos_dict = None

print('Made batches.')

def evaluate(data_source, pdata_source, perturb_fn, args):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    n = 0
    total_loss, total_entropy = 0., 0.
    all_losses, all_entropies = [], []

    # Number of batches excludes the last seq_len tokens as start of context, since the target index would lie out of bounds.
    # Skip examples at the beginning to ensure the first target is the same for each experiment.
    # All experiments then operate on the same examples.
    # Select the max_seq_len as the largest context size you'd like to test.
    examples_to_ignore = args.max_seq_len - args.seq_len
    print('Number of examples to ignore: {}'.format(examples_to_ignore))
    data_source = FixedLengthContextDataset(data_source[examples_to_ignore:], args.seq_len)
    pdata_source = FixedLengthContextDataset(pdata_source[examples_to_ignore:], args.seq_len) if pdata_source is not None else None

    with torch.no_grad():
        if not is_GPT2:
            # when using your own LM, ensure the init hidden function exists!
            init_hidden = model.init_hidden(args.batch_size)
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
        def convert_data_tuple(data_tuple):
            x, y = data_tuple
            return x.transpose(0, 1), y.transpose(0, 1)
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
            data = perturb_fn(data, pdata, targets, ptargets, args)

            if is_GPT2:
                output = model_fn(data)
            else:
                output, _ = model(data, init_hidden)
            if args.entropy:
                losses, entropies = get_perplexities_entropies(output_layer(output[-1]), targets[-1])
                all_losses.append(losses.tolist())
                all_entropies.append(entropies.tolist())
                total_loss += losses.sum()
                total_entropy += entropies.sum()
                loss = losses.mean()
                entropy = entropies.mean()
            else:
                loss = criterion_fn(output[-1:], targets[-1:])
                entropy = 0.
                total_loss += targets.size(1) * loss.item()
            n += targets.size(1)
            t.set_postfix_str('loss={:8.6f} ppl={:7.3f} entropy={:7.5f}'.format(loss, math.exp(loss), entropy))

        print('Last word: {}'.format(corpus.vocab.id_to_token_map_py[int(data[-1, -1])]))
        print('Total: {}'.format(n))

    return total_loss / n, total_entropy / n, all_losses, all_entropies

def print_results(prefix, loss, entropy):
    print('{}: loss: {}, ppl: {}, entropy: {}'.format(prefix, loss, math.exp(loss), entropy))

def save_results(file_name, items):
    with open(os.path.join(args.logdir, file_name), 'wb') as f:
        for item in items:
            pickle.dump(item, f)

if not args.func:
    loss, entropy, all_losses, all_entropies = evaluate(
        data_, pos_data,
        lambda data, pdata, targets, ptargets, args: data,
        args)
    print_results('seq_len={}'.format(args.seq_len), loss, entropy)
    save_results(
        'per_token_scores_{}_{}'.format(args.stage, args.seq_len),
        [args.seq_len, loss, entropy, all_losses, all_entropies],
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
prefix = '{}.per_tok_scores.'.format(args.func)

# pos2vocab is a map containing all words with the given POS tag in a list sorted by frequency
if args.func in ['replace_target', 'replace_target_with_nearby_token', 'drop_target']:
    assert pos_corpus is not None, "No POS file is found"
    pos2vocab = get_vocab_all_pos(
        os.path.join(pos_datafile, '{}.txt'.format(args.stage)),
        corpus.dictionary)
    loop_range = [-1]
    prefix += '{}.'.format(args.drop_or_replace_target_window)
else:
    pos2vocab = None

loop_range = list(filter(lambda x: x != 0, loop_range))

print(loop_range)

for boundary in loop_range:
    log_msg, cfunc, res_label = pick_perturbation(args, boundary)
    print(log_msg)
    loss, entropy, all_losses, all_entropies = evaluate(
        data_, pos_data,
        lambda data, pdata, targets, ptargets, args: perturb_data(
            data.transpose(1, 0), pdata.transpose(1, 0) if pdata is not None else None,
            boundary, cfunc, args, pos_dict,
            targets[-1], ptargets[-1] if ptargets is not None else None,
            pos2vocab),
        args)
    print_results(res_label, loss, entropy)
    save_results(
        prefix+str(boundary),
        [res_label, loss, entropy, all_losses, all_entropies],
    )
