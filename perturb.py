"""Evaluate a language model on perturbed input in the restricted context setting."""

import argparse
import time, os
import math
import numpy as np
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn

import data
import model

from utils_context import batchify_context, get_context_batch, get_vocab_all_pos
# this is not a good practice, but making an exception in this case
from perturbations import *

parser = argparse.ArgumentParser(description='Test language model on perturbed inputs')
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
                    help='max context size')
parser.add_argument('--freq', type=int, default=108,
                    help='Frequent words cut-off, all words with corpus count > 800')
parser.add_argument('--rare', type=int, default=1986,
                    help='Rare words cut-off, all words with corpus count < 50')
parser.add_argument('--func', type=str,
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

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.prepare_corpus(args.data)

pos_datafile = os.path.dirname(args.data if args.data.endswith('/') else args.data+'/')+'_pos/'
print(pos_datafile)
if os.path.exists(pos_datafile):
    pos_corpus = data.Corpus(pos_datafile)
    print('Built pos corpus')
else:
    print('POS file does not exist.')
    pos_corpus = None

print('On {} set!!!'.format(args.stage))
data_ = batchify_context(getattr(corpus, args.stage), args.batch_size)
if pos_corpus is not None:
    pos_data = batchify_context(getattr(pos_corpus, args.stage), args.batch_size)
else:
    pos_data = None
if args.cuda:
    data_ = data_.cuda()
    if pos_corpus is not None:
        pos_data = pos_data.cuda()
print('Made batches.')

def evaluate(data_source, pdata_source, boundary, func, args, pos2vocab):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.
    n = 0
    total_len = 0
    all_losses = []

    if pos_corpus is not None:
        pos_dict = pos_corpus.dictionary
    else:
        pos_dict = None

    nbatches = (data_source.size(0) - args.seq_len) // args.batch_size
    examples_to_ignore = args.max_seq_len - args.seq_len
    print('Number of examples to ignore: {}'.format(examples_to_ignore))
    batches_to_ignore = examples_to_ignore // args.batch_size

    cross_entropy = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        init_hidden = model.init_hidden(args.batch_size)
        t = tqdm(range(batches_to_ignore, nbatches))
        for i in t:
            data, targets = get_context_batch(data_source, i, args.batch_size, args.seq_len)
            if n == 0:
                print('First word: {}'.format(corpus.dictionary.idx2word[data.data[-1, 0]]))
            pdata, ptargets = get_context_batch(pdata_source, i, args.batch_size, args.seq_len)
            data = perturb_data(data.transpose(1, 0), pdata.transpose(1, 0), boundary, func, args, pos_dict, targets[-1], ptargets[-1], pos2vocab)

            output, _ = model(data, init_hidden)
            total_len += data.shape[0] * data.shape[1]

            output = output.reshape(*(data.size() + output.size()[-1:]))
            logits = model.decoder(output)
            CELoss = cross_entropy(logits.transpose(1, 2), targets)
            losses = CELoss[-1]
            all_losses += losses.tolist()
            loss = torch.sum(losses)
            total_loss += loss.item()
            n += targets.size(1)
            mean_loss = loss / targets.size(1)
            t.set_postfix_str('loss={:.6f} ppl={:.3f}'.format(mean_loss, math.exp(mean_loss)))

        print('Last word: {}' % (corpus.dictionary.idx2word[data.data[-1, -1]]))
        print('total: {}'.format(n))
        print('=' * 89)
        print('Average Sequence Length post changes  {:.4f}'.format(total_len / n))
        print('=' * 89)

    return total_loss / n, all_losses


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
    loss, all_losses = evaluate(data_, pos_data, boundary, cfunc, args, pos2vocab)
    print('{} loss: {}, ppl: {}'.format(res_label, loss, math.exp(loss)))

    with open(os.path.join(args.logdir, prefix+str(boundary)), 'wb') as f:
        pickle.dump(res_label, f)
        pickle.dump(all_losses, f)
