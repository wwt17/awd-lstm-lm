"""Evaluate a language model with restricted context, as opposed to infinite context."""

import argparse
import time, os
import math
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn

import data
import model

from utils_context import batchify_context, get_context_batch
from gpt2_decoder import GPT2Decoder

parser = argparse.ArgumentParser(description='Restrict context size provided to language model')
parser.add_argument('--data', type=str, default='data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./WT2.pt',
                    help='model checkpoint to use')
# Consistent with Language Modeling code from Merity et al.
parser.add_argument('--cuda', action='store_false',
                    help='Using this flag turns off CUDA, default value set to True')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--seq_len', type=int, default=50,
                    help='starting context size')
parser.add_argument('--max_seq_len', type=int, default=1000,
                    help='Maximum possible sequence length to make all experiments start at the same example i.e. skip the same number of tokens at the start')
parser.add_argument('--logdir', type=str, default='./',
                    help='location to write per token log loss')
parser.add_argument('--stage', default='valid', choices=['valid', 'test'],
                    help='Run on valid/test set')
args = parser.parse_args()
print(args)

# Set the random seed manually for reproducibility.
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

output_layer = model.decoder.output_layer if is_GPT2 else model.decoder
def criterion_fn(output, targets):
    return criterion(output_layer.weight, output_layer.bias, output.reshape(-1, output.size(-1)), targets.reshape(-1))

corpus = data.prepare_corpus(args.data)

print('On {} set!!!'.format(args.stage))
data_ = batchify_context(getattr(corpus, args.stage), args.batch_size)
if args.cuda:
    data_ = data_.cuda()
print('Made batches.')

print('Context Size: {}'.format(args.seq_len))

def evaluate(data_source, args):
    """Compute the log perplexities for the corpus, data_source."""
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_loss = 0.
    n = 0
    all_losses = []

    # Number of batches excludes the last seq_len tokens as start of context, since the target index would lie out of bounds.
    # Skip examples at the beginning to ensure the first target is the same for each experiment.
    # All experiments then operate on the same examples.
    # Select the max_seq_len as the largest context size you'd like to test.
    nbatches = (data_source.size(0) - args.seq_len) // args.batch_size
    examples_to_ignore = args.max_seq_len - args.seq_len
    print('Number of tokens to ignore: {}'.format(examples_to_ignore))
    batches_to_ignore = examples_to_ignore // args.batch_size

    cross_entropy = nn.CrossEntropyLoss(reduction='none')

    with torch.no_grad():
        # when using your own LM, ensure the init hidden function exists!
        init_hidden = model.init_hidden(args.batch_size)
        t = tqdm(range(batches_to_ignore, nbatches))
        for i in t:
            data, targets = get_context_batch(data_source, i, args.batch_size, args.seq_len)
            if n == 0:
                print('First word: {}'.format(corpus.dictionary.idx2word[data.data[-1, 0]]))
            output, _ = model(data, init_hidden)
            loss = criterion_fn(output[-1:], targets[-1:])
            total_loss += targets.size(1) * loss.item()
            n += targets.size(1)
            t.set_postfix_str('loss={:8.6f} ppl={:7.3f}'.format(loss, math.exp(loss)))

        print('Last word: {}'.format(corpus.dictionary.idx2word[data[-1, -1]]))
        print('Total examples processed: {}'.format(n))

    return total_loss / n, all_losses

loss, all_losses = evaluate(data_, args)
print('seq_len: {}, loss: {}, ppl: {}'.format(args.seq_len, loss, math.exp(loss)))

if False:
    with open(os.path.join(args.logdir, 'per_token_scores_{}_{}'.format(args.stage, args.seq_len)), 'wb') as f:
        pickle.dump(args.seq_len, f)
        pickle.dump(all_losses, f)
