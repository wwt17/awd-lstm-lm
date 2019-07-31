import os
import sys
import argparse
import time
from tqdm import tqdm
import h5py
import math
import numpy as np
import torch
import torch.nn as nn

import data
import model

import importlib
import texar as tx

from utils import batchify, get_batch, repackage_hidden, map_structure, loss_repr, get_model_fn, get_criterion_fn
from gpt2_decoder import GPT2Decoder

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM', 'QRNN', 'GRU', 'GPT2'],
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='train batch size')
parser.add_argument('--eval_batch_size', type=int, default=10,
                    help='eval batch size')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='test batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed. default to 1111.')
parser.add_argument('--nonmono', type=int, default=5,
                    help='nonmono')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--config_model', type=str, default='config_GPT2_117M',
                    help='The model configuration file to configure the model.')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd', choices=['sgd', 'adam'],
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--get_output_hidden', action='store_true',
                    help='whether to get output and hidden states')
parser.add_argument('--save_output_hidden_path', type=str, default='output_hidden',
                    help='path to save output and hidden states')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([m.state_dict() for m in [model, criterion, optimizer]], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        state_dicts = torch.load(f)
    for m, state_dict in zip([model, criterion, optimizer], state_dicts):
        m.load_state_dict(state_dict)

corpus = data.prepare_corpus(args.data)

if args.get_output_hidden:
    train_batch_size = 1
    eval_batch_size = 1
    test_batch_size = 1
else:
    train_batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    test_batch_size = args.test_batch_size
train_data = batchify(corpus.train, train_batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)
datasets = {
    'train': train_data,
    'valid': val_data,
    'test': test_data,
}

###############################################################################
# Build the model
###############################################################################


is_GPT2 = (args.model == 'GPT2')

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = corpus.vocab.size
print('ntokens: {}'.format(ntokens))
if is_GPT2:
    config_model = importlib.import_module(args.config_model)
    config_model = {
        k: v for k, v in config_model.__dict__.items()
        if not k.startswith('__')}
    config_model.pop('dim')
    config_model['vocab_size'] = ntokens
    model = GPT2Decoder(hparams=config_model)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)

if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(config_model['embed']['dim'] if is_GPT2 else args.emsize, splits=splits, verbose=False)
###
if args.cuda:
    device = torch.device('cuda')
    model.cuda()
    criterion.cuda()
else:
    device = torch.device('cpu')
###
if is_GPT2:
    model_fn = get_model_fn(model)
criterion_fn = get_criterion_fn(model, criterion, is_GPT2)
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(map(torch.Tensor.nelement, params))
print('Args: {}'.format(args))
print('Model total parameters: {}'.format(total_params))
print('Output layer parameters: {}'.format(sum(map(torch.Tensor.nelement, (model.decoder.output_layer if is_GPT2 else model.decoder).parameters()))))

# Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    if not is_GPT2:
        model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
        if args.wdrop:
            from weight_drop import WeightDrop
            for rnn in model.rnns:
                if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
                elif rnn.zoneout > 0: rnn.zoneout = args.wdrop

###############################################################################
# Training code
###############################################################################


# get hidden
def all_output_hidden(dataset):
    with torch.no_grad():
        if is_GPT2:
            seq_len = args.bptt
        else:
            hidden = model.init_hidden(dataset.size(1))
            seq_len = 1
        for i in tqdm(range(0, dataset.size(0) - 1, seq_len)):
            data, targets = get_batch(dataset, i, seq_len)
            if is_GPT2:
                yield model_fn(data), targets
            else:
                output, hidden = model(data, hidden)
                output = output.detach()
                hidden = repackage_hidden(hidden)
                yield (output, map_structure(lambda t: t.unsqueeze(0), hidden)), targets


if args.get_output_hidden:
    print('To get output and hidden states.')
    model.eval()
    if args.model == 'QRNN': model.reset()
    ninp = args.emsize
    nhid = args.nhid
    nlayers = args.nlayers
    tie_weights = args.tied
    os.makedirs(args.save_output_hidden_path, exist_ok=True)
    for stage, dataset in datasets.items():
        print('Working on {} set ...'.format(stage))
        save_path = os.path.join(args.save_output_hidden_path, '{}.h5py'.format(stage))
        with h5py.File(save_path, 'w') as f:
            n = dataset.size(0) - 1
            last_size = ninp if is_GPT2 or tie_weights else nhid
            m_output = f.create_dataset('output', (n, last_size), dtype='f')
            if is_GPT2:
                m = m_output
            else:
                m_hidden = []
                for l in range(nlayers):
                    grp = f.create_group('hidden/{}'.format(l))
                    size = nhid if l != nlayers -1 else last_size
                    m_h = grp.create_dataset('h', (n, 1, size), dtype='f')
                    m_c = grp.create_dataset('c', (n, 1, size), dtype='f')
                    m_hidden.append((m_h, m_c))
                m = (m_output, m_hidden)
            total_loss = 0.
            p = 0
            for states, targets in all_output_hidden(dataset):
                seq_len = len(targets)
                def write(t, m):
                    m[p : p + seq_len] = t.unsqueeze(-2)
                map_structure(write, states, m)
                output = states if is_GPT2 else states[0]
                loss = criterion_fn(output, targets)
                total_loss += loss.item()
                p += seq_len
            mean_loss = total_loss / len(dataset)
            print('| {}'.format(loss_repr(mean_loss)))
    sys.exit(0)


def evaluate(data_source, batch_size, prefix):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0.
    with torch.no_grad():
        seq_len = args.bptt
        if not is_GPT2:
            hidden = model.init_hidden(batch_size)
        for i in tqdm(range(0, data_source.size(0) - 1, seq_len)):
            data, targets = get_batch(data_source, i, seq_len)
            if is_GPT2:
                output = model_fn(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            loss = criterion_fn(output, targets)
            total_loss += len(data) * loss.item()
    mean_loss = total_loss / len(data_source)
    print('{} {}'.format(prefix, loss_repr(mean_loss)))
    return mean_loss


def train():
    print('Training...')
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0.
    start_time = time.time()
    if not is_GPT2:
        hidden = model.init_hidden(train_batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, seq_len)

        optimizer.zero_grad()

        if is_GPT2:
            output = model_fn(data)
        else:
            output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
            hidden = repackage_hidden(hidden)
        raw_loss = criterion_fn(output, targets)

        loss = raw_loss
        if not is_GPT2:
            # Activiation Regularization
            if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness)
            if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.item()
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | {}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, loss_repr(cur_loss)))
            total_loss = 0.
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

# Loop over epochs.
lr = args.lr
stored_loss = evaluate(val_data, eval_batch_size, 'valid')
best_val_loss = []

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data, eval_batch_size, 'valid')
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s'.format(
                    epoch, (time.time() - epoch_start_time)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        else:
            val_loss = evaluate(val_data, eval_batch_size, 'valid')
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s'.format(
              epoch, (time.time() - epoch_start_time)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size, 'test')
print('=' * 89)
print('| End of training')
print('=' * 89)
