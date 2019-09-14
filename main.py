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
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data
import model

import texar as tx

from data import FixedLengthContextDataset
from utils import batchify, get_batch, repackage_hidden, map_structure, get_config_model, get_splits, loss_repr, get_model_fn, get_criterion_fn, get_output_layer, get_perplexities_entropies, convert_data_tuple
from texar.torch.modules import GPT2Decoder

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', choices=['LSTM', 'QRNN', 'GRU', 'GPT2'],
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--pointer', action='store_true',
                    help='to test performance after adding cache pointer')
parser.add_argument('--window', type=int, default=3785,
                    help='pointer window length')
parser.add_argument('--theta', type=float, default=0.6625523432485668,
                    help='mix between uniform distribution and pointer softmax distribution over previous words')
parser.add_argument('--lambdasm', type=float, default=0.12785920428335693,
                    help='linear mix between only pointer (1) and only vocab (0) distribution')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=None,
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
parser.add_argument('--when', nargs="+", type=int, default=[],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--ppl_gap', type=float, default=0.3,
                    help='if val_ppl > best_ppl - ppl_gap then reduce lr')
parser.add_argument('--get_output_hidden', action='store_true',
                    help='whether to get output and hidden states')
parser.add_argument('--save_output_hidden_path', type=str, default='output_hidden',
                    help='path to save output and hidden states')
parser.add_argument('--get_output_hidden_loss', action='store_true',
                    help='whether to get loss when getting output and hidden states')
parser.add_argument('--sample_gap', type=int, default=1,
                    help='the gap of two consecutive samples in getting output and hidden states')
parser.add_argument('--eval_entropy', action='store_true',
                    help='get entropy in evaluation')
parser.add_argument('--num_workers', type=int, default=12,
                    help='number of workers to load data')
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
        torch.save((model, criterion, optimizer), f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

corpus = data.prepare_corpus(args.data)

train_batch_size = args.batch_size
eval_batch_size = args.eval_batch_size
test_batch_size = args.test_batch_size
batch_sizes = {
    'train': train_batch_size,
    'valid': eval_batch_size,
    'test': test_batch_size,
}
if args.get_output_hidden:
    datasets = {stage: getattr(corpus, stage) for stage in ['train', 'valid', 'test']}
else:
    train_data = batchify(corpus.train, train_batch_size, args) if hasattr(corpus, 'train') else None
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
    config_model = get_config_model(args.config_model, ntokens)
    model = GPT2Decoder(hparams=config_model)
else:
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)

last_size = config_model['embed']['dim'] if is_GPT2 else args.emsize

if not criterion:
    criterion = SplitCrossEntropyLoss(last_size, splits=get_splits(ntokens), verbose=False)

optimizer = None

if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    if args.lr is not None and optimizer is not None:
        optimizer.param_groups[0]['lr'] = args.lr
    if not is_GPT2:
        model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
        if args.wdrop:
            from weight_drop import WeightDrop
            for rnn in model.rnns:
                if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
                elif rnn.zoneout > 0: rnn.zoneout = args.wdrop

###
if args.cuda:
    device = torch.device('cuda')
    model.cuda()
    if criterion is not None:
        criterion.cuda()
else:
    device = torch.device('cpu')
###
if is_GPT2:
    model_fn = get_model_fn(model)
criterion_fn = get_criterion_fn(model, criterion, is_GPT2) if criterion is not None else None
params = list(model.parameters()) + (list(criterion.parameters()) if criterion is not None else [])
total_params = sum(map(torch.Tensor.nelement, params))
print('Args: {}'.format(args))
print('Model total parameters: {}'.format(total_params))
output_layer = get_output_layer(model, is_GPT2)
print('Output layer parameters: {}'.format(sum(map(torch.Tensor.nelement, output_layer.parameters()))))

if optimizer is None:
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

###############################################################################
# Training code
###############################################################################


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
            if is_GPT2:
                context_size = args.bptt
            else:
                assert batch_size == 1
                hidden = model.init_hidden(batch_size)
                context_size = 1
            dataset = FixedLengthContextDataset(dataset, context_size)
            n = len(dataset)
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
            if is_GPT2:
                sample_indices = list(range(-1, n, args.sample_gap))
                if sample_indices[-1] != n - 1:
                    sample_indices.append(n - 1)
                sample_indices_p = 0
                dataset = torch.utils.data.Subset(dataset, sample_indices[1:])
            data_loader = DataLoader(
                dataset,
                batch_size=batch_sizes[stage] if is_GPT2 else 1,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
            )
            p = 0
            if args.get_output_hidden_loss:
                total_loss = 0.
            with torch.no_grad():
                t = tqdm(data_loader)
                for batch_i, data_item in enumerate(t):
                    if args.cuda:
                        data_item = map_structure(lambda t: t.cuda(), data_item)
                    data, targets = convert_data_tuple(data_item)
                    if is_GPT2:
                        states = model_fn(data)
                        states, targets = tuple(map(lambda t: t.transpose(0, 1), (states, targets)))
                        new_states, new_targets = [], []
                        for state, target in zip(states, targets):
                            gap = sample_indices[sample_indices_p + 1] - sample_indices[sample_indices_p]
                            sample_indices_p += 1
                            new_states.append(state[-gap:])
                            new_targets.append(target[-gap:])
                        states, targets = tuple(map(lambda l: torch.cat(l).unsqueeze(1), (new_states, new_targets)))
                    else:
                        output, hidden = model(data, hidden)
                        output = output.detach()
                        hidden = repackage_hidden(hidden)
                        states = (output, map_structure(lambda t: t.unsqueeze(0), hidden))
                    seq_len = len(targets)
                    def write(t, m):
                        m[p : p + seq_len] = t.squeeze(-2).cpu()
                    map_structure(write, states, m)
                    p += seq_len
                    if args.get_output_hidden_loss:
                        output = states if is_GPT2 else states[0]
                        loss = criterion_fn(output, targets)
                        total_loss += loss.item() * seq_len
                        if (batch_i + 1) % args.log_interval == 0:
                            t.set_postfix_str(loss_repr(total_loss / p))
            if args.get_output_hidden_loss:
                mean_loss = total_loss / p
                print('| {}'.format(loss_repr(mean_loss)))
    sys.exit(0)


def evaluate(data_source, batch_size, prefix, window=None):
    pointer = (window is not None)
    if pointer:
        theta = args.theta
        lambdah = args.lambdasm
    eval_entropy = args.eval_entropy or pointer
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss, total_entropy = 0., 0.
    with torch.no_grad():
        seq_len = args.bptt
        if not is_GPT2:
            hidden = model.init_hidden(batch_size)
        if pointer:
            next_word_history = torch.zeros([0, batch_size, ntokens], dtype=torch.float, device=device)
            pointer_history = torch.zeros([0, batch_size, model.ninp if model.tie_weights else model.nhid], dtype=torch.float, device=device)
        for i in tqdm(range(0, data_source.size(0) - 1, seq_len)):
            data, targets = get_batch(data_source, i, seq_len)
            if is_GPT2:
                output = model_fn(data)
            else:
                if pointer:
                    output, hidden, rnn_outs, _ = model(data, hidden, return_h=True)
                    rnn_out = rnn_outs[-1]
                else:
                    output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            if eval_entropy:
                logits = output_layer(output)
                p = F.softmax(logits, -1)
                if pointer:
                    start_t = next_word_history.size(0)
                    next_word_history = torch.cat([next_word_history, F.one_hot(targets, ntokens).float()], dim=0).detach()
                    pointer_history = torch.cat([pointer_history, rnn_out], dim=0).detach()
                    ptr_p = []
                    for t in range(targets.size(0)):
                        en_t = start_t + t
                        st_t = max(0, en_t - window)
                        if st_t == en_t:
                            ptr_dist = torch.zeros_like(p[t])
                        else:
                            valid_next_word_history = next_word_history[st_t:en_t]
                            valid_pointer_history = pointer_history[st_t:en_t]
                            copy_scores = torch.einsum('tbi,bi->tb', valid_pointer_history, rnn_out[t])
                            ptr_attn = F.softmax(theta * copy_scores, 0)
                            ptr_dist = (ptr_attn.unsqueeze(-1) * valid_next_word_history).sum(0)
                        ptr_p.append(ptr_dist.detach())
                    ptr_p = torch.stack(ptr_p, dim=0)
                    p = lambdah * ptr_p + (1. - lambdah) * p
                    log_p = p.log()
                    next_word_history = next_word_history[-window:].detach()
                    pointer_history = pointer_history[-window:].detach()
                else:
                    log_p = F.log_softmax(logits, -1)
                losses, entropies = get_perplexities_entropies(p, log_p, targets)
                total_loss += losses.sum()
                total_entropy += entropies.sum()
            else:
                loss = criterion_fn(output, targets)
                total_loss += len(data) * loss.item() * batch_size
    total_tokens = len(data_source) * batch_size
    mean_loss = total_loss / total_tokens
    mean_entropy = total_entropy / total_tokens
    print('{} {} | entropy {}'.format(prefix, loss_repr(mean_loss), mean_entropy))
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

if args.pointer:
    # Run on val and test data.
    for stage in ['valid', 'test']:
        val_loss = evaluate(datasets[stage], batch_sizes[stage], stage, window=args.window)
    sys.exit(0)

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

            need_to_reduce_lr = (math.exp(val_loss) > math.exp(stored_loss) - args.ppl_gap)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when or need_to_reduce_lr:
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
