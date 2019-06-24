import os
import sys
import argparse
import time
from tqdm import tqdm
import math
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import data
from data import HiddenStateDataset
import model
import approx_models

from utils import map_structure

def arg_to_list(t):
    return lambda arg: list(map(t, arg.split(',')))

def int_or_list(arg):
    l = arg_to_list(int)(arg)
    return l[0] if len(l) == 1 else l

parser = argparse.ArgumentParser(description='PyTorch Approximator of RNN/LSTM Language Model')
# Path
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--output_hidden_path', type=str, default='output_hidden',
                    help='path to saved output and hidden states')
parser.add_argument('--approx_model', type=str, default='WT2.pt',
                    help='path of model to approxiate')
parser.add_argument('--ckpt', type=str,  default='',
                    help='path of model to save and resume')
# Hidden state
parser.add_argument('--predict_all_layers', action='store_true')
parser.add_argument('--predict_c', action='store_true')
# Model
parser.add_argument('--model_type', type=str, choices=['mlp', 'cnn'], default='cnn',
                    required=True,
                    help='Type of approximator model (mlp, cnn)')
## Shared
parser.add_argument('--context_size', type=int, required=True,
                    help='size of used context')
## MLP
parser.add_argument('--hidden_size', type=int,
                    help='size of hidden state')
## CNN
parser.add_argument('--n_layers', type=int,
                    help='number of CNN layers')
parser.add_argument('--channels', type=int_or_list,
                    help='number of CNN channels. can be a comma-separated list')
parser.add_argument('--kernel_size', type=int_or_list,
                    help='size of CNN kernels. can be a comma-separated list')
parser.add_argument('--variational', action='store_true',
                    help='whether to share CNN kernel paramters of different layers')
parser.add_argument('--padding', action='store_false',
                    help='whether not to use padding before CNNs')
parser.add_argument('--residual', action='store_true',
                    help='whether to add residual links in CNNs')
parser.add_argument('--output_layer_type', type=str, choices=['fc'], default='fc',
                    help='type of CNN output layer')
# Training/evaluation/test
## Meta
parser.add_argument('--seed', type=int, default=0,
                    help='random seed. default to 0.')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--num_workers', type=int, default=12,
                    help='number of data loader workers. default to 12.')
## Procedure
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
## Batch size
parser.add_argument('--train_batch_size', type=int, default=16, metavar='N',
                    help='train batch size')
parser.add_argument('--valid_batch_size', type=int, default=80,
                    help='eval batch size')
parser.add_argument('--test_batch_size', type=int, default=80,
                    help='test batch size')
## Optimize
parser.add_argument('--optimizer', type=str,  default='adam', choices=['sgd', 'adam'],
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=100.,
                    help='gradient clipping')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--wdecay', type=float, default=0.,
                    help='weight decay applied to all weights')
parser.add_argument('--input_dropout', type=float, default=0.,
                    help='dropout applied to embedded input (0 = no dropout)')
parser.add_argument('--hidden_dropout', type=float, default=0.,
                    help='dropout for hidden layers (0 = no dropout)')
parser.add_argument('--output_dropout', type=float, default=0.,
                    help='dropout applied to output (0 = no dropout)')
parser.add_argument('--weight_dropout', type=float, default=0.5,
                    help='amount of weight dropout to apply to the hidden matrices')
args = parser.parse_args()
required_args = {
    'mlp': ['hidden_size'],
    'cnn': ['n_layers', 'channels', 'kernel_size', 'variational', 'output_layer_type'],
}[args.model_type]
for a in required_args:
    assert getattr(args, a) is not None, 'must specify {}'.format(a)

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

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

context_size = args.context_size

datasets = {
    stage: HiddenStateDataset(
        getattr(corpus, stage),
        os.path.join(args.output_hidden_path, '{}.h5py'.format(stage)),
        context_size,
        corpus.dictionary.eos_id,
        args.predict_all_layers,
        args.predict_c,
    )
    for stage in ['train', 'valid', 'test']
}

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
approx_criterion = None

vocab_size = len(corpus.dictionary)
###
print('Loading approximated model ...')
with open(args.approx_model, 'rb') as f:
    approx_model, approx_criterion, _ = torch.load(f)
###
if not approx_criterion:
    splits = []
    if vocab_size > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif vocab_size > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    approx_criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)


encoder = approx_model.encoder
decoder = approx_model.decoder

embedding_size = encoder.weight.size(1)
hidden_size = args.hidden_size
target_size = datasets['train'].target_size
print('embedding_size={} hidden_size={} target_size={}'.format(
    embedding_size, hidden_size, target_size))

if args.model_type == 'mlp':
    model = approx_models.MLP_Approximator(
        context_size, embedding_size, hidden_size, target_size,
        args.input_dropout, args.hidden_dropout, args.output_dropout,
        args.weight_dropout)
elif args.model_type == 'cnn':
    model = approx_models.CNN_Approximator(
        context_size, embedding_size, args.n_layers, args.channels,
        args.kernel_size, target_size, variational=args.variational,
        padding=args.padding, residual=args.residual,
        output_layer_type=args.output_layer_type,
        input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
        output_dropout=args.output_dropout)
criterion = nn.MSELoss()
if args.cuda:
    approx_criterion.cuda()
    encoder.cuda()
    decoder.cuda()
    model.cuda()
    criterion.cuda()
params = list(model.parameters()) + list(criterion.parameters())
print('parameters:')
for name, param in list(model.named_parameters()) + list(criterion.named_parameters()):
    print('{}:\t{}'.format(name, param.size()))
total_params = sum(map(torch.Tensor.nelement, params))
print('Model total parameters:', total_params)

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

global_step = 0

def model_load(path):
    states = torch.load(model_path)
    for module, state_dict in zip((model, criterion, optimizer), states):
        module.load_state_dict(state_dict)

    global global_step
    if len(states) > 3:
        global_step = states[3]
    else:
        global_step = list(states[2]['state'].values())[0]['step']

def model_save(path):
    global global_step
    torch.save(
        tuple(module.state_dict() for module in (model, criterion, optimizer))
        + (global_step,),
        path)

os.makedirs(args.ckpt, exist_ok=True)
model_path = os.path.join(args.ckpt, 'best.pt')

try:
    model_load(model_path)
except FileNotFoundError:
    pass


###############################################################################
# Training code
###############################################################################

writer = SummaryWriter(os.path.join(args.ckpt, 'log'))


def evaluate(dataset=datasets['valid'], batch_size=args.valid_batch_size, prefix='valid'):
    global global_step
    # Turn on evaluation mode which disables dropout.
    model.eval()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    total_loss = 0.
    total_approx_loss = 0.
    with torch.no_grad():
        for x, y, target in tqdm(data_loader, desc=prefix, dynamic_ncols=True):
            if args.cuda:
                x, y, target = map_structure(
                    lambda t: t.to('cuda', non_blocking=True),
                    (x, y, target))
            input = encoder(x)
            prediction = model(input)
            loss = criterion(prediction, target)
            total_loss += len(x) * loss.item()

            output = dataset.get_output(prediction)
            approx_loss = approx_criterion(
                decoder.weight, decoder.bias,
                output, y)
            total_approx_loss += len(y) * approx_loss

    loss = total_loss / len(dataset)
    approx_loss = total_approx_loss / len(dataset)
    ppl = math.exp(approx_loss)
    print('{} loss={:.6f} approx loss={:6.2f} ppl={:6.2f}'.format(
        prefix, loss, approx_loss, ppl))
    writer.add_scalar('{}/loss'.format(prefix), loss, global_step)
    writer.add_scalar('{}/approx_loss'.format(prefix), approx_loss, global_step)
    writer.add_scalar('{}/ppl'.format(prefix), ppl, global_step)
    return loss, approx_loss


def train(dataset=datasets['train'], batch_size=args.train_batch_size):
    prefix = 'train'
    global global_step
    # Turn on training mode which enables dropout.
    model.train()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    total_loss = 0.
    interval_loss = 0.
    t = tqdm(data_loader, desc='epoch {:3d}'.format(epoch), dynamic_ncols=True)
    for i_batch, (x, y, target) in enumerate(t):
        global_step += 1
        if args.cuda:
            x, y, target = map_structure(
                lambda t: t.to('cuda', non_blocking=True),
                (x, y, target))
        optimizer.zero_grad()
        input = encoder(x)
        prediction = model(input)
        loss = criterion(prediction, target)
        writer.add_scalar('{}/loss'.format(prefix), loss.item(), global_step)
        total_loss += len(y) * loss.item()
        interval_loss += loss.item()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.clip)
        writer.add_scalar('{}/grad_norm'.format(prefix), grad_norm, global_step)
        optimizer.step()
        writer.add_scalar('{}/lr'.format(prefix), optimizer.param_groups[0]['lr'], global_step)

        if (i_batch + 1) % args.log_interval == 0:
            mean_loss = interval_loss / args.log_interval
            t.set_postfix_str('lr={:05.5f} loss={:.6f}'.format(
                optimizer.param_groups[0]['lr'],
                mean_loss))
            interval_loss = 0.
    loss = total_loss / len(dataset)
    print('train loss={:.6f}'.format(
        loss))

# Loop over epochs.
valid_loss = evaluate()
best_valid_loss = valid_loss
valid_losses = [valid_loss]

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        valid_loss = evaluate()

        if valid_loss < best_valid_loss:
            model_save(os.path.join(args.ckpt, 'best.pt'))
            print('Saving model (new best validation)')
            best_valid_loss = valid_loss

        if epoch in args.when:
            print('Saving model before learning rate decreased')
            model_save(os.path.join(args.ckpt, 'epoch{}.pt'.format(epoch)))
            print('Dividing learning rate by 10')
            optimizer.param_groups[0]['lr'] /= 10.

        valid_losses.append(valid_loss)

except KeyboardInterrupt:
    print('Exiting from training early')

# Load the best saved model.
model_load(os.path.join(args.ckpt, 'best.pt'))

# Run on test data.
test_loss = evaluate(datasets['test'], args.test_batch_size, prefix='test')

writer.close()
