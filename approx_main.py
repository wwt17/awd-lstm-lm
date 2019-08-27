import os
import sys
import argparse
import time
from tqdm import tqdm
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import data
from data import HiddenStateDataset
import model
import approx_models

from utils import map_structure, get_config_model, get_splits, get_embedder, get_embedding_size, get_output_layer, get_model_fn, get_criterion_fn
from gpt2_decoder import GPT2Decoder

def arg_to_list(t):
    return lambda arg: list(map(t, arg.split(',')))

def int_or_list(arg):
    l = arg_to_list(int)(arg)
    return l[0] if len(l) == 1 else l

parser = argparse.ArgumentParser(description='PyTorch Approximator of RNN/LSTM Language Model')
# Path
parser.add_argument('--data', type=str, default='data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--output_hidden_path', type=str, default='output_hidden',
                    help='path to saved output and hidden states')
parser.add_argument('--approx_model', type=str, default='WT2.pt',
                    help='path of model to approxiate')
parser.add_argument('--ckpt', type=str,  default='',
                    help='path of model to save and resume')
# target
parser.add_argument('--approx_distribution', action='store_true')
parser.add_argument('--approx_dist_temp', type=float, default=1.)
parser.add_argument('--approx_dist_lambda', type=float, default=1.)
# Hidden state
parser.add_argument('--predict_all_layers', action='store_true')
parser.add_argument('--predict_c', action='store_true')
# Model
parser.add_argument('--model_type', type=str, choices=['mlp', 'cnn', 'lstm', 'transformer'], default='cnn',
                    required=True,
                    help='Type of approximator model (mlp, cnn, lstm, transformer)')
## Shared
parser.add_argument('--context_size', type=int, required=True,
                    help='size of used context')
parser.add_argument('--last_n', type=int, default=None)
## Shared by MLP, LSTM and Transformer
parser.add_argument('--hidden_size', type=int,
                    help='size of hidden state')
## Shared by CNN, LSTM
parser.add_argument('--n_layers', type=int,
                    help='number of CNN/LSTM layers')
## CNN
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
## LSTM
parser.add_argument('--no_transform_output', action='store_true')
## Transformer
parser.add_argument('--config_model', type=str, default='config_GPT2_117M',
                    help='The model configuration file to configure the model.')
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
parser.add_argument('--eval-interval', type=int, default=10000,
                    help='eval interval')
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
parser.add_argument('--reduce_lr_factor', type=float, default=.1)
parser.add_argument('--reduce_lr_patience', type=int, default=20)
parser.add_argument('--reduce_lr_threshold', type=float, default=1e-4)
parser.add_argument('--reduce_lr_cooldown', type=int, default=20)
parser.add_argument('--reduce_lr_min_lr', type=float, default=0.)
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
parser.add_argument('--eval_approxed_loss', action='store_true',
                    help='whether to evaluate the loss of the approximated model')
args = parser.parse_args()
required_args = {
    'mlp': ['hidden_size'],
    'cnn': ['n_layers', 'channels', 'kernel_size', 'variational', 'output_layer_type'],
    'lstm': ['hidden_size', 'n_layers'],
    'transformer': ['config_model'],
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

corpus = data.prepare_corpus(args.data)

context_size = args.context_size

###############################################################################
# Load model
###############################################################################

from splitcross import SplitCrossEntropyLoss
approx_criterion = None

vocab_size = corpus.vocab.size
print('vocab_size = {}'.format(vocab_size))
###
print('Loading approximated model ...')
with open(args.approx_model, 'rb') as f:
    approx_model, approx_criterion, _ = torch.load(f)
###
if not approx_criterion:
    approx_criterion = SplitCrossEntropyLoss(args.emsize, splits=get_splits(vocab_size), verbose=False)
if args.cuda:
    approx_model.cuda()
    approx_criterion.cuda()
is_GPT2 = isinstance(approx_model, GPT2Decoder)

###############################################################################
# Load dataset
###############################################################################

datasets = {
    stage: HiddenStateDataset(
        getattr(corpus, stage),
        os.path.join(args.output_hidden_path, '{}.h5py'.format(stage)),
        context_size,
        last_n=args.last_n if stage == 'train' else None,
        is_GPT2=is_GPT2,
        predict_all_layers=args.predict_all_layers,
        predict_c=args.predict_c,
    )
    for stage in ['train', 'valid', 'test']
}

###############################################################################
# Build the model
###############################################################################

embedder = get_embedder(approx_model, is_GPT2)
output_layer = get_output_layer(approx_model, is_GPT2)

embedding_size = get_embedding_size(embedder, is_GPT2)
hidden_size = args.hidden_size
output_size = datasets['train'].target_size if not args.approx_distribution else vocab_size
# output_size = output_layer.weight.size(1)
print('embedding_size={} hidden_size={} output_size={}'.format(
    embedding_size, hidden_size, output_size))

if args.model_type == 'mlp':
    model = approx_models.MLP_Approximator(
        context_size, embedding_size, hidden_size, output_size,
        args.input_dropout, args.hidden_dropout, args.output_dropout,
        args.weight_dropout)
elif args.model_type == 'cnn':
    model = approx_models.CNN_Approximator(
        context_size, embedding_size, args.n_layers, args.channels,
        args.kernel_size, output_size, variational=args.variational,
        padding=args.padding, residual=args.residual,
        output_layer_type=args.output_layer_type,
        input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
        output_dropout=args.output_dropout)
elif args.model_type == 'lstm':
    model = approx_models.LSTM_Approximator(
        embedding_size, hidden_size, args.n_layers,
        None if args.no_transform_output else output_size,
        input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
        output_dropout=args.output_dropout)
elif args.model_type == 'transformer':
    config_model = get_config_model(args.config_model, vocab_size)
    model = approx_models.Transformer_Approximator(
        hparams=config_model,
        output_size=None if args.no_transform_output else output_size,
        keep_output_layer=args.approx_distribution,
    )
criterion = nn.MSELoss()
if args.cuda:
    model.cuda()
    criterion.cuda()
approx_model.eval()
if is_GPT2:
    approx_model_fn = get_model_fn(approx_model)
approx_criterion_fn = get_criterion_fn(approx_model, approx_criterion, is_GPT2)
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
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    factor=args.reduce_lr_factor,
    patience=args.reduce_lr_patience,
    verbose=True,
    threshold=args.reduce_lr_threshold,
    cooldown=args.reduce_lr_cooldown,
    min_lr=args.reduce_lr_min_lr,
)

global_step = 0

def model_load(path):
    global model, criterion, optimizer, global_step, lr_scheduler
    loaded = torch.load(model_path)
    model, criterion, optimizer, global_step = loaded[:4]
    if len(loaded) > 4:
        lr_scheduler = loaded[4]

def model_save(path):
    torch.save((model, criterion, optimizer, global_step, lr_scheduler), path)

os.makedirs(args.ckpt, exist_ok=True)
model_path = os.path.join(args.ckpt, 'best.pt')

try:
    model_load(model_path)
except FileNotFoundError:
    pass

params = list(model.parameters()) + list(criterion.parameters())

###############################################################################
# Training code
###############################################################################

writer = SummaryWriter(os.path.join(args.ckpt, 'log'))


def get_prediction_and_loss(x, y, approx_output, get_output):
    input = embedder(x).detach()
    prediction = model(input)
    if isinstance(model, (approx_models.LSTM_Approximator, approx_models.Transformer_Approximator)):
        if args.last_n is None or not model.training:
            prediction = prediction[:, -1]
        else:
            prediction = prediction[:, -args.last_n:]
    if args.approx_distribution:
        T = args.approx_dist_temp
        L = args.approx_dist_lambda
        logits = prediction
        teacher_logits = output_layer(get_output(approx_output)).detach()
        s = 1.
        for d in logits.shape[1:-1]:
            s *= d
        kl_loss = F.kl_div((logits / T).log_softmax(-1), (teacher_logits / T).softmax(-1), reduction='batchmean') / s
        logits_dim = len(logits.shape)
        gt_ce_loss = F.cross_entropy(logits.permute(*([0, logits_dim - 1] + list(range(1, logits_dim - 1)))), y)
        loss = (L * T * T) * kl_loss + (1. - L) * gt_ce_loss
    else:
        loss = criterion(prediction, approx_output)
        gt_ce_loss = None
    return prediction, loss, gt_ce_loss

def evaluate(dataset=datasets['valid'], batch_size=args.valid_batch_size, prefix='valid'):
    global global_step
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.last_n is not None:
        model.last_n = None
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    total_loss = 0.
    total_approx_loss = 0.
    if args.eval_approxed_loss:
        total_approxed_loss = 0.
    with torch.no_grad():
        t = tqdm(data_loader, desc=prefix, dynamic_ncols=True)
        n = 0
        for data_item in t:
            if args.cuda:
                data_item = map_structure(
                    lambda t: t.to('cuda', non_blocking=True),
                    data_item)
            x, y, approx_output = data_item
            batch_size = len(y)
            n += batch_size
            prediction, loss, gt_ce_loss = get_prediction_and_loss(x, y, approx_output, dataset.get_output)
            total_loss += loss.item() * batch_size
            if args.approx_distribution:
                approx_loss = gt_ce_loss
            else:
                output = dataset.get_output(prediction)
                approx_loss = approx_criterion_fn(output, y)
            total_approx_loss += approx_loss.item() * batch_size
            postfix = 'ppl={:.3f}'.format(
                math.exp(total_approx_loss / n))
            if args.eval_approxed_loss:
                approxed_loss = approx_criterion_fn(dataset.get_output(approx_output), y)
                total_approxed_loss += approxed_loss * batch_size
                postfix += ' approxed_ppl={:.3f}'.format(
                    math.exp(total_approxed_loss / n))
            t.set_postfix_str(postfix)

    loss = total_loss / len(dataset)
    approx_loss = total_approx_loss / len(dataset)
    ppl = math.exp(approx_loss)
    print('{} loss={:.6f} approx loss={:6.2f} ppl={:6.2f}'.format(
        prefix, loss, approx_loss, ppl))
    writer.add_scalar('{}/loss'.format(prefix), loss, global_step)
    writer.add_scalar('{}/approx_loss'.format(prefix), approx_loss, global_step)
    writer.add_scalar('{}/ppl'.format(prefix), ppl, global_step)

    if args.last_n is not None:
        model.last_n = args.last_n
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
    for i_batch, data_item in enumerate(t):
        global_step += 1
        if args.cuda:
            data_item = map_structure(
                lambda t: t.to('cuda', non_blocking=True),
                data_item)
        x, y, approx_output = data_item
        batch_size = len(y)
        optimizer.zero_grad()
        prediction, loss, gt_ce_loss = get_prediction_and_loss(x, y, approx_output, dataset.get_output)
        writer.add_scalar('{}/loss'.format(prefix), loss.item(), global_step)
        if gt_ce_loss is not None:
            writer.add_scalar('{}/gt_ce_loss'.format(prefix), gt_ce_loss.item(), global_step)
        total_loss += loss.item() * batch_size
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

        if global_step % args.eval_interval == 0:
            valid_loss = evaluate()
            update_valid_loss(valid_loss)
            model.train()

    loss = total_loss / len(dataset)
    print('train loss={:.6f}'.format(
        loss))

# Loop over epochs.
valid_loss = evaluate()
best_valid_loss, _ = valid_loss

def update_valid_loss(valid_loss):
    valid_loss, valid_approx_loss = valid_loss
    global best_valid_loss
    if valid_loss < best_valid_loss:
        print('Saving model (new best validation)')
        model_save(os.path.join(args.ckpt, 'step{}.pt'.format(global_step)))
        model_save(os.path.join(args.ckpt, 'best.pt'))
        best_valid_loss = valid_loss
    lr_scheduler.step(valid_loss)

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        valid_loss = evaluate()
        update_valid_loss(valid_loss)

except KeyboardInterrupt:
    print('Exiting from training early')

# Load the best saved model.
model_load(os.path.join(args.ckpt, 'best.pt'))

# Run on test data.
test_loss = evaluate(datasets['test'], args.test_batch_size, prefix='test')

writer.close()
