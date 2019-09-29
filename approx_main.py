import os
import sys
import argparse
import time
import itertools
from tqdm import tqdm
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import PackedSequence

import data
import model
import approx_models

from utils import map_structure, get_config_model, get_splits, get_embedder, get_embedding_size, get_output_layer, get_criterion_fn, force_reduce_lr, set_lr
from gpt2_decoder import GPT2Decoder

def arg_to_list(t):
    return lambda arg: list(map(t, arg.split(',')))

def int_or_list(arg):
    l = arg_to_list(int)(arg)
    return l[0] if len(l) == 1 else l

parser = argparse.ArgumentParser(description='Language Model Approximators')
# Path
parser.add_argument('--data', type=str, default='data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--output_hidden_path', type=str, default='output_hidden/WT103.GPT2.512.6/512_32',
                    help='path of saved output and hidden states')
parser.add_argument('--teacher_model', type=str,
                    help='path of model to approxiate')
parser.add_argument('--ckpt', type=str,  default='',
                    help='path of model to save and resume')
# target
parser.add_argument('--approx_dist', action='store_true')
parser.add_argument('--approx_dist_temp', type=float, default=1.)
parser.add_argument('--approx_dist_lambda', type=float, default=1.)
parser.add_argument('--approx_logit', action='store_true')
# Hidden state
parser.add_argument('--predict_all_layers', action='store_true')
parser.add_argument('--predict_c', action='store_true')
# Model
parser.add_argument('--new_embedder', action='store_true')
parser.add_argument('--new_output_layer', action='store_true')
parser.add_argument('--model_type', type=str, choices=['mlp', 'cnn', 'lstm', 'transformer'], default='lstm',
                    required=True,
                    help='Type of approximator model (mlp, cnn, lstm, transformer)')
parser.add_argument('--copy_w', type=float)
## Shared
parser.add_argument('--output_seq', action='store_true')
parser.add_argument('--skip_link', type=str, choices=['res'],
                    help='The type of skip link (res)')
parser.add_argument('--normalization', type=str, choices=['layer'],
                    help='The type of normalization (layer)')
parser.add_argument('--input_transform', action='store_true')
parser.add_argument('--context_size', type=int,
                    help='size of used context')
parser.add_argument('--last_n', type=int)
parser.add_argument('--tied', action='store_true',
                    help='tie the embedding and output_layer')
parser.add_argument('--embedding_size', type=int)
parser.add_argument('--output_size', type=int)
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
                    help='whether to share CNN kernel parameters of different layers')
parser.add_argument('--padding', action='store_false',
                    help='whether not to use padding before CNNs')
parser.add_argument('--output_layer_type', type=str, choices=['fc'], default='fc',
                    help='type of CNN output layer')
## LSTM
parser.add_argument('--bidirectional', action='store_true')
parser.add_argument('--explicit_stack', action='store_true')
parser.add_argument('--no_transform_output', action='store_true')
## Transformer
parser.add_argument('--config_model', type=str, default='config_GPT2_117M',
                    help='The model configuration file to configure the model.')
# Training/evaluation/test
## Meta
parser.add_argument('--seed', type=int,
                    help='random seed.')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of data loader workers. default to 4.')
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
parser.add_argument('--lr', type=float,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--reduce_lr_factor', type=float, default=.1)
parser.add_argument('--reduce_lr_patience', type=int, default=20)
parser.add_argument('--reduce_lr_threshold', type=float, default=1e-4)
parser.add_argument('--reduce_lr_cooldown', type=int, default=20)
parser.add_argument('--reduce_lr_min_lr', type=float, default=0.)
parser.add_argument('--force_reduce_lr', action='store_true')
parser.add_argument('--wdecay', type=float, default=0.,
                    help='weight decay applied to all weights')
parser.add_argument('--input_dropout', type=float, default=0.,
                    help='dropout applied to embedded input (0 = no dropout)')
parser.add_argument('--hidden_dropout', type=float, default=0.,
                    help='dropout for hidden layers (0 = no dropout)')
parser.add_argument('--output_dropout', type=float, default=0.,
                    help='dropout applied to output (0 = no dropout)')
parser.add_argument('--weight_dropout', type=float, default=0.,
                    help='amount of weight dropout to apply to the hidden matrices')
# Logging/save
parser.add_argument('--eval_teacher_loss', action='store_true',
                    help='whether to evaluate the loss of the approximated model')
parser.add_argument('--save_intermediate', action='store_true',
                    help='whether to save intermediate results besides the best one')
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
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

is_classification = 'yelp' in args.data
if is_classification:
    if 'polarity' in args.data:
        n_classes = 2
    elif 'full' in args.data:
        n_classes = 5
    else:
        raise ValueError("Cannot infer number of classes from {}".format(args.data))

###############################################################################
# Load data
###############################################################################

corpus = data.prepare_corpus(args.data, data.get_review_and_star if is_classification else data.get_holistic_text)

###############################################################################
# Load approximated model
###############################################################################

from splitcross import SplitCrossEntropyLoss

vocab_size = corpus.vocab.size
print('vocab_size = {}'.format(vocab_size))
if not is_classification:
    n_classes = vocab_size

print('Loading approximated model ...')
if args.teacher_model is not None:
    teacher_model, teacher_criterion, _ = torch.load(args.teacher_model)
    if teacher_criterion is None:
        teacher_criterion = SplitCrossEntropyLoss(args.embedding_size, splits=get_splits(vocab_size), verbose=False)
    teacher_model.to(device)
    teacher_criterion.to(device)
    is_GPT2 = isinstance(teacher_model, GPT2Decoder)
    # Freeze all parameters of the approximated model, including the embedder and output_layer
    for param in itertools.chain(teacher_model.parameters(), teacher_criterion.parameters()):
        param.requires_grad = False
else:
    assert not args.eval_teacher_loss
    is_GPT2 = True

###############################################################################
# Load dataset
###############################################################################

datasets = {
    stage:
    data.TextClassificationDataset(
        getattr(corpus, stage),
    )
    if is_classification else
    data.HiddenStateDataset(
        getattr(corpus, stage),
        os.path.join(args.output_hidden_path, '{}.h5py'.format(stage)),
        args.context_size,
        last_n=args.last_n if stage == 'train' else None,
        is_GPT2=is_GPT2,
        predict_all_layers=args.predict_all_layers,
        predict_c=args.predict_c,
    )
    for stage in ['train', 'valid', 'test']
}

collate_fn = data.text_classification_collate_fn if is_classification else None

###############################################################################
# Build the model
###############################################################################

def model_load(path):
    global model, criterion, optimizer, global_step, lr_scheduler, embedder, output_layer, copy_w
    try:
        loaded = torch.load(model_path)
    except FileNotFoundError:
        loaded = []
        success = False
    else:
        success = True
        print('use checkpoint {}'.format(model_path))
    loaded += (None,) * (8 - len(loaded))
    model, criterion, optimizer, global_step, lr_scheduler, embedder, output_layer, copy_w = loaded
    return success

def model_save(path):
    torch.save((model, criterion, optimizer, global_step, lr_scheduler, embedder, output_layer, copy_w), path)

os.makedirs(args.ckpt, exist_ok=True)
model_path = os.path.join(args.ckpt, 'best.pt')

model_load(model_path)

if args.teacher_model is not None:
    if embedder is None:
        embedder = get_embedder(teacher_model, is_GPT2)
    if output_layer is None:
        output_layer = get_output_layer(teacher_model, is_GPT2)

embedding_size = args.embedding_size if embedder is None else get_embedding_size(embedder, is_GPT2)
hidden_size = args.hidden_size
output_size = datasets['train'].target_size if isinstance(datasets['train'], data.HiddenStateDataset) else args.output_size
# output_size = output_layer.weight.size(1)
print('embedding_size={} hidden_size={} output_size={}'.format(
    embedding_size, hidden_size, output_size))

if embedder is None and args.new_embedder:
    embedder = nn.Embedding(vocab_size, embedding_size)
if output_layer is None and args.new_output_layer:
    output_layer = nn.Linear(output_size, n_classes)
    if args.tied:
        assert output_size == embedding_size and n_classes == vocab_size
        output_layer.weight = embedder.weight

if model is None:
    if args.model_type == 'mlp':
        model = approx_models.MLP_Approximator(
            args.context_size, embedding_size, hidden_size, output_size,
            args.input_dropout, args.hidden_dropout, args.output_dropout,
            args.weight_dropout)
    elif args.model_type == 'cnn':
        model = approx_models.CNN_Approximator(
            args.context_size, embedding_size, args.n_layers, args.channels,
            args.kernel_size, output_size, variational=args.variational,
            padding=args.padding, skip_link=args.skip_link,
            output_layer_type=args.output_layer_type,
            input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
            output_dropout=args.output_dropout)
    elif args.model_type == 'lstm':
        model = approx_models.LSTM_Approximator(
            embedding_size, hidden_size, args.n_layers,
            output_seq=args.output_seq,
            bidirectional=args.bidirectional,
            explicit_stack=args.explicit_stack,
            skip_link=args.skip_link,
            normalization=args.normalization,
            input_transform=args.input_transform,
            output_size=None if args.no_transform_output else output_size,
            input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
            output_dropout=args.output_dropout)
    elif args.model_type == 'transformer':
        config_model = get_config_model(args.config_model, vocab_size)
        model = approx_models.Transformer_Approximator(
            hparams=config_model,
            output_size=None if args.no_transform_output else output_size,
        )

if copy_w is None and args.copy_w is not None:
    copy_w = nn.Parameter(torch.tensor(args.copy_w, requires_grad=True, device=device))

criterion = nn.MSELoss()
model.to(device)
embedder.to(device)
output_layer.to(device)
criterion.to(device)
if args.teacher_model is not None:
    teacher_criterion_fn = get_criterion_fn(teacher_model, teacher_criterion, is_GPT2)
    del teacher_model

def get_named_params():
    named_params = list(itertools.chain(model.named_parameters(), criterion.named_parameters()))
    if args.new_embedder:
        named_params.extend(embedder.named_parameters())
    if args.new_output_layer:
        named_params.extend(output_layer.named_parameters())
    if copy_w is not None:
        named_params.append(('copy_w', copy_w))
    return named_params

named_params = get_named_params()
print('parameters:')
for name, param in named_params:
    print('{}:\t{}'.format(name, param.size()))
params = [param for name, param in named_params]
total_params = sum(map(torch.Tensor.nelement, params))
print('Model total # parameters:', total_params)

if lr_scheduler is not None:
    if args.lr is not None:
        set_lr(lr_scheduler, args.lr)
    elif args.force_reduce_lr:
        force_reduce_lr(lr_scheduler)

if optimizer is None and args.lr is not None:
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

if global_step is None:
    global_step = 0

###############################################################################
# Training code
###############################################################################

writer = SummaryWriter(os.path.join(args.ckpt, 'log'))


def get_prediction_and_loss(x, y, teacher_output=None, get_output=None):
    _embedder_fn = lambda x: embedder(x).detach()
    input = (PackedSequence(_embedder_fn(x.data), **{key: getattr(x, key) for key in ('batch_sizes', 'sorted_indices', 'unsorted_indices')}) if isinstance(x, PackedSequence) else _embedder_fn(x))
    prediction_ = model(input)
    def get_last_n(t):
        if isinstance(model, (approx_models.LSTM_Approximator, approx_models.Transformer_Approximator)) and args.output_seq:
            if args.last_n is None or not model.training:
                t = t[:, -1]
            else:
                t = t[:, -args.last_n:]
        return t
    prediction = get_last_n(prediction_)
    assert not (args.approx_dist and args.approx_logit)
    if args.approx_dist or args.approx_logit:
        logits = output_layer(prediction)

        if copy_w is not None:
            seq_len = prediction_.size(1)
            scores = torch.einsum('bik,bjk->bij', prediction_, prediction_)
            bias = torch.triu(torch.full([seq_len, seq_len], -1e18, device=device))
            scores += bias
            scores = scores.narrow(-1, 0, scores.size(-1) - 1)
            scores = get_last_n(scores)
            scores *= copy_w
            logits_ = torch.cat([logits, scores], -1)
            p_ = logits_.softmax(-1)
            p, p_copy_seq = p_.narrow(-1, 0, vocab_size), p_.narrow(-1, vocab_size, scores.size(-1))
            x_ = x[:, 1:]
            if p_copy_seq.dim() > 2:
                x_ = x_.unsqueeze(1).expand_as(p_copy_seq)
            p = p.scatter_add(-1, x_, p_copy_seq)
            # TODO: fix NaN or Inf in following two lines
            log_p = p.log()
            logits = log_p
        else:
            p = logits.softmax(-1)
            log_p = logits.log_softmax(-1)
        entropies = -(p * log_p).sum(-1)
        if teacher_output is not None:
            teacher_logits = output_layer(get_output(teacher_output)).detach()
        if args.approx_logit:
            assert teacher_output is not None
            loss = F.mse_loss(logits, teacher_logits)
            gt_ce_loss = None
        else:
            T = args.approx_dist_temp
            L = args.approx_dist_lambda
            if L > 0.:
                assert teacher_output is not None
                s = 1.
                for d in logits.shape[1:-1]:
                    s *= d
                kl_loss = F.kl_div((logits / T).log_softmax(-1), (teacher_logits / T).softmax(-1), reduction='batchmean') / s
            logits_dim = logits.dim()
            gt_ce_loss = F.cross_entropy(logits.permute(*([0, logits_dim - 1] + list(range(1, logits_dim - 1)))), y)
            loss = ((L * T * T) * kl_loss if L > 0. else 0.) + (1. - L) * gt_ce_loss
    else:
        assert teacher_output is not None
        loss = criterion(prediction, teacher_output)
        gt_ce_loss, entropies = None, None
    return prediction, loss, gt_ce_loss, entropies

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
        **({'collate_fn': collate_fn} if collate_fn is not None else {}),
    )
    total_loss = 0.
    total_entropy = 0.
    total_gt_ce_loss = 0.
    if args.eval_teacher_loss:
        total_teacher_loss = 0.
    with torch.no_grad():
        t = tqdm(data_loader, desc=prefix, dynamic_ncols=True)
        n = 0
        for data_item in t:
            if args.cuda:
                data_item = map_structure(
                    lambda t: t.cuda(non_blocking=True),
                    data_item)
            if is_classification:
                x, y = data_item
            else:
                x, y, teacher_output = data_item
            batch_size = len(y)
            n += batch_size
            prediction, loss, gt_ce_loss, entropies = get_prediction_and_loss(x, y, teacher_output, dataset.get_output) if not is_classification else get_prediction_and_loss(x, y)
            total_loss += loss.item() * batch_size
            if entropies is not None:
                total_entropy += entropies.sum().item()
            if not args.approx_dist:
                output = dataset.get_output(prediction)
                gt_ce_loss = teacher_criterion_fn(output, y)
            total_gt_ce_loss += gt_ce_loss.item() * batch_size
            postfix = 'ppl={:.3f}'.format(
                math.exp(total_gt_ce_loss / n))
            if entropies is not None:
                postfix += ' ent={:.3f}'.format(
                    total_entropy / n)
            if args.eval_teacher_loss:
                teacher_loss = teacher_criterion_fn(dataset.get_output(teacher_output), y)
                total_teacher_loss += teacher_loss * batch_size
                postfix += ' teacher_ppl={:.3f}'.format(
                    math.exp(total_teacher_loss / n))
            if copy_w is not None:
                postfix += ' copy_w={:.3f}'.format(copy_w.item())
            t.set_postfix_str(postfix)

    loss = total_loss / len(dataset)
    entropy = total_entropy / len(dataset)
    gt_ce_loss = total_gt_ce_loss / len(dataset)
    ppl = math.exp(gt_ce_loss)
    print('{} loss={:.6f} approx loss={:6.2f} ppl={:6.2f} ent={:2.3f}'.format(
        prefix, loss, gt_ce_loss, ppl, entropy))
    writer.add_scalar('{}/loss'.format(prefix), loss, global_step)
    writer.add_scalar('{}/gt_ce_loss'.format(prefix), gt_ce_loss, global_step)
    writer.add_scalar('{}/ppl'.format(prefix), ppl, global_step)
    if entropy:
        writer.add_scalar('{}/entropy'.format(prefix), entropy, global_step)

    if args.last_n is not None:
        model.last_n = args.last_n
    return loss, gt_ce_loss


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
        **({'collate_fn': collate_fn} if collate_fn is not None else {}),
    )
    total_loss = 0.
    total_entropy = 0.
    interval_loss = 0.
    interval_entropy = 0.
    t = tqdm(data_loader, desc='epoch {:3d}'.format(epoch), dynamic_ncols=True)
    for i_batch, data_item in enumerate(t):
        global_step += 1
        if args.cuda:
            data_item = map_structure(
                lambda t: t.cuda(non_blocking=True),
                data_item)
        if is_classification:
            x, y = data_item
        else:
            x, y, teacher_output = data_item
        batch_size = len(y)
        optimizer.zero_grad()
        prediction, loss, gt_ce_loss, entropies = get_prediction_and_loss(x, y, teacher_output, dataset.get_output) if not is_classification else get_prediction_and_loss(x, y)
        writer.add_scalar('{}/loss'.format(prefix), loss.item(), global_step)
        if gt_ce_loss is not None:
            writer.add_scalar('{}/gt_ce_loss'.format(prefix), gt_ce_loss.item(), global_step)
        total_loss += loss.item() * batch_size
        interval_loss += loss.item()
        if entropies is not None:
            entropy = entropies.sum()
            total_entropy += entropy.item()
            writer.add_scalar('{}/entropy'.format(prefix), entropies.mean().item(), global_step)
            interval_entropy += entropy.item()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.clip)
        writer.add_scalar('{}/grad_norm'.format(prefix), grad_norm, global_step)
        optimizer.step()
        writer.add_scalar('{}/lr'.format(prefix), optimizer.param_groups[0]['lr'], global_step)
        if copy_w is not None:
            writer.add_scalar('{}/copy_w'.format(prefix), copy_w.item(), global_step)

        if (i_batch + 1) % args.log_interval == 0:
            mean_loss = interval_loss / args.log_interval
            mean_entropy = interval_entropy / args.log_interval / batch_size
            t.set_postfix_str('lr={:05.5f} loss={:.6f} ent={:.6f}{}'.format(
                optimizer.param_groups[0]['lr'],
                mean_loss,
                mean_entropy,
                ' copy_w={:.3f}'.format(copy_w.item()) if copy_w is not None else '',))
            interval_loss = 0.
            interval_entropy = 0.

        if global_step % args.eval_interval == 0:
            valid_loss = evaluate()
            update_valid_loss(valid_loss)
            model.train()

    loss = total_loss / len(dataset)
    entropy = total_entropy / len(dataset)
    print('train loss={:.6f} entropy={:.6f}'.format(
        loss, entropy))

def update_valid_loss(valid_loss):
    valid_loss, valid_gt_ce_loss = valid_loss
    global best_valid_loss
    if valid_loss < best_valid_loss:
        print('Saving model (new best validation)')
        if args.save_intermediate:
            model_save(os.path.join(args.ckpt, 'step{}.pt'.format(global_step)))
        model_save(os.path.join(args.ckpt, 'best.pt'))
        best_valid_loss = valid_loss
    lr_scheduler.step(valid_loss)

# Loop over epochs.
# At any point you can hit Ctrl + C to break out of training early.
try:
    valid_loss = evaluate()
    best_valid_loss, _ = valid_loss

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        valid_loss = evaluate()
        update_valid_loss(valid_loss)

except KeyboardInterrupt:
    print('Exiting from training early')

# Load the best saved model.
best_model_path = os.path.join(args.ckpt, 'best.pt')
if not model_load(best_model_path):
    raise Exception('{} is not found'.format(best_model_path))

# Run on test data.
test_loss = evaluate(datasets['test'], args.test_batch_size, prefix='test')

writer.close()
