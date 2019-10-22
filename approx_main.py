import os
import sys
import argparse
import time
import itertools
from tqdm import tqdm
import h5py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import texar as tx
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import PackedSequence

import data
import model
import approx_models

from utils import map_structure, get_config_model, get_splits, get_embedder, get_embedding_size, get_output_layer, get_criterion_fn, force_reduce_lr, set_lr, cross_entropy, set_all_requires_grad
import glue
import superglue
from gpt2_decoder import GPT2Decoder

def arg_to_list(t):
    return lambda arg: list(map(t, arg.split(',')))

def int_or_list(arg):
    l = arg_to_list(int)(arg)
    return l[0] if len(l) == 1 else l

parser = argparse.ArgumentParser(description='Language Model Approximators')
# Path
parser.add_argument('--data', type=str,
                    help='location of the data corpus')
parser.add_argument('--output_hidden_path', type=str,
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
parser.add_argument('--fix_embedder', action='store_true')
parser.add_argument('--fix_output_layer', action='store_true')
parser.add_argument('--model_type', type=str, choices=['mlp', 'cnn', 'lstm', 'transformer'],
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
parser.add_argument('--transform_output', action='store_true')
## Transformer
parser.add_argument('--config_model', type=str,
                    help='The model configuration file to configure the model.')
parser.add_argument('--pretrained_model_name',
                    help='the name of the pretarined model to use.')
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
parser.add_argument('--valid_metric', type=str, choices=['loss', 'gt_ce_loss', 'entropy', 'acc'], default='loss')
parser.add_argument('--eval_teacher_loss', action='store_true',
                    help='whether to evaluate the loss of the approximated model')
parser.add_argument('--save_intermediate', action='store_true',
                    help='whether to save intermediate results besides the best one')
parser.add_argument('--get_output_hidden', action='store_true')
args = parser.parse_args()

teacher_exists = args.teacher_model is not None

if args.output_hidden_path is None:
    if teacher_exists:
        args.output_hidden_path = os.path.join(os.path.dirname(args.teacher_model), 'output_hidden')
    elif args.get_output_hidden:
        args.output_hidden_path = os.path.join(args.ckpt, 'output_hidden')

batch_sizes = {stage: getattr(args, '{}_batch_size'.format(stage)) for stage in ('train', 'valid', 'test')}

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

if 'yelp' in args.data:
    is_classification = True
    if 'polarity' in args.data:
        n_classes = 2
    elif 'full' in args.data:
        n_classes = 5
    else:
        raise ValueError("Cannot infer number of classes from {}".format(args.data))
    get_fn = data.get_review_and_star
elif 'SuperGLUE' in args.data:
    is_classification = True
    track = os.path.basename(args.data)
    n_classes = superglue.get_n_classes(track)
    get_fn = superglue.get_superglue
elif 'glue_data' in args.data:
    is_classification = True
    track = os.path.basename(args.data)
    n_classes = glue.get_n_classes(track)
    get_fn = glue.get_glue
else:
    is_classification = False
    get_fn = data.get_holistic_text


###############################################################################
# Load data
###############################################################################

corpus = data.prepare_corpus(args.data, get_fn)

###############################################################################
# Load teacher model
###############################################################################

from splitcross import SplitCrossEntropyLoss

vocab_size = corpus.vocab.size
print('vocab_size = {}'.format(vocab_size))
if not is_classification:
    n_classes = vocab_size

if teacher_exists:
    print('Loading teacher model ...')
    loaded = torch.load(args.teacher_model)
    teacher_model, teacher_criterion = loaded[:2]
    new_format_teacher = (len(loaded) >= 7)
    if new_format_teacher:
        teacher_embedder, teacher_output_layer = loaded[5:7]
        is_GPT2 = True
    else:
        is_GPT2 = isinstance(teacher_model, GPT2Decoder)
        teacher_embedder = get_embedder(teacher_model, is_GPT2)
        teacher_output_layer = get_output_layer(teacher_output_layer, is_GPT2)
        if teacher_criterion is None:
            teacher_criterion = SplitCrossEntropyLoss(args.embedding_size, splits=get_splits(vocab_size), verbose=False)
    teacher_model.to(device)
    teacher_criterion.to(device)
    teacher_embedder.to(device)
    teacher_output_layer.to(device)
    # Freeze all parameters of the teacher model, including the embedder and output_layer
    set_all_requires_grad(itertools.chain(teacher_model.parameters(), teacher_criterion.parameters()), False)
else:
    assert not args.eval_teacher_loss

###############################################################################
# Load dataset
###############################################################################

def get_dataset(stage):
    raw_data = getattr(corpus, stage)
    output_seq = args.output_seq and stage == 'train'
    if is_classification:
        text_dataset = data.TextClassificationDataset(raw_data)
    else:
        text_dataset = data.FixedLengthContextDataset(
            raw_data,
            args.context_size,
            output_seq=output_seq,
            last_n=args.last_n,
        )
    if teacher_exists:
        hidden_state_dataset = data.HiddenStateDataset(
            os.path.join(args.output_hidden_path, '{}.h5py'.format(stage)),
            output_seq=output_seq,
            last_n=args.last_n,
            is_GPT2=is_GPT2,
            predict_all_layers=args.predict_all_layers,
            predict_c=args.predict_c,
        )
        if not is_classification:
            n = min(len(text_dataset), len(hidden_state_dataset))
            text_dataset.start += len(text_dataset) - n
            hidden_state_dataset.start += len(hidden_state_dataset) - n
        dataset = data.ZipDataset(text_dataset, hidden_state_dataset)
    else:
        dataset = text_dataset
    return dataset

datasets = {
    stage: get_dataset(stage)
    for stage in ['train', 'valid', 'test']
}

from torch.utils.data._utils.collate import default_collate
if is_classification:
    collate_fn = data.text_classification_collate_fn
    if teacher_exists:
        collate_fn = data.zip_collate_fn(collate_fn, default_collate)
else:
    collate_fn = default_collate

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

if teacher_exists:
    if embedder is None and not args.new_embedder:
        embedder = teacher_embedder
    else:
        del teacher_embedder
    if output_layer is None and not args.new_output_layer:
        output_layer = teacher_output_layer

embedding_size = args.embedding_size if embedder is None else get_embedding_size(embedder)
hidden_size = args.hidden_size
output_size = datasets['train'].datasets[1].target_size if teacher_exists else args.output_size
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
            output_size=output_size if args.transform_output else None,
            input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
            output_dropout=args.output_dropout)
    elif args.model_type == 'transformer':
        config_model = get_config_model(args.config_model, vocab_size) if args.config_model is not None else None
        use_pretrained = args.pretrained_model_name is not None
        model = approx_models.Transformer_Approximator(
            hparams=config_model,
            output_seq=args.output_seq,
            bidirectional=args.bidirectional,
            input_size=embedding_size if args.input_transform else None,
            output_size=output_size if args.transform_output and not use_pretrained else None,
            remove_word_embedder=not use_pretrained,
            pretrained_model_name=args.pretrained_model_name,
        )
        if use_pretrained:
            embedder = model.word_embedder
            model.remove_word_embedder()

set_all_requires_grad(embedder.parameters(), not args.fix_embedder)
set_all_requires_grad(output_layer.parameters(), not args.fix_output_layer)

if copy_w is None and args.copy_w is not None:
    copy_w = nn.Parameter(torch.tensor(args.copy_w, requires_grad=True, device=device))

criterion = nn.MSELoss()
model.to(device)
embedder.to(device)
output_layer.to(device)
criterion.to(device)
if teacher_exists:
    teacher_criterion_fn = (lambda output, targets: cross_entropy(teacher_output_layer(output), targets)) if new_format_teacher else get_criterion_fn(teacher_model, teacher_criterion, is_GPT2)
    del teacher_model

def get_named_main_params():
    named_params = list(itertools.chain(model.named_parameters(), criterion.named_parameters()))
    return named_params

def get_named_params():
    named_params = get_named_main_params()
    if not args.fix_embedder:
        named_params.extend(embedder.named_parameters())
    if not args.fix_output_layer:
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
named_main_params = get_named_main_params()
main_params = [param for name, param in named_main_params]
total_main_params = sum(map(torch.Tensor.nelement, main_params))
print('Model total # main parameters:', total_main_params)


def get_prediction_and_loss(data_item, teacher_output=None, get_output=None):
    if is_classification:
        x, segment_ids, y = data_item.token_ids, data_item.segment_ids, data_item.label_id
    else:
        x, y = data_item
    _embedder_fn = lambda x: embedder(x).detach()
    input = (PackedSequence(_embedder_fn(x.data), **{key: getattr(x, key) for key in ('batch_sizes', 'sorted_indices', 'unsorted_indices')}) if isinstance(x, PackedSequence) else _embedder_fn(x))
    prediction_ = model(input, segment_ids=segment_ids) if is_classification else model(input)
    def get_last_n(t):
        if isinstance(model, (approx_models.LSTM_Approximator, approx_models.Transformer_Approximator)) and args.output_seq:
            if args.last_n is None or not model.training:
                t = t[:, -1]
            else:
                t = t[:, -args.last_n:]
        return t
    prediction = get_last_n(prediction_)

    if teacher_output is not None:
        teacher_logits = output_layer(get_output(teacher_output)).detach()

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
        gt_ce_loss = cross_entropy(logits, y)
        entropies = -(p * log_p).sum(-1)
        corrects = (p.argmax(-1) == y)
        if args.approx_logit:
            assert teacher_output is not None
            loss = F.mse_loss(logits, teacher_logits)
        else:
            T = args.approx_dist_temp
            L = args.approx_dist_lambda
            if L > 0.:
                assert teacher_output is not None
                s = 1.
                for d in logits.shape[1:-1]:
                    s *= d
                kl_loss = F.kl_div((logits / T).log_softmax(-1), (teacher_logits / T).softmax(-1), reduction='batchmean') / s
            loss = ((L * T * T) * kl_loss if L > 0. else 0.) + (1. - L) * gt_ce_loss
    else:
        assert teacher_output is not None
        loss = criterion(prediction, teacher_output)
        gt_ce_loss = teacher_criterion_fn(prediction, y)
        entropies, corrects = None, None

    return prediction, loss, gt_ce_loss, entropies, corrects


writer = SummaryWriter(os.path.join(args.ckpt, 'log'))


def evaluate(dataset=datasets['valid'], batch_size=args.valid_batch_size, prefix='valid', output_hidden_file=None):
    global global_step
    # Turn on evaluation mode which disables dropout.
    model.eval()
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    total_loss = 0.
    total_entropy = 0.
    total_correct = 0
    total_gt_ce_loss = 0.
    if args.eval_teacher_loss:
        total_teacher_loss = 0.

    if output_hidden_file is not None:
        m_output = f.create_dataset('output', (len(dataset), output_size), dtype='f')
        m = m_output

    if teacher_exists:
        get_output = dataset.datasets[1].get_output

    with torch.no_grad():
        t = tqdm(data_loader, desc=prefix, dynamic_ncols=True)
        n = 0
        for data_item in t:
            if args.cuda:
                data_item = map_structure(
                    lambda t: t.cuda(non_blocking=True),
                    data_item)
            if teacher_exists:
                text_data_item, teacher_output = data_item
            else:
                text_data_item = data_item
            y = text_data_item[-1]
            batch_size = len(y)
            n += batch_size
            prediction, loss, gt_ce_loss, entropies, corrects = get_prediction_and_loss(text_data_item, teacher_output, get_output) if teacher_exists else get_prediction_and_loss(text_data_item)
            total_loss += loss.item() * batch_size
            if entropies is not None:
                total_entropy += entropies.sum().item()
            if corrects is not None:
                total_correct += corrects.int().sum().item()
            total_gt_ce_loss += gt_ce_loss.item() * batch_size
            postfix = 'ppl={:.3f}'.format(
                math.exp(total_gt_ce_loss / n))
            if entropies is not None:
                postfix += ' ent={:.3f}'.format(
                    total_entropy / n)
            if corrects is not None:
                postfix += ' acc={:7.2%}'.format(
                    total_correct / n)
            if args.eval_teacher_loss:
                teacher_loss = teacher_criterion_fn(get_output(teacher_output), y)
                total_teacher_loss += teacher_loss * batch_size
                postfix += ' teacher_ppl={:.3f}'.format(
                    math.exp(total_teacher_loss / n))
            if copy_w is not None:
                postfix += ' copy_w={:.3f}'.format(copy_w.item())
            t.set_postfix_str(postfix)

            if output_hidden_file is not None:
                #TODO: allow last_n
                m[n - batch_size : n] = prediction.cpu()

    loss = total_loss / len(dataset)
    entropy = total_entropy / len(dataset) if entropies is not None else None
    gt_ce_loss = total_gt_ce_loss / len(dataset)
    acc = total_correct / len(dataset) if corrects is not None else None
    ppl = math.exp(gt_ce_loss)
    write_scalars = {
        'loss': loss,
        'gt_ce_loss': gt_ce_loss,
        'ppl': ppl,
    }
    pr = '{} loss={:.6f} gt_ce_loss={:6.2f} ppl={:6.2f}'.format(
        prefix, loss, gt_ce_loss, ppl)
    if entropy is not None:
        write_scalars['entropy'] = entropy
        pr += ' ent={:2.3f}'.format(entropy)
    if acc is not None:
        write_scalars['acc'] = acc
        pr += ' acc={:7.2%}'.format(acc)
    print(pr)
    if output_hidden_file is None:
        for name, value in write_scalars.items():
            writer.add_scalar('{}/{}'.format(prefix, name), value, global_step)

    return loss, gt_ce_loss, entropy, acc


if args.get_output_hidden:
    print('To get output and hidden states.')
    os.makedirs(args.output_hidden_path, exist_ok=True)
    for stage, dataset in datasets.items():
        save_path = os.path.join(args.output_hidden_path, '{}.h5py'.format(stage))
        with h5py.File(save_path, 'w') as f:
            evaluate(dataset=datasets[stage], batch_size=batch_sizes[stage], prefix=stage, output_hidden_file=f)
    sys.exit(0)


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
        collate_fn=collate_fn,
    )
    total_loss = 0.
    total_entropy = 0.
    total_correct = 0
    interval_loss = 0.
    interval_entropy = 0.
    interval_correct = 0
    interval_nelement = 0

    if teacher_exists:
        get_output = dataset.datasets[1].get_output

    t = tqdm(data_loader, desc='epoch {:3d}'.format(epoch), dynamic_ncols=True)
    for i_batch, data_item in enumerate(t):
        global_step += 1
        if args.cuda:
            data_item = map_structure(
                lambda t: t.cuda(non_blocking=True),
                data_item)
        if teacher_exists:
            text_data_item, teacher_output = data_item
        else:
            text_data_item = data_item
        y = text_data_item[-1]
        batch_size = len(y)
        optimizer.zero_grad()
        prediction, loss, gt_ce_loss, entropies, corrects = get_prediction_and_loss(text_data_item, teacher_output, get_output) if teacher_exists else get_prediction_and_loss(text_data_item)
        writer.add_scalar('{}/loss'.format(prefix), loss.item(), global_step)
        writer.add_scalar('{}/gt_ce_loss'.format(prefix), gt_ce_loss.item(), global_step)
        total_loss += loss.item() * batch_size
        interval_nelement += y.nelement()
        interval_loss += loss.item()
        if entropies is not None:
            entropy = entropies.sum()
            total_entropy += entropy.item()
            interval_entropy += entropy.item()
            writer.add_scalar('{}/entropy'.format(prefix), entropy.item() / entropies.nelement(), global_step)
        if corrects is not None:
            correct = corrects.int().sum()
            total_correct += correct.item()
            interval_correct += correct.item()
            writer.add_scalar('{}/acc'.format(prefix), correct.item() / corrects.nelement(), global_step)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.clip)
        writer.add_scalar('{}/grad_norm'.format(prefix), grad_norm, global_step)
        optimizer.step()
        writer.add_scalar('{}/lr'.format(prefix), optimizer.param_groups[0]['lr'], global_step)
        if copy_w is not None:
            writer.add_scalar('{}/copy_w'.format(prefix), copy_w.item(), global_step)

        if (i_batch + 1) % args.log_interval == 0:
            postfix = 'lr={:.1e} loss={:.2f}'.format(
                optimizer.param_groups[0]['lr'],
                interval_loss / args.log_interval,
            )
            if entropies is not None:
                postfix += ' ent={:.2f}'.format(interval_entropy / interval_nelement)
            if corrects is not None:
                postfix += ' acc={:7.2%}'.format(interval_correct / interval_nelement)
            if copy_w is not None:
                postfix += ' copy_w={:.3f}'.format(copy_w.item())
            t.set_postfix_str(postfix)
            interval_nelement = 0
            interval_loss = 0.
            interval_entropy = 0.
            interval_correct = 0

        if global_step % args.eval_interval == 0:
            valid_result = evaluate()
            update_valid_result(valid_result)
            model.train()

    loss = total_loss / len(dataset)
    entropy = total_entropy / len(dataset)
    print('train loss={:.6f} entropy={:.6f}'.format(
        loss, entropy))

def update_valid_result(valid_result, saving=True):
    loss, gt_ce_loss, entropy, acc = valid_result
    if saving and args.save_intermediate:
        model_save(os.path.join(args.ckpt, 'step{}.pt'.format(global_step)))
    ori_value = locals()[args.valid_metric]
    value = ori_value
    if args.valid_metric == 'acc':
        value = 1. - value
    global best_valid_value
    if best_valid_value is None or value < best_valid_value:
        print('new best validation {} {}'.format(args.valid_metric, ori_value))
        if saving:
            model_save(os.path.join(args.ckpt, 'best.pt'))
        best_valid_value = value
    lr_scheduler.step(value)

# Loop over epochs.
# At any point you can hit Ctrl + C to break out of training early.
try:
    best_valid_value = None
    valid_result = evaluate()
    update_valid_result(valid_result, saving=False)

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        valid_result = evaluate()
        update_valid_result(valid_result)

except KeyboardInterrupt:
    print('Exiting from training early')

# Load the best saved model.
best_model_path = os.path.join(args.ckpt, 'best.pt')
if not model_load(best_model_path):
    raise Exception('{} is not found'.format(best_model_path))

# Run on test data.
test_result = evaluate(datasets['test'], args.test_batch_size, prefix='test')

writer.close()
