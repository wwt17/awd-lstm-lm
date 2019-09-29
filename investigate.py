import os
import pickle
import argparse
import random
import numpy as np
import math
import torch
import numpy as np
from typing import NamedTuple, List, Tuple, Union
from collections import defaultdict
import data
from data import FixedLengthContextDataset

class Limits:
    def __init__(self, l=+math.inf, h=-math.inf):
        self.l = l
        self.h = h
    def update(self, values):
        for value in values:
            self.l = min(self.l, value)
            self.h = max(self.h, value)

class Data(NamedTuple):
    loss: float
    entropy: float
    topk: Tuple[np.array, np.array]
    def __str__(self):
        return \
        'Data(\n'\
        ' loss: {loss:.6f}\n'\
        ' perplexity: {perplexity:.6f}\n'\
        ' entropy: {entropy:.6f}\n'\
        ' topk:\n{topk}\n'\
        ')'.format(
            loss=self.loss,
            perplexity=math.exp(self.loss),
            entropy=self.entropy,
            topk='\n'.join('{:.4f}: {}'.format(prob, id_to_token_map_py[int(idx)]) for prob, idx in zip(*self.topk)),
        )

def inversed(a: Data, b: Data) -> bool:
    return (a.loss < b.loss and a.entropy > b.entropy) or (a.loss > b.loss and a.entropy < b.entropy)

class Result(NamedTuple):
    name: str
    tag: Union[str, int]
    mean_loss: float
    mean_entropy: float
    all_losses: List[float]
    all_entropies: List[float]
    all_topk: List[Tuple[np.array, np.array]]
    @property
    def n(self):
        N = len(self.all_losses)
        assert len(self.all_entropies) == N
        assert len(self.all_topk) == N
        return N
    def geti(self, i):
        return Data(self.all_losses[i], self.all_entropies[i], self.all_topk[i])

def mean(a):
    return sum(a) / len(a)

def read_file(fname):
    print('reading {}'.format(fname))
    with open(fname, 'rb') as f:
        items = []
        while True:
            try:
                item = pickle.load(f)
            except EOFError:
                break
            items.append(item)
    return Result(fname, *items)

def split_pos_seq(ptarget_token_ids, id_to_token_map_py):
    def split_pos_token_id(ptarget_token_id):
        ptarget_token = id_to_token_map_py[int(ptarget_token_id)]
        if ptarget_token in ['<pad>', '<bos>', '<eos>', '<unk>']:
            w, p = ptarget_token, ptarget_token
        else:
            try:
                w, p = ptarget_token.split('_')
            except:
                print('ptaraget_token: {}'.format(ptarget_token))
                raise
        return w, p

    return list(map(split_pos_token_id, ptarget_token_ids))

if __name__ == '__main__':
    _input = input
    argparser = argparse.ArgumentParser()
    argparser.add_argument('op', type=str, choices=['pos_analysis', 'plot', 'plot_loss', 'view', 'compare', 'compare_inverse'])
    argparser.add_argument('files', type=str, nargs='+')
    argparser.add_argument('--data', type=str, default='data/wikitext-103')
    argparser.add_argument('--pos_data', type=str, default='data/wikitext-103_pos')
    argparser.add_argument('--max_seq_len', type=int, default=1000)
    argparser.add_argument('--seq_len', type=int, default=512)
    argparser.add_argument('--context_sizes', nargs='+', type=int)
    argparser.add_argument('--random', action='store_true')
    argparser.add_argument('--reverse', action='store_false')
    argparser.add_argument('--gap', type=float, default=-math.inf)
    argparser.add_argument('--relative', action='store_true')
    argparser.add_argument('--plot', nargs='+', choices=['sorted', 'pdf', 'cdf'], default=[])
    argparser.add_argument('--plot_sample_gap', type=int, nargs='+', default=None)
    argparser.add_argument('--n_bins', type=int, default=100)
    argparser.add_argument('--scatter_size', type=float, default=.01)
    argparser.add_argument('--fig_size', type=float, nargs=2, default=None)
    argparser.add_argument('--save_fig', type=str)
    args = argparser.parse_args()

    results = list(map(read_file, args.files))
    if args.plot_sample_gap is not None:
        assert len(results) == len(args.plot_sample_gap)
    else:
        args.plot_sample_gap = [1] * len(results)
    n = results[0].n
    assert all(result.n == n for result in results), '\n'.join('{}:\t{}'.format(result.name, result.n) for result in results)

    stages = []
    for result in results:
        for stage in ['train', 'valid', 'test']:
            if stage in result.name:
                break
        else:
            stage = None
        stages.append(stage)

    if args.op == 'pos_analysis' or args.op.startswith('plot'):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('agg')

        if args.op == 'pos_analysis':
            corpus = data.prepare_corpus(args.pos_data, data.get_holistic_text)
            id_to_token_map_py = corpus.vocab.id_to_token_map_py
            seq = split_pos_seq(getattr(corpus, stage), id_to_token_map_py)
            all_res = defaultdict(lambda: defaultdict(dict))
            cnt_pos = {}
            for result in results:
                print('{}:'.format(result.name))
                dir_name = os.path.basename(os.path.dirname(result.name))
                context_size = int(result.name.split('.')[-1])
                pos_stat = defaultdict(list)
                total_loss = 0.
                for i, (w, p) in enumerate(seq[-result.n:]):
                    item = result.geti(i)
                    pos_stat[p].append(item)
                    total_loss += item.loss
                mean_loss = total_loss / result.n
                res = []
                for p, stat in pos_stat.items():
                    N = len(stat)
                    sum_loss, sum_entropy = map(sum, list(zip(*stat))[:2])
                    loss, entropy = sum_loss / N, sum_entropy / N
                    res.append((p, N, sum_loss - mean_loss * N, loss, entropy))
                res.sort(key=lambda x: x[2])
                for r in res:
                    pos, N, weight, loss, entropy = r
                    print('POS: {:<5s} N: {:>6d} weight: {:>10.3f} loss: {:>6.3f} ppl: {:>8.3f} entropy: {:>6.3f}'.format(
                        pos, N, weight, loss, math.exp(loss), entropy))
                    cnt_pos[pos] = N
                    all_res[pos][dir_name][context_size] = (loss, entropy)
            plt.title('Different POS loss')
            plt.xlabel('context size')
            plt.ylabel('loss')
            context_limits = Limits(0)
            loss_limits = Limits()
            pos_cnt = 0
            colors = ['black', 'purple', 'blue', 'cyan', 'green', 'yellow', 'darkorange', 'red', 'magenta', 'brown']
            linestyles = ['-', '--', '-.', ':']
            all_dir_names = {}
            for pos, pos_content in all_res.items():
                if cnt_pos[pos] >= 8000:
                    for dir_name, a in sorted(list(pos_content.items())):
                        if dir_name not in all_dir_names:
                            all_dir_names[dir_name] = len(all_dir_names)
                        a = list(a.items())
                        a.sort(key=lambda x: x[0])
                        context_sizes, items = zip(*a)
                        losses, entropies = zip(*items)
                        y = [loss - losses[-1] for loss in losses] if args.relative else losses
                        loss_limits.update(y)
                        context_limits.update(context_sizes)
                        plt.plot(context_sizes, y, label='{}/{}'.format(dir_name, pos), color=colors[pos_cnt], linestyle=linestyles[all_dir_names[dir_name]])
                    pos_cnt += 1
            plt.xlim(context_limits.l, context_limits.h)
            plt.ylim(loss_limits.l, loss_limits.h)
            plt.legend()

        elif args.op == 'plot':
            ax_rows = len(results)
            ax_columns = 1 + 2 * len(args.plot)
            ax_id = 0

            for result_i, result in enumerate(results):
                sample_gap = args.plot_sample_gap[result_i]
                def gap_sample(a):
                    return a[::sample_gap]

                ax_id += 1
                ax = plt.subplot(ax_rows, ax_columns, ax_id)
                ax.set_title(result.name)
                ax.set_xlabel('loss')
                ax.set_ylabel('entropy')
                ax.scatter(*map(gap_sample, (result.all_losses, result.all_entropies)), s=args.scatter_size)
                ax.set_xlim(0, 20)
                ax.set_ylim(0, 8.5)

                for attr, y in (('loss', result.all_losses), ('entropy', result.all_entropies)):
                    for plot_type in args.plot:
                        title = '{} {}'.format(attr, plot_type)
                        print('plotting {}'.format(title))
                        ax_id += 1
                        ax = plt.subplot(ax_rows, ax_columns, ax_id)
                        ax.set_title(title)

                        if plot_type == 'sorted':
                            ax.set_xlabel('example')
                            ax.set_ylabel(attr)
                            x = np.arange(result.n)
                            ax.bar(*map(gap_sample, (x, sorted(y))), width=sample_gap, align='edge')
                            ax.set_xlim(0, result.n)

                        elif plot_type in ['pdf', 'cdf']:
                            ax.set_xlabel(attr)
                            ax.set_ylabel('example #')
                            ax.hist(y, bins=args.n_bins, cumulative=(plot_type == 'cdf'))

                        else:
                            raise NotImplementedError(plot_type)

        elif args.op == 'plot_loss':
            assert len(results) == 2
            plt.title('{} vs. {}'.format(results[0].name, results[1].name))
            plt.xlabel('{} loss'.format(results[0].name))
            plt.ylabel('{} loss'.format(results[1].name))
            plt.scatter(results[0].all_losses, results[1].all_losses, s=args.scatter_size)

        else:
            raise NotImplementedError('Ignored op {}'.format(args.op))

        fig = plt.gcf()
        if args.fig_size is not None:
            fig.set_size_inches(*args.fig_size)

        if args.save_fig is not None:
            plt.savefig(args.save_fig)
        else:
            plt.show()

    elif args.op.startswith('compare') or args.op.startswith('view'):
        context_sizes = args.context_sizes
        context_sizes.sort(reverse=True)
        corpus = data.prepare_corpus(args.data, data.get_holistic_text)
        id_to_token_map_py = corpus.vocab.id_to_token_map_py
        seq = getattr(corpus, stage)
        dataset = FixedLengthContextDataset(seq[args.max_seq_len - args.seq_len :], args.seq_len)
        # assert len(dataset) == n, 'len(dataset)={} n={}'.format(len(dataset), n)

        if args.op == 'compare_inverse':
            indices = []
            for i in range(n):
                if inversed(results[0].geti(i), results[1].geti(i)):
                    indices.append(i)
            def inversed_value(i):
                a, b = results[0].geti(i), results[1].geti(i)
                return abs(a.entropy - b.entropy) + 0.2 * abs(a.loss - b.loss)
            key = inversed_value

        elif args.op == 'compare':
            def better_value(i):
                a, b = results[0].geti(i), results[1].geti(i)
                return a.loss - b.loss
            indices = list(filter(lambda i: better_value(i) > args.gap, range(n)))
            key = better_value

        elif args.op == 'view':
            key = lambda i: i
            indices = list(range(n))

        else:
            raise NotImplementedError('Ignored op {}'.format(args.op))

        if args.random:
            random.shuffle(indices)
        else:
            indices.sort(key=key, reverse=args.reverse)

        if len(results) >= 2:
            n_loss_leq = len(list(filter(lambda i: results[0].geti(i).loss <= results[1].geti(i).loss, indices)))
            print('total: {} | #(a.loss <= b.loss): {} | #(a.loss > b.loss): {}'.format(len(indices), n_loss_leq, len(indices) - n_loss_leq))

        for I in indices:
            input, target = dataset[I]
            input, target = (x.tolist() for x in (input, target))
            input_tokens, target_tokens = ([id_to_token_map_py[int(idx)] for idx in x] for x in (input, target))
            print('input:')
            for i in range(len(context_sizes)):
                print(' '.join(input_tokens[-context_sizes[i]:-context_sizes[i+1]] if i < len(context_sizes) - 1 else input_tokens[-context_sizes[i]:]))
                print('=' * 10)
            print('target: {}'.format(target_tokens[-1]))
            print('occured:', end='')
            for i, idx in enumerate(input):
                if idx == target[-1]:
                    print(' {}'.format(len(input) - i), end='')
            print()
            for result in results:
                print(result.geti(I))
            _input()

    else:
        raise NotImplementedError('Ignored op {}'.format(args.op))
