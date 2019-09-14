import pickle
import argparse
import random
import numpy as np
import math
import torch
import numpy as np
from typing import NamedTuple, List, Tuple, Union
import data
from data import FixedLengthContextDataset

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
    with open(fname, 'rb') as f:
        items = []
        while True:
            try:
                item = pickle.load(f)
            except EOFError:
                break
            items.append(item)
    return Result(fname, *items)

def load_pos_file(fname):
    with open(fname, 'rb') as f:
        items = []
        while True:
            try:
                item = pickle.load(f)
            except EOFError:
                break
            items.append(item)
    assert len(items) == 1
    return items[0]

if __name__ == '__main__':
    _input = input
    argparser = argparse.ArgumentParser()
    argparser.add_argument('op', type=str, choices=['pos_analysis', 'plot', 'plot_loss', 'view', 'compare', 'compare_inverse'])
    argparser.add_argument('files', type=str, nargs='+')
    argparser.add_argument('--pos_file', type=str, default=None)
    argparser.add_argument('--data', type=str, default='data/wikitext-103')
    argparser.add_argument('--max_seq_len', type=int, default=1000)
    argparser.add_argument('--seq_len', type=int, default=512)
    argparser.add_argument('--context_sizes', nargs='+', type=int)
    argparser.add_argument('--random', action='store_true')
    argparser.add_argument('--reverse', action='store_false')
    argparser.add_argument('--gap', type=float, default=-math.inf)
    args = argparser.parse_args()

    results = list(map(read_file, args.files))
    n = results[0].n
    assert all(result.n == n for result in results), ' '.join(str(result.n) for result in results)

    if args.op == 'pos_analysis':
        ptarget_tokens = load_pos_file(args.pos_file)
        for result in results:
            pos_stat = {}
            total_loss = 0.
            for i in range(result.n):
                item = result.geti(i)
                ptarget_token = ptarget_tokens[i]
                try:
                    w, p = ptarget_token.split('_')
                except:
                    print('ptaraget_token: {}'.format(ptarget_token))
                    raise
                if p not in pos_stat:
                    pos_stat[p] = []
                pos_stat[p].append(item)
                total_loss += item.loss
            mean_loss = total_loss / result.n
            res = []
            for p, stat in sorted(list(pos_stat.items()), key=lambda x: len(x[1]), reverse=True):
                N = len(stat)
                sum_loss, sum_entropy = map(sum, list(zip(*stat))[:2])
                loss, entropy = sum_loss / N, sum_entropy / N
                res.append((p, N, sum_loss - mean_loss * N, loss, math.exp(loss), entropy))
            res.sort(key=lambda x: x[2])
            for r in res:
                print('POS: {:<5s} N: {:>6d} weight: {:>10.3f} loss: {:>6.3f} ppl: {:>8.3f} entropy: {:>6.3f}'.format(*r))
    if args.op.startswith('plot'):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('agg')
        if args.op == 'plot':
            for result in results:
                plt.title(result.name)
                plt.scatter(result.all_losses, result.all_entropies, s=.01)
                plt.xlim(0, 20)
                plt.ylim(0, 8.5)
                plt.show()
        elif args.op == 'plot_loss':
            assert len(results) == 2
            plt.title('{} vs. {}'.format(results[0].name, results[1].name))
            plt.scatter(results[0].all_losses, results[1].all_losses, s=.01)
            plt.show()
        else:
            raise NotImplementedError('Ignored op {}'.format(args.op))
    elif args.op.startswith('compare') or args.op.startswith('view'):
        context_sizes = args.context_sizes
        context_sizes.sort(reverse=True)
        corpus = data.prepare_corpus(args.data)
        id_to_token_map_py = corpus.vocab.id_to_token_map_py
        stages = []
        for result in results:
            for stage in ['valid', 'test']:
                if stage in result.name:
                    break
            else:
                stage = None
            stages.append(stage)
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
