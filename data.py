import os
import torch
from torch.utils.data import Dataset
from texar.data import Vocab
import h5py

from collections import Counter

from utils import map_structure


pad_token = '<pad>'
bos_token = '<bos>'
eos_token = '<eos>'
unk_token = '<unk>'


class Corpus(object):
    def __init__(self, path):
        self.vocab = Vocab(
            os.path.join(path, 'vocab.txt'),
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
        )
        for stage in ['train', 'valid', 'test']:
            setattr(self, stage, self.tokenize(os.path.join(path, '{}.txt'.format(stage))))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)

        # Tokenize file content
        ids = torch.LongTensor(tokens)
        with open(path, 'r') as f:
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                l = len(words)
                ids[token : token + l] = torch.from_numpy(
                    self.vocab.map_tokens_to_ids_py(words))
                token += l

        return ids


def prepare_corpus(data_name):
    import hashlib
    fn = 'corpus.{}.data'.format(hashlib.md5(data_name.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = Corpus(data_name)
        torch.save(corpus, fn)
    return corpus


def get_vocab_all_pos(pos_datafile, vocab):
    """
    Generate a map.
    Keys = POS tag
    Values = a list of words with that POS tag, sorted by frequency
    """
    pos_ = {}
    with open(pos_datafile, 'r') as f:
        for line in f:
            line = line.strip().split(' ') + ['<eos>_<eos>'] if len(line.strip()) > 0 else ['<eos>_<eos>']
            for word_pair in line:
                w, p = word_pair.split('_')
                if p not in pos_:
                    pos_[p] = {}
                token_id = vocab.token_to_id_map_py[w]
                pos_[p][token_id] = vocab.counter[token_id]

    for tag in pos_:
        # sort dictionary by rank and throw away the frequencies
        pos_[tag] = sorted(pos_[tag], key=pos_[tag].get)

    return pos_


class FixedLengthContextDataset(Dataset):
    def __init__(self, seq, context_size):
        self.seq = seq
        self.context_size = context_size

    def __len__(self):
        return len(self.seq) - self.context_size

    def __getitem__(self, i):
        x = self.seq[i : i + self.context_size]
        y = self.seq[i + 1 : i + 1 + self.context_size]
        return x, y


class HiddenStateDataset(Dataset):
    def __init__(self, seq, hidden_state_h5py, context_size, last_n=None, is_GPT2=False, predict_all_layers=False, predict_c=False):
        self.context_size = context_size
        if last_n is None:
            self.no_last_n = True
            self.last_n = 1
        else:
            self.no_last_n = False
            self.last_n = last_n
        self.seq = seq
        self.hidden_state_h5py = h5py.File(hidden_state_h5py, 'r')
        self.output = self.hidden_state_h5py['output']
        self.is_GPT2 = is_GPT2
        if is_GPT2:
            self.states = [self.output]
        else:
            grp_hidden = self.hidden_state_h5py['hidden']
            self.hidden = []
            for l in range(len(grp_hidden)):
                grp_layer = grp_hidden[str(l)]
                self.hidden.append((grp_layer['h'], grp_layer['c']))
            self.states = []
            for h, c in (self.hidden if predict_all_layers else self.hidden[-1:]):
                if predict_c:
                    self.states.append(c)
                self.states.append(h)

    def __len__(self):
        return len(self.seq) - self.context_size - (self.last_n - 1)

    def __getitem__(self, i):
        i += self.last_n - 1
        x = self.seq[i : self.context_size + i]
        y = self.seq[self.context_size + i - self.last_n + 1 : self.context_size + i + 1]
        out = torch.cat([torch.tensor(h[i - self.last_n + 1 : i + 1]) for h in self.states], -1)
        if not self.is_GPT2:
            out = out.squeeze(-2)
        if self.no_last_n:
            y = y.squeeze(0)
            out = out.squeeze(0)
        return x, y, out

    @property
    def target_size(self):
        return sum(h.shape[-1] for h in self.states)

    def get_output(self, hidden_state):
        l = self.states[-1].shape[-1]
        return hidden_state.narrow(-1, -l, l)
