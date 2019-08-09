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


def get_vocab_all_pos(pos_datafile, corpus_dict):
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
                token_id = corpus_dict.word2idx[w]
                pos_[p][token_id] = corpus_dict.counter[token_id]

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
    def __init__(self, seq, hidden_state_h5py, context_size, pad_id, predict_all_layers, predict_c):
        self.context_size = context_size
        self.seq = torch.cat([torch.full((context_size - 1,), pad_id, dtype=torch.long), seq])
        self.hidden_state_h5py = h5py.File(hidden_state_h5py, 'r')
        self.output = self.hidden_state_h5py['output']
        grp_hidden = self.hidden_state_h5py['hidden']
        self.hidden = []
        for l in range(len(grp_hidden)):
            grp_layer = grp_hidden[str(l)]
            self.hidden.append((grp_layer['h'], grp_layer['c']))
        self.states = (self.output, self.hidden)
        self.predict_all_layers = predict_all_layers
        self.predict_c = predict_c

    def __len__(self):
        return self.output.shape[0] - 1

    def __getitem__(self, i):
        x = self.seq[i : self.context_size + i]
        y = self.seq[self.context_size + i]
        target = self.hidden if self.predict_all_layers else self.hidden[-1:]
        if not self.predict_c:
            target = [(h,) for h, c in target]
        target = torch.cat([torch.cat([
            torch.tensor(b[i]) for b in a], -1) for a in target], -1).squeeze(0).squeeze(0)
        return x, y, target

    @property
    def target_size(self):
        s = self.hidden if self.predict_all_layers else self.hidden[-1:]
        if self.predict_c:
            s = [h.shape[-1] + c.shape[-1] for h, c in s]
        else:
            s = [h.shape[-1] for h, c in s]
        return sum(s)

    def get_output(self, hidden_state):
        h, c = self.hidden[-1]
        h, c = map_structure(lambda x: x.shape[-1], (h, c))
        if self.predict_c:
            return hidden_state[:, -h-c : -c]
        else:
            return hidden_state[:, -h:]
