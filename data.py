import os
import torch
from torch.utils.data import Dataset
import h5py

from collections import Counter

from utils import map_structure


eos_token = '<eos>'

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    @property
    def eos_id(self):
        return self.word2idx[eos_token]


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


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
