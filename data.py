import os
import traceback
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
    def __init__(self, path, get_fn, with_pos=False):
        special_token_fn = (lambda w: w + '_' + w) if with_pos else (lambda w: w)
        vocab_filename = os.path.join(path, 'vocab.txt')
        self.vocab = Vocab(
            vocab_filename,
            **{s: special_token_fn(globals()[s]) for s in
               ['pad_token', 'bos_token', 'eos_token', 'unk_token']}
        )
        # Set designated speial tokens in some cases
        if path.endswith('_bpe') or 'Bert' in path:
            with open(vocab_filename, 'r') as vocab_file:
                vocab = list(line.strip() for line in vocab_file)
            if path.endswith('_bpe'):
                special_token = vocab[-1]
                special_tokens = {key: special_token for key in ['pad', 'bos', 'eos', 'unk']}
            elif 'Bert' in path:
                special_tokens = {
                    'pad': '[PAD]',
                    'bos': '[CLS]',
                    'eos': '[SEP]',
                    'unk': '[UNK]',
                }
            for key, token in special_tokens.items():
                setattr(self.vocab, '_{}_token'.format(key), token)
            self.vocab._id_to_token_map_py = dict(zip(range(len(vocab)), vocab))
            self.vocab._token_to_id_map_py = dict(zip(vocab, range(len(vocab))))

        for stage in ['train', 'valid', 'test']:
            try:
                setattr(self, stage, get_fn(path, stage, self.vocab))
            except:
                traceback.print_exc()
                print('no {} set'.format(stage))
                continue


def get_holistic_text(path, stage, vocab):
    text_file_path = os.path.join(path, '{}.txt'.format(stage))
    assert os.path.exists(text_file_path)

    """Tokenizes a text file."""
    eos_token = vocab.eos_token

    # Add words to the dictionary
    with open(text_file_path, 'r') as f:
        tokens = 0
        for line in f:
            words = line.split() + [eos_token]
            tokens += len(words)

    # Tokenize file content
    ids = torch.LongTensor(tokens)
    with open(text_file_path, 'r') as f:
        token = 0
        for line in f:
            words = line.split() + [eos_token]
            l = len(words)
            ids[token : token + l] = torch.from_numpy(
                vocab.map_tokens_to_ids_py(words))
            token += l

    return ids


def get_review_and_star(path, stage, vocab):
    review_tokens_file_name = '{}.review_tokens.txt'.format(stage)
    star_file_name = '{}.star.txt'.format(stage)
    examples = []
    with open(os.path.join(path, review_tokens_file_name), 'r') as review_tokens_file, \
         open(os.path.join(path, star_file_name), 'r') as star_file:
        for review_tokens, star in zip(map(str.split, review_tokens_file), map(int, star_file)):
            review_token_ids = vocab.map_tokens_to_ids_py([bos_token] + review_tokens + [eos_token])
            examples.append((review_token_ids, star))
    return examples


def prepare_corpus(data_name, get_fn, with_pos=False):
    import hashlib
    fn = 'corpus.{}.data'.format(hashlib.md5(data_name.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = Corpus(data_name, get_fn, with_pos=with_pos)
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
            line = line.strip().split() + ['<eos>_<eos>']
            for word_pair in line:
                w, p = word_pair.split('_')
                if p not in pos_:
                    pos_[p] = {}
                token_id = vocab.token_to_id_map_py[w]
                pos_[p][token_id] = token_id

    for tag in pos_:
        # sort dictionary by rank and throw away the frequencies
        pos_[tag] = sorted(pos_[tag], key=pos_[tag].get)

    return pos_


class ZipDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        lengths = list(map(len, self.datasets))
        n = lengths[0]
        assert all(length == n for length in lengths)
        return n

    def __getitem__(self, i):
        return tuple(dataset[i] for dataset in self.datasets)


def zip_collate_fn(*collate_fns):
    def fn(examples):
        return tuple(collate_fn(part) for collate_fn, part in zip(collate_fns, zip(*examples)))


class FixedLengthContextDataset(Dataset):
    def __init__(self, seq, context_size, output_seq=True, last_n=None):
        self.seq = seq
        self.context_size = context_size
        self.output_seq = output_seq
        self.last_n = last_n
        self.start = self.context_size

    def __len__(self):
        return len(self.seq) - self.start

    def __getitem__(self, i):
        i += self.start
        x = self.seq[i - self.context_size : i]
        if self.output_seq:
            y = self.seq[i + 1 - (self.last_n if self.last_n is not None else self.context_size) : i + 1]
        else:
            y = self.seq[i]
        return x, y


class HiddenStateDataset(Dataset):
    def __init__(self, hidden_state_h5py, output_seq=True, last_n=None, is_GPT2=False, predict_all_layers=False, predict_c=False):
        self.output_seq = output_seq
        if not self.output_seq:
            last_n = 1
        self.last_n = last_n
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
        self.n = len(self.states[0])
        assert all(len(s) == self.n for s in self.states)
        self.start = self.last_n - 1

    def __len__(self):
        return self.n - self.start

    def __getitem__(self, i):
        i += self.start
        out = torch.cat([torch.tensor(h[i - self.last_n + 1 : i + 1]) for h in self.states], -1)
        if not self.is_GPT2:
            out = out.squeeze(-2)
        if not self.output_seq:
            out = out.squeeze(0)
        return out

    @property
    def target_size(self):
        return sum(h.shape[-1] for h in self.states)

    def get_output(self, hidden_state):
        l = self.states[-1].shape[-1]
        return hidden_state.narrow(-1, -l, l)


class TextClassificationDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def text_classification_collate_fn(examples):
    reviews, stars = zip(*examples)
    reviews = torch.nn.utils.rnn.pack_sequence(list(map(torch.from_numpy, reviews)), enforce_sorted=False)
    stars = torch.tensor(stars)
    return reviews, stars
