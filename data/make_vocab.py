import os
import argparse
from texar.data import make_vocab

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('dataset')
    argparser.add_argument('--max', type=int, default=-1)
    args = argparser.parse_args()
    dataset_dir = args.dataset
    max_vocab_size = args.max
    if not os.path.isdir(dataset_dir):
        raise ValueError("{} does not exist.".format(dataset_dir))
    train_file = os.path.join(dataset_dir, 'train.txt')
    words, counts = make_vocab(
        train_file,
        max_vocab_size=max_vocab_size,
        newline_token='<eos>',
        return_count=True,
    )
    tot = sum(counts)
    cnt = 0
    part = 0.
    for i, (word, count) in enumerate(reversed(list(zip(words, counts)))):
        cnt += count
        if cnt / tot >= part:
            print('{:.2f}'.format(cnt / tot), len(counts) - i, word, count, cnt)
            part += 0.01
    with open(os.path.join(dataset_dir, 'vocab.txt'), 'w') as vocab_file:
        for word in words:
            if word not in ['<eos>', '<unk>']:
                print(word, file=vocab_file)
