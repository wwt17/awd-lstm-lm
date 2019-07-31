import torch
import torch.nn.functional as F

def batchify_context(data, batch_size):
    """Truncate corpus so remaining data can be split into batches evenly."""
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)

    print('Number of tokens after processing: %d' % data.size(0))

    return data

def get_context_batch(source, i, batch_size, seq_len):
    """
    For restricted context size, the hidden state is not copied across targets, where almost every token serves as a target. The amount of data used depends on the sequence length.
    Examples of (context, target) pairs for the corpus "The cat sat on the mat to play with yarn" and sequence length 5:
        ("The cat sat on the", "mat")
        ("cat sat on the mat", "to")
        ("sat on the mat to", "play")
        ...
    """

    data_ = []
    target_ = []
    for j in range(batch_size):
        start = i * batch_size + j
        end = start + seq_len
        data_.append(source[start:end])
        target_.append(source[start+1:end+1])

    # No training, so volatile always True
    # sequence length x batch size for consistency with Merity et al.
    data = torch.stack(data_, 1)
    target = torch.stack(target_, 1)

    # Since each example corresponds to 1 target, only the last row of the targets variable are relevant, but passing the whole tensor for complete info.
    return data, target

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

def get_perplexities_entropies(logits, target):
    log_softmaxed = F.log_softmax(logits, -1)
    softmaxed = F.softmax(logits, -1)
    perplexities = -torch.gather(log_softmaxed, -1, target.unsqueeze(-1)).squeeze(-1)
    entropies = -(softmaxed * log_softmaxed).sum(-1)
    return perplexities, entropies
