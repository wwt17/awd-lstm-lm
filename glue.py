import os
import json
import functools


def process_texts_and_label(example, vocab, tokenize, text_keys, label_key, labels):
    tokens = [vocab.bos_token]
    for text_key in text_keys:
        text = example[text_key]
        text_tokens = tokenize(example[text_key])
        tokens.extend(text_tokens)
        tokens.append(vocab.eos_token)

    token_ids = vocab.map_tokens_to_ids_py(tokens)

    label = example[label_key]
    label_id = labels.index(label)

    return (token_ids, label_id)


process_BoolQ = functools.partial(
    process_texts_and_label,
    text_keys=["question", "passage"],
    label_key="label",
    labels=[False, True],
)
process_CB = functools.partial(
    process_texts_and_label,
    text_keys=["premise", "hypothesis"],
    label_key="label",
    labels=["entailment", "contradiction", "neutral"],
)
process_RTE = functools.partial(
    process_texts_and_label,
    text_keys=["premise", "hypothesis"],
    label_key="label",
    labels=["entailment", "not_entailment"],
)

processors = {
    'BoolQ': process_BoolQ,
    'CB': process_CB,
    'RTE': process_RTE,
    #TODO 'WiC': process_WiC,
}


def get_n_classes(track):
    return len(processors[track].keywords['labels'])


def get_glue(path, stage, vocab):
    track = os.path.basename(path)
    assert not track.startswith('AX-') #TODO
    stage_ = {
        'train': 'train',
        'valid': 'val',
        'test': 'val',
    }[stage]
    from tokenizer import get_tokenizer
    tokenize = get_tokenizer('transformers')
    with open(os.path.join(path, '{}.jsonl'.format(stage_)), 'r') as jsonl_file:
        examples = map(json.loads, jsonl_file)
        examples = map(
            functools.partial(
                processors[track],
                vocab=vocab,
                tokenize=tokenize),
            examples)
        examples = list(examples)
    return examples
