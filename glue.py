import os
import numpy as np

from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import BertTokenizer

from data import TextAndLabel


def get_sequence_length_from_attention_mask(attention_mask):
    l = len(attention_mask)
    while l > 0 and attention_mask[l-1] == 0:
        l -= 1
    return l


def strip_padding(example):
    sequence_length = get_sequence_length_from_attention_mask(example.attention_mask)
    return TextAndLabel(
        np.array(example.input_ids[:sequence_length]),
        np.array(example.token_type_ids[:sequence_length]),
        example.label)


def get_n_classes(track):
    track = track.lower()
    processor = processors[track]()
    label_list = processor.get_labels()
    return len(label_list)


def get_glue(path, stage, vocab):
    track = os.path.basename(path)
    track = track.lower()
    processor = processors[track]()
    output_mode = output_modes[track]
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    label_list = processor.get_labels()
    stage_ = {
        'train': 'train',
        'valid': 'dev',
        'test': 'dev',
    }[stage]
    examples = getattr(processor, 'get_{}_examples'.format(stage_))(path)
    examples = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=512,
        output_mode=output_mode,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )
    examples = list(map(strip_padding, examples))
    return examples


def compute_glue_metrics(glue_task, logits, labels):
    glue_task = glue_task.lower()
    tasks = ('mnli', 'mnli-mm') if glue_task == 'mnli' else (glue_task,)
    results = {}
    for task in tasks:
        output_mode = output_modes[task]
        if output_mode == 'classification':
            preds = np.argmax(logits, axis=-1)
        elif output_mode == 'regression':
            preds = np.squeeze(logits, axis=-1)
        else:
            raise Exception('Unknown output_mode {}'.format(output_mode))
        results[task] = compute_metrics(task, preds, labels)
    ret = results[glue_task]
    if glue_task == 'mnli':
        for key, value in results['mnli-mm'].items():
            ret['{}-mm'.format(key)] = value
    return ret
