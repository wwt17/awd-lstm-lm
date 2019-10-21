import os
import numpy as np

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
    from transformers import glue_processors as processors
    track = track.lower()
    processor = processors[track]()
    label_list = processor.get_labels()
    return len(label_list)


def get_glue(path, stage, vocab):
    track = os.path.basename(path)
    track = track.lower()
    from transformers import glue_processors as processors
    processor = processors[track]()
    from transformers import glue_output_modes as output_modes
    output_mode = output_modes[track]
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    label_list = processor.get_labels()
    stage_ = {
        'train': 'train',
        'valid': 'dev',
        'test': 'dev',
    }[stage]
    examples = getattr(processor, 'get_{}_examples'.format(stage_))(path)
    from transformers import glue_convert_examples_to_features as convert_examples_to_features
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
