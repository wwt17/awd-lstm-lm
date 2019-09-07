# Investigating LSTM and Transformer

This repository is forked from [this repository](https://github.com/salesforce/awd-lstm-lm) and contains some code from [this repository](https://github.com/urvashik/lm-context-analysis).

## Software Requirements

Python 3 and PyTorch 0.4 are required for the current codebase.

See [requirements.txt](requirements.txt) for requirements of Python packages.

## Experiments

For data setup, see [data/Makefile](data/Makefile) for how to download and process the datasets needed, and run `make ${DATASET}` command under the `data` directory.

### Word level WikiText-103 (WT103) with LSTM

+ `python main.py --data data/wikitext-103 --model LSTM --nlayers 3 --emsize 200 --nhid 1000 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 140 --batch_size 30 --optimizer adam --lr 1e-4 --ppl_gap -10 --save WT103.LSTM.pt --eval_entropy`

### Word level WikiText-103 (WT103) with GPT2

+ `python main.py --data data/wikitext-103 --model GPT2 --config_model config_model --bptt 512 --batch_size 6 --optimizer adam --lr 0.5 --ppl_gap 0. --save WT103.GPT2.pt --eval_entropy`
