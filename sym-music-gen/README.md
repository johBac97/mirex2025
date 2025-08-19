# MIREX 2025 Symbolic Music Generation 

Repo for developing a model for the MIREX 2025 symbolic music generation competition.

# Instructions for generating MIREX 2025 Submission

## 1. Install uv

On linux/mac run the following command

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If `curl` is unavailable run, 

```
wget -qO- https://astral.sh/uv/install.sh | sh
```

## 2. Retrieve model weights 

Get the model weights from..... (TO BE DECIDED).

## 3. Run Generation

```
./generation.sh <INPUT-JSON-PROMPT> <GENERATIONS-OUTPUT> <NUMBER-GENERATIONS>
```


## Installation 

I recommend using `uv` for package management. Install it with;

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then to install the environment simply run;

```
uv sync
```

At this point you can either activate the created environment (`source .venv/bin/activate`) or prefix commands with `uv run ...`.

There is also a pip `requirements.txt` file.

## Train

To train the model run the following command;

```
uv run src/train.py \
    --train-config <PATH-TO-TRAIN-CONFIG> \
    --model <MODEL> \
    --train-data <TRAIN-DATASET> \
    --val-data <VAL-DATASET> 
```


where `<PATH-TO-TRAIN-CONFIG>` is a json file that will be passed to the `TrainingArguments`, `<MODEL>` is the huggingface model that will be used for the training, `<TRAIN-DATASET>` and `<VAL-DATASET>` is the path to the train and validation datasets.


## Generation

To create completions from a prompt in json format run the following command,

```
uv run src/generate.py <MODEL> <PROMPT> --num-generations <NUMBER-GENERATIONS>
```

Where the `<MODEL>` is the model checkpoint to use, loads both the model and the REMI tokenizer from this path, the `<PROMPT>` is the path to the song prompt in json format.

