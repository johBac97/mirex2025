# MIREX 2025 Symbolic Music Generation 

Repo for developing a model for the MIREX 2025 symbolic music generation competition.


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
uv run src/generate.py <MODEL> <PROMPT> 
```

Where the `<MODEL>` is the model checkpoint to use, loads both the model and the REMI tokenizer from this path, the `<PROMPT>` is the path to the song prompt in json format.

