import argparse
import miditok
import transformers
import json

from pathlib import Path


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=Path)
    parser.add_argument("prompt", type=Path)

    return parser.parse_args()


def main():
    args = __parse_args()

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model).eval().cuda()

    tokenizer = miditok.REMI.from_pretrained(args.model)

    with args.prompt.open() as io:
        prompt = json.load(io)

    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
