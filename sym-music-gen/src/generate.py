import argparse
import transformers

from pathlib import Path


def __parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=Path)

    return parser.parse_args()


def main():
    args = __parse_args()

    model = transformers.AutoModelForCausalLM.from_pretrained(args.model).eval().cuda()


if __name__ == "__main__":
    main()
