#could port to c++ later see if that works
#add decode later


import argparse
import json
from transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.2-1B"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text that is to be tokenized: ")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenizer.encode(args.text)
    ids2 = " ".join([str(i) for i in ids])
    print(ids2)


if __name__ == "__main__":
    main()














