#could port to c++ later see if that works

import argparse
import json
from Transformers import AutoTokenizer

MODEL = "meta-llama/Llama-3.2-1B"

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenizer.encode("Hello, world!")
    print(ids)


if __name__ == "__main__":
    main()














