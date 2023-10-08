from transformers import AutoTokenizer

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    return _tokenizer


SPLITS = ("train", "validation", "test")
