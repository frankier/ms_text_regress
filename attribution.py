import os
import sys

import numpy
import torch
from captum.attr import LayerIntegratedGradients
from nltk.corpus import sentiwordnet as swn

from ms_text_regress.baseline_models.regression import (
    BertForMultiScaleSequenceRegression,
)
from ms_text_regress.datasets import load_from_disk_with_labels
from ms_text_regress.scripts.utils import get_tokenizer

checkpoint = sys.argv[1]
dataset_path = sys.argv[2]
split = sys.argv[3]
outfn = sys.argv[4]
# max_rows = int(sys.argv[4]) if len(sys.argv) > 4 else None
max_rows = int(os.environ["MAX_ROWS"]) if "MAX_ROWS" in os.environ else None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Loading {checkpoint} on {dataset_path} {split}")

model = BertForMultiScaleSequenceRegression.from_pretrained(
    checkpoint, device_map=device
)
print("Model loaded")
model.eval()
model.zero_grad()
tokenizer = get_tokenizer()
print("Tokenizer loaded")
dataset, num_labels = load_from_disk_with_labels(dataset_path)
print("Dataset loaded")


def predict(model, inputs, attention_mask=None):
    pred = model(inputs, attention_mask=attention_mask)
    return pred.hidden_linear


def blank_reference_input(tokenized_input, blank_token_id):
    blank_input_ids = tokenized_input.input_ids.clone().detach()
    blank_input_ids[tokenized_input.special_tokens_mask == 0] = blank_token_id
    return blank_input_ids, tokenized_input.attention_mask


def is_cont(token):
    return token.startswith("##")


def norm(token):
    return token.replace("##", "")


def get_word_at(tokens, idx):
    bits = []
    has_prev = is_cont(tokens[idx])
    bk_idx = idx - 1
    while bk_idx > 0 and has_prev:
        has_prev = is_cont(tokens[bk_idx])
        bits.insert(0, norm(tokens[bk_idx]))
        bk_idx -= 1
    bits.append(norm(tokens[idx]))
    fd_idx = idx + 1
    while fd_idx < len(tokens) and is_cont(tokens[fd_idx]):
        bits.append(norm(tokens[fd_idx]))
        fd_idx += 1
    return "".join(bits)


def explain(text, model, tokenizer, int_bs, n_steps=50):
    inp = tokenizer(
        text, return_tensors="pt", return_special_tokens_mask=True, truncation=True
    ).to(model.device)
    b_input_ids, b_attention_mask = blank_reference_input(
        inp, tokenizer.convert_tokens_to_ids("-")
    )

    def predict_f(inputs, attention_mask=None):
        return predict(model, inputs, attention_mask)

    lig = LayerIntegratedGradients(predict_f, model.bert.embeddings)
    attrs = lig.attribute(
        inputs=(inp.input_ids, inp.attention_mask),
        baselines=(b_input_ids, b_attention_mask),
        target=(0,),
        internal_batch_size=int_bs,
        n_steps=n_steps,
    )
    # print("big")
    # print(attrs)
    attrs_sum = attrs.sum(dim=-1)
    # attrs_sum = attrs_sum / torch.norm(attrs_sum)
    # print("small")
    # print(attrs_sum)
    # print(inp.input_ids)
    tokens = tokenizer.convert_ids_to_tokens(inp.input_ids[0])
    # pprint(list(enumerate(tokens)))
    worst_token = attrs_sum.argmin()
    best_token = attrs_sum.argmax()
    # print("Worst", worst_token)
    # print("Best", best_token)
    return (get_word_at(tokens, worst_token), get_word_at(tokens, best_token))


crosstab = numpy.zeros((2, 4), dtype=int)
bests = []
worsts = []


SWN_NEG = 0
SWN_POS = 1
SWN_NEU = 2
SWN_UNK = 3
SWN_KEY = ["neu", "pos", "neg", "unk"]


def classify_sentiment(word):
    synsets = list(swn.senti_synsets(word))
    if len(synsets) == 0:
        return SWN_UNK
    synset = synsets[0]
    # print(f"word {word} lemma {synset.synset.name()}")
    if synset.pos_score() > synset.neg_score():
        return SWN_POS
    elif synset.pos_score() < synset.neg_score():
        return SWN_NEG
    else:
        return SWN_NEU


if "INTERNAL_BATCH_SIZE" in os.environ:
    internal_batch_size = int(os.environ["INTERNAL_BATCH_SIZE"])
else:
    internal_batch_size = 10

row_idx = 0
if max_rows is not None:
    num_rows = min(max_rows, len(dataset[split]))
else:
    num_rows = len(dataset[split])
for row in dataset[split]:
    worst, best = explain(row["text"], model, tokenizer, internal_batch_size)
    worst_swn = classify_sentiment(worst)
    best_swn = classify_sentiment(best)
    print(
        f"{row_idx}/{num_rows} worst {worst} {SWN_KEY[worst_swn]}\tbest {best} {SWN_KEY[best_swn]}"
    )
    crosstab[0, worst_swn] += 1
    crosstab[1, best_swn] += 1
    row_idx += 1
    if max_rows is not None and row_idx > max_rows:
        break


print(crosstab)
crosstab.dump(outfn)
