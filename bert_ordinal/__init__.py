import bert_ordinal.ordinal as ord
import bert_ordinal.ordinal_models.bert as ord_bert
from bert_ordinal.ordinal import (
    ordinal_decode_labels_np,
    ordinal_decode_labels_pt,
    ordinal_encode_labels,
)
from bert_ordinal.ordinal_models.bert import BertForOrdinalRegression, Trainer

if hasattr(ord, "ordinal_decode_multi_labels_pt"):
    ordinal_decode_multi_labels_pt = ord.ordinal_decode_multi_labels_pt
if hasattr(ord_bert, "BertForMultiCutoffOrdinalRegression"):
    BertForMultiCutoffOrdinalRegression = ord_bert.BertForMultiCutoffOrdinalRegression
