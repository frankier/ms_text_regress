import bert_ordinal.ordinal_models.bert as ord_bert
from bert_ordinal.ordinal_models.bert import (
    BertForOrdinalRegression,
    Trainer
)
import bert_ordinal.ordinal as ord
from bert_ordinal.ordinal import ordinal_encode_labels, ordinal_decode_labels_pt, ordinal_decode_labels_np

if hasattr(ord, "ordinal_decode_multi_labels_pt"):
    ordinal_decode_multi_labels_pt = ord.ordinal_decode_multi_labels_pt
if hasattr(ord_bert, "BertForMultiCutoffOrdinalRegression"):
    BertForMultiCutoffOrdinalRegression = ord_bert.BertForMultiCutoffOrdinalRegression
