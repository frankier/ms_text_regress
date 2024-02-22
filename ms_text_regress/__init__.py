import ms_text_regress.ordinal as ord
import ms_text_regress.ordinal_models.bert as ord_bert
from ms_text_regress.ordinal import (
    ordinal_decode_labels_np,
    ordinal_decode_labels_pt,
    ordinal_encode_labels,
)
from ms_text_regress.ordinal_models.bert import BertForOrdinalRegression
from ms_text_regress.transformers_utils import Trainer

__all__ = [
    "ord",
    "ord_bert",
    "ordinal_decode_labels_np",
    "ordinal_decode_labels_pt",
    "ordinal_encode_labels",
    "BertForOrdinalRegression",
    "Trainer",
]


if hasattr(ord, "ordinal_decode_multi_labels_pt"):
    ordinal_decode_multi_labels_pt = ord.ordinal_decode_multi_labels_pt
    __all__.append("ordinal_decode_multi_labels_pt")
if hasattr(ord_bert, "BertForMultiScaleOrdinalRegression"):
    BertForMultiScaleOrdinalRegression = ord_bert.BertForMultiScaleOrdinalRegression
    __all__.append("BertForMultiScaleOrdinalRegression")
