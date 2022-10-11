from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bert.modeling_bert import (
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
    BertPreTrainedModel,
    BertModel
)


@dataclass
class OrdinalRegressionOutput(ModelOutput):
    """
    Base class for outputs of ordinal regression models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Ordinal regression loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Ordinal regression scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def ordinal_encode_labels(input: torch.LongTensor, num_labels: int) -> torch.FloatTensor:
    """
    Performs ordinal encoding of a tensor of label indices.

    Args:
        input (`torch.LongTensor`):
            A tensor of label indices typically shape (batch_size,) of labels in the range [0, num_labels)
        num_labels (`int`):
            The number of labels
    Returns:
        `torch.FloatTensor`: A tensor of shape (batch_size, num_labels - 1)
    """
    return (input.unsqueeze(1) >= torch.arange(1, num_labels)).float()


def ordinal_decode_labels_pt(input: torch.FloatTensor) -> torch.LongTensor:
    """
    Performs ordinal decoding of a tensor of ordinal encoded logits label indices.

    Args:
        input (`torch.LongTensor`):
            A PyTorch tensor of ordinal encoded logits, typically of shape
            (batch_size, num_labels - 1), typically predictions from a model
    Returns:
        `torch.FloatTensor`: A PyTorch tensor typically of shape (batch_size,)
    """
    # sigmoid(0) = 0.5
    return (input >= 0.0).sum(dim=-1) + 1


def ordinal_decode_labels_np(input: numpy.array) -> numpy.array:
    """
    Performs ordinal decoding of a NumPy array of ordinal encoded logits into a
    NumPy array of label indices.

    Args:
        input (`numpy.array`):
            A NumPy array of ordinal encoded logits, typically of shape
            (batch_size, num_labels - 1), typically predictions from a model
    Returns:
        `numpy.array`: A numpy array typically of shape (batch_size,)
    """
    # sigmoid(0) = 0.5
    return (input >= 0.0).sum(axis=-1) + 1


class OrdinalCutoffs(nn.Module):
    """
    This layer keeps track of the cutoff points between the ordered classes (in
    logit space).

    Args:
        num_labels (`int`):
            The number of labels
    """
    def __init__(self, num_labels, device=None):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.empty(num_labels - 1, device=device, dtype=torch.float))
        self.reset_parameters()

    def reset_parameters(self):
        print("OrdinalCutoffs reset_parameters")
        with torch.no_grad():
            torch.nn.init.normal_(self.weights)
        self.weights.data.copy_(torch.sort(self.weights)[0])

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        print("OrdinalCutoffs")
        res = input - self.weights
        print("input", input)
        print("weights", self.weights)
        print("res", res)
        return res


@add_start_docstrings(
    """
    Bert Model transformer with an ordinal regression head on top (a linear layer on top of the pooled
    output) e.g. for essay grading.
    """,
    BERT_START_DOCSTRING,
)
class BertForOrdinalRegression(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.cutoffs = OrdinalCutoffs(config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()


    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], OrdinalRegressionOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence ordinal regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        mid_logits = self.classifier(pooled_output)
        ordinal_logits = self.cutoffs(mid_logits)

        loss = None

        if labels is not None:
            labels_ord_enc = ordinal_encode_labels(labels, self.num_labels)
            loss_fct = BCEWithLogitsLoss()
            print("ordinal_logits", ordinal_logits)
            print("labels_ord_enc", labels_ord_enc)
            loss = loss_fct(ordinal_logits, labels_ord_enc)
        if not return_dict:
            output = (ordinal_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return OrdinalRegressionOutput(
            loss=loss,
            logits=ordinal_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
