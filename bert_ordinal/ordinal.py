from dataclasses import dataclass
from typing import Optional, Tuple, Union
import packaging.version

import numpy
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
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
        hidden_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Latent variable, before logits is applied.
        task_cutoffs (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`), *optional*, returned when `task_ids` is provided):
            Cutoffs corresponding to each task_id.
        ordinal_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`, *optional*, returned when `task_ids` is provided):
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
    hidden_logits: torch.FloatTensor = None
    task_cutoffs: Optional[torch.FloatTensor] = None
    ordinal_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def ordinal_encode_labels(input: torch.LongTensor, num_labels: int) -> torch.FloatTensor:
    """
    Performs ordinal encoding of a batch/tensor of label indices.

    Args:
        input (`torch.LongTensor`):
            A tensor of label indices typically shape (batch_size,) of labels in the range [0, num_labels)
        num_labels (`int`):
            The number of labels
    Returns:
        `torch.FloatTensor`: A tensor of shape (batch_size, num_labels - 1)
    """
    return (input.unsqueeze(1) >= torch.arange(1, num_labels, device=input.device)).float()


def ordinal_decode_labels_pt(input: torch.FloatTensor) -> torch.LongTensor:
    """
    Performs ordinal decoding of a batch/tensor of ordinal encoded logits label indices.

    Args:
        input (`torch.LongTensor`):
            A PyTorch tensor of ordinal encoded logits, typically of shape
            (batch_size, num_labels - 1), typically predictions from a model
    Returns:
        `torch.FloatTensor`: A PyTorch tensor typically of shape (batch_size,)
    """
    # sigmoid(0) = 0.5
    return (input >= 0.0).sum(dim=-1)


def score_labels_one_pt(input: torch.FloatTensor) -> torch.FloatTensor:
    """
    Scores a batch/tensor of ordinal encoded logits.

    Args:
        input (`torch.FloatTensor`):
            A PyTorch tensor of ordinal encoded logits, of shape
            (batch_size, num_labels - 1), typically predictions from a model
    Returns:
        `torch.FloatTensor`: A PyTorch tensor of scores, of shape (batch_size, num_labels)
    """
    gte_scores = input.sigmoid()
    return (
        torch.cat([torch.ones(1), gte_scores], dim=-1)
        - torch.cat([gte_scores, torch.zeros(1)], dim=-1)
    )


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
    return (input >= 0.0).sum(axis=-1)


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
        with torch.no_grad():
            torch.nn.init.normal_(self.weights)
        self.weights.data.copy_(torch.sort(self.weights)[0])

    def forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
        return input - self.weights


if packaging.version.parse(torch.__version__) >= packaging.version.parse("1.13"):
    import torch.nested
    from torch.nn.functional import binary_cross_entropy_with_logits

    def score_labels_ragged_pt(input: torch.FloatTensor) -> torch.FloatTensor:
        """
        Scores a batch/tensor of ordinal encoded logits.

        Args:
            input (`torch.FloatTensor`):
                A PyTorch tensor of ordinal encoded logits, typically of shape
                (batch_size, num_labels - 1), typically predictions from a model
        Returns:
            `torch.FloatTensor`: A PyTorch tensor typically of shape (batch_size, num_labels)
        """
        return torch.nested.nested_tensor([score_labels_one_pt(t) for t in input.unbind()])

    def bce_with_logits_ragged_mean(input: torch.Tensor, target: torch.Tensor):
        """
        Like `binary_cross_entropy_with_logits(...)`, but take the mean as if
        taken first across each example, and then across the batch, so that the
        loss contribution of an example is normalized with respect to its
        length.

        Args:
            input (`torch.nested.nested_tensor`):
                The input tensor in logit space
            target (`torch.nested.nested_tensor`):
                The target tensor in 0..1 space
        Returns:
            `torch.FloatTensor`: The loss
        """
        batch_size = input.size(0)
        bces = binary_cross_entropy_with_logits(
            input.values(),
            target.values(),
            reduction="none"
        )
        # XXX: Better floating point precision might be possible here by
        # grouping tensors by length and summing, then dividing them all at
        # once, and then summing again
        denoms = torch.hstack([
            torch.full(
                (len(t),), len(t),
                device=input.device,
                dtype=torch.float
            ) for t in input.unbind()
        ])
        return torch.sum(bces / denoms) / batch_size

    def ordinal_encode_multi_labels(input: torch.LongTensor, num_labels: torch.LongTensor) -> torch.Tensor:
        """
        Performs ordinal encoding of a batch/tensor of label indices. Each
        label `input[i]`, has a corresponding number of labels `num_labels[i]`.

        Args:
            input (`torch.LongTensor`):
                A tensor of label indices typically shape (batch_size,) of labels in the range [0, num_labels)
            num_labels (`torch.LongTensor`):
                The number of possible labels for each input label
        Returns:
            `torch.nested.nested_tensor`: A ragged tensor of shape (batch_size, num_labels[i] - 1)
        """
        return torch.nested.as_nested_tensor([
            (inp >= torch.arange(1, nl, device=input.device)).float()
            for inp, nl in zip(input, num_labels)
        ])

    def ordinal_decode_multi_labels_pt(input: torch.nested.nested_tensor) -> torch.LongTensor:
        return torch.tensor(
            [torch.count_nonzero(t >= 0.0) for t in input.unbind()], 
            device=input.device
        )

    class MultiOrdinalCutoffs(nn.Module):
        """
        This layer keeps track of the cutoff points between the ordered classes (in
        logit space).

        Args:
            num_labels (`int`):
                The number of labels
        """
        def __init__(self, num_labels, device=None):
            super().__init__()
            # Could be a nested_tensor with
            # https://github.com/pytorch/pytorch/issues/87034
            self.weights = torch.nn.ParameterList([
                torch.empty(cutoff_labels - 1, device=device, dtype=torch.float)
                for cutoff_labels in num_labels
            ])
            self.reset_parameters()

        def reset_parameters(self):
            with torch.no_grad():
                for cutoff in self.weights:
                    torch.nn.init.normal_(cutoff)
                    cutoff.data.copy_(torch.sort(cutoff)[0])

        def forward(self, input: torch.FloatTensor, cutoff_ids: torch.LongTensor) -> torch.FloatTensor:
            # Broadcasting would be nice https://github.com/pytorch/pytorch/issues/86888
            # As would getting a view into self.weights https://github.com/pytorch/pytorch/issues/86890
            repeated_hiddens = torch.nested.as_nested_tensor([
                input[i, 0].repeat(len(self.weights[cutoff_ids[i]]))
                for i in range(len(input))
            ])
            # This is negated here since we don't have nested_tensor - nested_tensor
            #   https://github.com/pytorch/pytorch/issues/86889
            pos_cutoffs = torch.nested.nested_tensor([
                self.weights[cutoff_id]
                for cutoff_id in cutoff_ids
            ])
            neg_cutoffs = torch.nested.nested_tensor([
                -self.weights[cutoff_id]
                for cutoff_id in cutoff_ids
            ])
            return repeated_hiddens + neg_cutoffs, pos_cutoffs