"""
This module implements contains basic NumPy and PyTorch functions and PyTorch
modules for ordinal data and regression.  Many of these functions are not
specific to BERT/NLP.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy
import packaging.version
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.utils import ModelOutput


@dataclass
class OrdinalRegressionOutput(ModelOutput):
    """
    Base class for outputs of ordinal regression models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Ordinal regression loss.
        hidden_linear (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Latent variable, before logits is applied.
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

    loss: Optional[torch.Tensor] = None
    hidden_linear: Optional[torch.FloatTensor] = None
    ordinal_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def ordinal_encode_labels(input: torch.Tensor, num_labels: int) -> torch.Tensor:
    """
    Performs ordinal encoding of a batch/tensor of label indices. This is
    equivalent to the ordinal encoding of the forward cumulative probability
    multinomial-ordinal family.

    Args:
        input (`torch.LongTensor`):
            A tensor of label indices typically shape (batch_size,) of labels in the range [0, num_labels)
        num_labels (`int`):
            The number of labels
    Returns:
        `torch.FloatTensor`: A tensor of shape (batch_size, num_labels - 1)
    """
    return (
        input.unsqueeze(1) >= torch.arange(1, num_labels, device=input.device)
    ).float()


def ordinal_decode_labels_pt(input: torch.Tensor) -> torch.Tensor:
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


def score_labels_one_pt(input: torch.Tensor) -> torch.Tensor:
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
    return torch.cat([torch.ones(1), gte_scores], dim=-1) - torch.cat(
        [gte_scores, torch.zeros(1)], dim=-1
    )


def ordinal_decode_labels_np(input: numpy.ndarray) -> numpy.ndarray:
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


class ElementWiseAffine(nn.Module):
    """
    This layer does a slightly specialised element-wise affine transform of an
    single ordered scale into num_labels - 1 predictors ready to be used as
    logits with an something from the ELMO family. It induces cutoff points
    between the ordered classes (in logit space).

    The main ordinal-regression specific functionality is:

     1) Different choices of discrimination modes, including none, a single, and
        multi which has a discrimination per δ_k.
     2) Different initialisation modes to reasonable values for the offsets.

    Args:
        with_discrimination: ("none" | "single" | "multi")
            Whether to have not discrimination parameters, a single common one,
            or one per threshold.
        num_labels (`int`):
            The number of labels
        device (`torch.device`) *optional*:
            The device to put the parameters on
    """

    def __init__(self, with_discrimination, num_labels, device=None):
        super().__init__()
        if with_discrimination == "none":
            self.discrimination = torch.ones(1)
        elif with_discrimination == "single":
            self.discrimination = nn.Parameter(torch.ones(1))
        elif with_discrimination == "multi":
            self.discrimination = nn.Parameter(torch.ones(num_labels - 1))
        else:
            raise ValueError(
                f"Unknown discrimination type: {with_discrimination}, "
                "must be one of ('none', 'single', 'multi')"
            )
        self.offsets = torch.nn.Parameter(
            torch.empty(num_labels - 1, device=device, dtype=torch.float)
        )
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: reasonable initialisation probably depends on the link used
        with torch.no_grad():
            torch.nn.init.normal_(self.offsets)
        self.offsets.data.copy_(torch.sort(self.offsets)[0])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.discrimination * (input + self.offsets)

    def summary(self):
        return self.discrimination, self.offsets


DEFAULT_DISCRIMINATION_MODE = "none"


def ordinal_loss(
    input: torch.Tensor, target: torch.Tensor, link, num_labels: int
) -> torch.Tensor:
    target_enc, weights = link.link(target, num_labels)
    if weights is None:
        bces = binary_cross_entropy_with_logits(
            input, target_enc, weights, reduction="mean"
        )
    else:
        bces = binary_cross_entropy_with_logits(
            input, target_enc, weights, reduction="none"
        ) / weights.sum(1).unsqueeze(1)
    return bces.sum()


if packaging.version.parse(torch.__version__) >= packaging.version.parse("1.13"):
    from .mt_ordinal import *
