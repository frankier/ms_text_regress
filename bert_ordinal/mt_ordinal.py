from typing import Tuple

import torch
import torch.nested
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits


def score_labels_ragged_pt(input: torch.Tensor) -> torch.Tensor:
    """
    Scores a batch/tensor of ordinal encoded logits.

    Args:
        input (`torch.FloatTensor`):
            A PyTorch tensor of ordinal encoded logits, typically of shape
            (batch_size, num_labels - 1), typically predictions from a model
    Returns:
        `torch.FloatTensor`: A PyTorch tensor typically of shape (batch_size, num_labels)
    """
    from .ordinal import score_labels_one_pt

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
        input.values(), target.values(), reduction="none"
    )
    # XXX: Better floating point precision might be possible here by
    # grouping tensors by length and summing, then dividing them all at
    # once, and then summing again
    denoms = torch.hstack(
        [
            torch.full((len(t),), len(t), device=input.device, dtype=torch.float)
            for t in input.unbind()
        ]
    )
    return torch.sum(bces / denoms) / batch_size


def ordinal_encode_multi_labels(
    input: torch.Tensor, num_labels: torch.Tensor
) -> torch.Tensor:
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
    return torch.nested.as_nested_tensor(
        [
            (inp >= torch.arange(1, nl, device=input.device)).float()
            for inp, nl in zip(input, num_labels)
        ]
    )


def ordinal_decode_multi_labels_pt(input: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [torch.count_nonzero(t >= 0.0) for t in input.unbind()], device=input.device
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
        self.weights = torch.nn.ParameterList(
            [
                torch.empty(cutoff_labels - 1, device=device, dtype=torch.float)
                for cutoff_labels in num_labels
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for cutoff in self.weights:
                torch.nn.init.normal_(cutoff)
                cutoff.data.copy_(torch.sort(cutoff)[0])

    def forward(
        self, input: torch.Tensor, cutoff_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Broadcasting would be nice https://github.com/pytorch/pytorch/issues/86888
        # As would getting a view into self.weights https://github.com/pytorch/pytorch/issues/86890
        repeated_hiddens = torch.nested.as_nested_tensor(
            [
                input[i, 0].repeat(len(self.weights[cutoff_ids[i]]))
                for i in range(len(input))
            ]
        )
        # This is negated here since we don't have nested_tensor - nested_tensor
        #   https://github.com/pytorch/pytorch/issues/86889
        pos_cutoffs = torch.nested.nested_tensor(
            [self.weights[cutoff_id] for cutoff_id in cutoff_ids]
        )
        neg_cutoffs = torch.nested.nested_tensor(
            [-self.weights[cutoff_id] for cutoff_id in cutoff_ids]
        )
        return repeated_hiddens + neg_cutoffs, pos_cutoffs
