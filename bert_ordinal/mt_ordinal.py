from typing import Optional, Tuple

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


def bce_with_logits_ragged_mean(
    input: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Like `binary_cross_entropy_with_logits(...)`, but take the mean as if
    taken first across each example, and then across the batch, so that the
    loss contribution of an example is normalized with respect to its
    length.

    Args:
        input (`torch.nested.nested_tensor`):
            The input tensor in logit space
        target (`torch.nested.nested_tensor`):
            The target tensor in 0..1 space. Same shape as `input`.
        weight (`torch.nested.nested_tensor`), *optional*:
            A manual reweighting for each element. Same shape as `input` and `target if given.
    Returns:
        `torch.FloatTensor`: The loss
    """
    batch_size = input.size(0)
    if weight is not None:
        weight_flat = weight.values()
    else:
        weight_flat = None
    bces = binary_cross_entropy_with_logits(
        input.values(), target.values(), weight_flat, reduction="none"
    )
    # XXX: Better floating point precision might be possible here by
    # grouping tensors by length and summing, then dividing them all at
    # once, and then summing again
    if weight is None:
        denoms = torch.hstack(
            [
                torch.full((len(t),), len(t), device=input.device, dtype=torch.float)
                for t in input.unbind()
            ]
        )
    else:
        denoms = torch.hstack(
            [
                torch.full((len(w),), sum(w), device=input.device, dtype=torch.float)
                for w in weight.unbind()
            ]
        )
    return torch.sum(bces / denoms) / batch_size


def ordinal_encode_multi_labels(
    link, target: torch.Tensor, num_labels: torch.Tensor
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Performs EL-MO encoding of a batch/tensor of label indices. Each label
    `input[i]`, has a corresponding number of labels `num_labels[i]`.

    Args:
        link:
            The EL-MO link function use
        input (`torch.LongTensor`):
            A tensor of label indices typically shape (batch_size,) of labels in the range [0, num_labels)
        num_labels (`torch.LongTensor`):
            The number of possible labels for each input label
    Returns:
        `torch.nested.nested_tensor`: A ragged tensor of shape (batch_size, num_labels[i] - 1)
    """
    target_enc = []
    weights = None
    for inp, nl in zip(target, num_labels, strict=True):
        one_target_enc, one_weights = link.link(inp, nl)
        target_enc.append(one_target_enc)
        if one_weights is not None:
            if weights is None:
                weights = []
            weights.append(one_weights)
    target_tensor = torch.nested.as_nested_tensor(target_enc)
    return (
        target_tensor,
        torch.nested.as_nested_tensor(weights) if weights is not None else None,
    )


def ordinal_loss_multi_labels(
    input: torch.Tensor, target: torch.Tensor, link, num_labels: torch.Tensor
) -> torch.Tensor:
    target_enc, weights = ordinal_encode_multi_labels(link, target, num_labels)
    return bce_with_logits_ragged_mean(input, target_enc, weights)


def ordinal_decode_multi_labels_pt(input: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [torch.count_nonzero(t >= 0.0) for t in input.unbind()], device=input.device
    )


class MultiElementWiseAffine(nn.Module):
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

    def __init__(
        self,
        with_discrimination,
        num_labels,
        linear_parameterisation=False,
        exp_discrimination=False,
        device=None,
    ):
        super().__init__()
        # Could be a nested_tensor with
        # https://github.com/pytorch/pytorch/issues/87034
        self.with_discrimination = with_discrimination
        self.linear_parameterisation = linear_parameterisation
        self.exp_discrimination = exp_discrimination
        discrim_init = self.discrimination_init_func()
        if with_discrimination == "none":
            self.discrimination = nn.Parameter(discrim_init(()), requires_grad=False)
        elif with_discrimination == "single":
            self.discrimination = nn.Parameter(discrim_init(()))
        elif with_discrimination == "per_task":
            self.discrimination = nn.Parameter(discrim_init(len(num_labels)))
        elif with_discrimination == "multi":
            self.discrimination = torch.nn.ParameterList(
                [
                    discrim_init(nl - 1, device=device, dtype=torch.float)
                    for nl in num_labels
                ]
            )
        else:
            raise ValueError(
                f"Unknown discrimination type: {with_discrimination}, "
                "must be one of ('none', 'single', 'per_task', 'multi')"
            )
        self.offsets = torch.nn.ParameterList(
            [torch.empty(nl - 1, device=device, dtype=torch.float) for nl in num_labels]
        )
        self.reset_parameters()

    def discrimination_init_func(self):
        if self.exp_discrimination:
            return torch.zeros
        else:
            return torch.ones

    def reset_parameters(self):
        with torch.no_grad():
            for offset in self.offsets:
                torch.nn.init.normal_(offset)
                offset.data.copy_(torch.sort(offset)[0])

    def forward(self, input: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        # Broadcasting would be nice https://github.com/pytorch/pytorch/issues/86888
        # As would getting a view into self.offsets https://github.com/pytorch/pytorch/issues/86890
        repeated_hiddens = torch.nested.as_nested_tensor(
            [
                input[i, 0].repeat(len(self.offsets[task_ids[i]]))
                for i in range(len(input))
            ]
        )
        offsets = torch.nested.as_nested_tensor(
            [self.offsets[task_id] for task_id in task_ids]
        )
        discrimination = self.transform_discrimination(
            self.task_discriminations(task_ids)
        )
        if self.linear_parameterisation:
            return discrimination * repeated_hiddens + offsets
        else:
            return discrimination * (repeated_hiddens + offsets)

    def task_discriminations(self, task_ids):
        if isinstance(self.discrimination, torch.nn.ParameterList):
            return torch.nested.as_nested_tensor(
                [self.discrimination[task_id] for task_id in task_ids]
            )
        elif self.discrimination.dim() == 1:
            # With more nested_tensor ops, this need only be
            # discrimination = self.discrimination[task_ids]
            return torch.nested.as_nested_tensor(
                [
                    self.discrimination[task_id].repeat(len(self.offsets[task_id]))
                    for task_id in task_ids
                ]
            )
        else:
            return self.discrimination

    def transform_discrimination(self, discrimination):
        if self.exp_discrimination:
            return torch.exp(discrimination)
        else:
            return discrimination

    def transform_offsets(self, offsets, discriminations):
        if self.linear_parameterisation:
            return offsets / discriminations
        else:
            return offsets

    def summary(self):
        if isinstance(self.discrimination, torch.nn.ParameterList):
            discriminations = torch.nested.nested_tensor(self.discrimination)
        else:
            discriminations = self.discrimination
        discriminations = (self.transform_discrimination(discriminations),)
        return (
            discriminations,
            self.transform_offsets(
                torch.nested.nested_tensor(self.offsets), discriminations
            ),
        )

    def task_summary(self, task_id):
        if isinstance(self.discrimination, torch.nn.ParameterList):
            discriminations = self.discrimination[task_id]
        elif self.discrimination.dim() == 1:
            discriminations = self.discrimination[task_id]
        else:
            discriminations = self.discrimination
        discriminations = self.transform_discrimination(discriminations)
        return discriminations, self.transform_offsets(
            self.offsets[task_id], discriminations
        )
