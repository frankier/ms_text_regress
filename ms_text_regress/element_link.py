"""
## Element-Link Multinomial-Ordinal (EL-MO) families

These are defined using the same language as in the following article

Wurm, M. J., Rathouz, P. J., & Hanlon, B. M. (2021).
*Regularized Ordinal Regression and the ordinalNet R Package.*
Journal of Statistical Software, 99(6).

| MO family                         | δ_k                          |
|-----------------------------------|------------------------------|
| Cumulative Probability (forward)  | P(Y ≤ k)                     |
| Cumulative Probability (backward) | P(Y ≥ k + 1)                 |
| Stopping Ratio (forward)          | P(Y = k | Y ≥ k)             |
| Stopping Ratio (backward)         | P(Y = k + 1 | Y ≤ k + 1)     |
| Continuation Ratio (forward)      | P(Y > k | Y ≥ k)             |
| Continuation Ratio (backward)     | P(Y < k + 1 | Y ≤ k + 1)     |
| Adjacent Category (forward)       | P(Y = k + 1 | k ≤ Y ≤ k + 1) |
| Adjacent Category (backward)      | P(Y = k | k ≤ Y ≤ k + 1)     |

The main differences are that we use 0-based indexing and that we use
num_labels = K + 1.
"""

from operator import ge, gt, le, lt
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

link_registry = {}


def register_link(cls):
    class_attrs = vars(cls)
    name = class_attrs["name"]
    inst = cls()
    globals()[name] = inst
    link_registry[name] = inst
    return cls


def get_link_by_name(name):
    return link_registry[name]


class ElementLink:
    @classmethod
    def link(
        cls, target: torch.Tensor, num_labels: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes the given EL-MO encoding of a batch/tensor of label indices.

        Args:
            target (`torch.LongTensor`):
                A tensor of label indices typically shape (batch_size,) of labels in
                the range [0, num_labels)
            num_labels (`int`):
                The number of labels
        Returns:
            `torch.FloatTensor`: A tensor of shape (batch_size, num_labels - 1)
            `torch.FloatTensor`: Weights corresponding to the conditional part of
                                the probability
        """
        raise NotImplementedError()

    def top_from_preds_and_logits(
        cls, preds: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """
        The idea here is that we can use either preds or logits. If the link has
        not specialised top_from_logits, we should just use top_from_preds to
        avoid recomputing the preds.
        """
        if cls.top_from_logits is not ElementLink.top_from_logits:
            return cls.top_from_logits(logits)
        else:
            return cls.top_from_preds(preds)

    @classmethod
    def top_from_preds(cls, preds: torch.Tensor) -> torch.Tensor:
        return cls.label_dist_from_preds(preds).argmax(dim=-1)

    @classmethod
    def top_from_logits(cls, logits: torch.Tensor) -> torch.Tensor:
        return cls.label_dist_from_logits(logits).argmax(dim=-1)

    @classmethod
    def label_dist_from_logits(cls, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the label distribution from logits.

        Args:
            logits (`torch.FloatTensor`):
                A PyTorch tensor of ordinal encoded logits, of shape
                (batch_size, num_labels - 1), typically predictions from a model
        Returns:
            `torch.FloatTensor`: A PyTorch tensor of label probabilities, of shape (batch_size, num_labels)
        """
        return cls.label_dist_from_preds(logits.sigmoid())

    @classmethod
    def label_dist_from_preds(cls, preds: torch.Tensor) -> torch.Tensor:
        """
        Gets the label P(Y=k) from a probability distribution over the δ_k
        values.

        Args:
            preds (`torch.FloatTensor`):
                A PyTorch prediction tensor of shape (batch_size, num_labels - 1)
        Returns:
            `torch.FloatTensor`: A PyTorch tensor of of label probabilities, of shape (batch_size, num_labels)
        """
        raise NotImplementedError()

    @classmethod
    def summarize_logits(cls, logits: torch.Tensor) -> List[Dict[str, Any]]:
        out = []
        probs = logits.sigmoid()
        for idx, (logit, prob) in enumerate(zip(logits, probs)):
            out.append(
                {
                    "index": idx,
                    "subprob": cls.repr_subproblem(idx),
                    "logit": logit.item(),
                    "score": prob.item(),
                }
            )
        return out

    @classmethod
    def repr_subproblem(cls, idx) -> str:
        raise NotImplementedError()


def cmp_target(op, target, start, num_labels):
    return op(
        target.unsqueeze(-1),
        torch.arange(start, start + num_labels - 1, device=target.device),
    ).float()


@register_link
class FwdCumulative(ElementLink):
    """
    Forward cumulative probability from the EL-MO family. Models P(Y ≤ k).
    """

    name = "fwd_cumulative"

    @classmethod
    def link(
        cls, target: torch.Tensor, num_labels: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        delta = cmp_target(le, target, 0, num_labels)
        weights = None
        return delta, weights

    @classmethod
    def label_dist_from_preds(cls, preds: torch.Tensor) -> torch.Tensor:
        # P(Y ≤ k) - P(Y ≤ k - 1)
        return F.pad(preds, (0, 1), value=1.0) - F.pad(preds, (1, 0), value=0.0)

    @classmethod
    def repr_subproblem(cls, idx) -> str:
        return f"P(Y ≤ {idx})"


@register_link
class BwdCumulative(ElementLink):
    """
    Backward cumulative probability from the EL-MO family. Models P(Y ≥ k + 1).
    """

    name = "bwd_cumulative"

    @classmethod
    def link(
        cls, target: torch.Tensor, num_labels: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        delta = cmp_target(ge, target, 1, num_labels)
        weights = None
        return delta, weights

    @classmethod
    def label_dist_from_preds(cls, preds: torch.Tensor) -> torch.Tensor:
        # P(Y ≥ k) - P(Y ≥ k + 1)
        return F.pad(preds, (1, 0), value=1.0) - F.pad(preds, (0, 1), value=0.0)

    @classmethod
    def repr_subproblem(cls, idx) -> str:
        return f"P(Y ≥ {idx + 1})"


def accumulate_probs(preds: torch.Tensor, reduce_fn, out_fn, type="fwd"):
    """
    Accumulates probabilities in the forward or backward direction.
    """
    preds_shape = preds.size()
    preds_batch = preds_shape[:-1]
    num_cutoffs = preds_shape[-1]
    if type == "fwd":
        start_idx = 0
        end_idx = num_cutoffs
        iter_range = range(1, num_cutoffs)
    elif type == "bwd":
        start_idx = -1
        end_idx = 0
        iter_range = range(-2, -num_cutoffs - 1, -1)
    else:
        raise ValueError(f"Unknown type {type}, must be 'fwd' or 'bwd'")
    output = torch.empty(*preds_batch, num_cutoffs + 1)
    output[..., start_idx] = out_fn(preds[..., start_idx], 1.0)
    # P(Y = k | Y ≥ k) * (1 - P(Y < k))
    el = preds[..., start_idx].detach()
    acc = reduce_fn(el, 1.0)
    for idx in iter_range:
        el = preds[..., idx]
        output[..., idx] = out_fn(el, acc)
        acc = reduce_fn(el, acc)
    output[..., end_idx] = acc
    return output


@register_link
class FwdSratio(ElementLink):
    """
    Forward stopping ratio from the EL-MO family. Models P(Y = k | Y ≥ k).
    """

    name = "fwd_sratio"

    @classmethod
    def link(
        cls, target: torch.Tensor, num_labels: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        delta = F.one_hot(target, num_labels)[..., :-1]
        weights = cmp_target(ge, target, 0, num_labels)
        return delta.float(), weights

    @classmethod
    def label_dist_from_preds(cls, preds: torch.Tensor) -> torch.Tensor:
        return accumulate_probs(
            preds, lambda el, acc: acc * (1 - el), lambda el, acc: el * acc, type="fwd"
        )

    @classmethod
    def repr_subproblem(cls, idx) -> str:
        return f"P(Y = {idx} | Y ≥ {idx})"


@register_link
class BwdSratio(ElementLink):
    """
    Backward stopping ratio from the EL-MO family. Models P(Y = k + 1 | Y ≤ k + 1).
    """

    name = "bwd_sratio"

    @classmethod
    def link(
        cls, target: torch.Tensor, num_labels: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        delta = F.one_hot(target, num_labels)[..., 1:]
        weights = cmp_target(le, target, 1, num_labels)
        return delta.float(), weights

    @classmethod
    def label_dist_from_preds(cls, preds: torch.Tensor) -> torch.Tensor:
        return accumulate_probs(
            preds, lambda el, acc: acc * (1 - el), lambda el, acc: el * acc, type="bwd"
        )

    @classmethod
    def repr_subproblem(cls, idx) -> str:
        return f"P(Y = {idx + 1} | Y ≤ {idx + 1})"


@register_link
class FwdCratio(ElementLink):
    """
    Forward continuation ratio from the EL-MO family. Models P(Y > k | Y ≥ k).
    """

    name = "fwd_cratio"

    @classmethod
    def link(
        cls, target: torch.Tensor, num_labels: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        delta = cmp_target(gt, target, 0, num_labels)
        weights = cmp_target(ge, target, 0, num_labels)
        return delta, weights

    @classmethod
    def label_dist_from_preds(cls, preds: torch.Tensor) -> torch.Tensor:
        return accumulate_probs(
            preds, lambda el, acc: el * acc, lambda el, acc: (1 - el) * acc, type="fwd"
        )

    @classmethod
    def repr_subproblem(cls, idx) -> str:
        return f"P(Y > {idx} | Y ≥ {idx})"


@register_link
class BwdCratio(ElementLink):
    """
    Backward continuation ratio from the EL-MO familys. Models P(Y < k + 1 | Y ≤ k + 1).
    """

    name = "bwd_cratio"

    @classmethod
    def link(
        cls, target: torch.Tensor, num_labels: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        delta = cmp_target(lt, target, 1, num_labels)
        weights = cmp_target(le, target, 1, num_labels)
        return delta, weights

    @classmethod
    def label_dist_from_preds(cls, preds: torch.Tensor) -> torch.Tensor:
        return accumulate_probs(
            preds, lambda el, acc: el * acc, lambda el, acc: (1 - el) * acc, type="bwd"
        )

    @classmethod
    def repr_subproblem(cls, idx) -> str:
        return f"P(Y < {idx + 1} | Y ≤ {idx + 1})"


class AcatBase(ElementLink):
    @classmethod
    def top_from_logits(cls, logits: torch.Tensor) -> torch.Tensor:
        return cls._class_logits_from_ord_logits(logits).argmax(dim=-1)

    @classmethod
    def label_dist_from_logits(cls, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(cls._class_logits_from_ord_logits(logits), dim=-1)

    @classmethod
    def _class_logits_from_ord_logits(cls, ord_logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


@register_link
class FwdAcat(AcatBase):
    """
    Forward adjacent category from the EL-MO family. Models P(Y = k + 1 | k ≤ Y ≤ k + 1).
    """

    name = "fwd_acat"

    @classmethod
    def link(
        cls, target: torch.Tensor, num_labels: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        one_hot = F.one_hot(target, num_labels)
        delta = one_hot[..., 1:]
        weights = delta + one_hot[..., :-1]
        return delta.float(), weights.float()

    @classmethod
    def _class_logits_from_ord_logits(cls, ord_logits: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(F.pad(ord_logits, (1, 0), value=0.0), dim=-1)

    @classmethod
    def repr_subproblem(cls, idx) -> str:
        return f"P(Y = {idx + 1} | {idx} ≤ Y ≤ {idx + 1})"


def rcumsum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Workaround for lack of reverse cumsum/zero copy flip in PyTorch.
    See https://github.com/pytorch/pytorch/issues/33520
    """
    csum = torch.cumsum(x, dim=dim)
    return x - csum + torch.select(csum, dim, -1).unsqueeze(-1)


@register_link
class BwdAcat(AcatBase):
    """
    Backward adjacent category from the EL-MO family. Models P(Y = k | k ≤ Y ≤ k + 1).
    """

    name = "bwd_acat"

    @classmethod
    def link(
        cls, target: torch.Tensor, num_labels: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        one_hot = F.one_hot(target, num_labels)
        delta = one_hot[..., :-1]
        weights = delta + one_hot[..., 1:]
        return delta.float(), weights.float()

    @classmethod
    def _class_logits_from_ord_logits(cls, ord_logits: torch.Tensor) -> torch.Tensor:
        return rcumsum(F.pad(ord_logits, (0, 1), value=0.0))

    @classmethod
    def repr_subproblem(cls, idx) -> str:
        return f"P(Y = {idx} | {idx} ≤ Y ≤ {idx + 1})"


DEFAULT_LINK_NAME = "fwd_acat"
