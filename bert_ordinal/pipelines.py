from operator import itemgetter

import torch
from transformers.pipelines.base import PIPELINE_INIT_ARGS, Pipeline
from transformers.utils import add_end_docstrings

from bert_ordinal.ordinal import score_labels_ragged_pt


class TextClassificationPipeline(Pipeline):
    pass


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""""",
)
class OrdinalRegressionPipeline(Pipeline):
    """
    Text classification pipeline using `BertForOrdinalRegression`.
    """

    return_all_scores = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # TODO? self.check_model_type

    def _sanitize_parameters(self, **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        postprocess_params = {}

        return preprocess_params, {}, postprocess_params

    def __call__(self, *args, **kwargs):
        """
        Classify the text(s) given as inputs.
        Args:
            args (`str` or `List[str]` or `Dict[str]`, or `List[Dict[str]]`):
                One or several texts to classify. In order to use text pairs for your classification, you can send a
                dictionary containing `{"text", "text_pair"}` keys, or a list of those.
        Return:
            A list or a list of list of `dict`: Each result comes as list of dictionaries with the following keys:
            - **label** (`str`) -- The label predicted.
            - **score** (`float`) -- The corresponding probability.
        """
        return super().__call__(*args, **kwargs)

    def preprocess(self, inputs, **tokenizer_kwargs):  # -> Dict[str, GenericTensor]:
        return_tensors = self.framework
        if isinstance(inputs, dict):
            if "task_ids" in inputs:
                return (
                    self.tokenizer(
                        inputs["text"],
                        return_tensors=return_tensors,
                        **tokenizer_kwargs
                    ),
                    inputs["task_ids"],
                )
            else:
                return self.tokenizer(
                    inputs["text"], return_tensors=return_tensors, **tokenizer_kwargs
                )
        else:
            return self.tokenizer(
                inputs, return_tensors=return_tensors, **tokenizer_kwargs
            )

    def _forward(self, model_inputs):
        if isinstance(model_inputs, tuple):
            return self.model(
                **model_inputs[0], task_ids=torch.tensor([model_inputs[1]])
            )
        else:
            return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        link = self.model.link

        # print()
        # print(model_outputs)
        hidden = model_outputs["hidden_linear"].item()
        ordinal_logits = model_outputs["ordinal_logits"][0]
        top = link.top_from_logits(ordinal_logits).item()
        label_dist = link.label_dist_from_logits(ordinal_logits)
        print(label_dist)

        # scores = score_labels_ragged_pt(model_outputs["ordinal_logits"])

        dict_scores = sorted(
            [{"index": i, "score": score.item()} for i, score in enumerate(label_dist)],
            key=itemgetter("score"),
            reverse=True,
        )
        return {
            "hidden": hidden,
            "el_mo_summary": link.summarize_logits(ordinal_logits),
            "scores": dict_scores,
            **summarize_label_dist(label_dist),
        }
