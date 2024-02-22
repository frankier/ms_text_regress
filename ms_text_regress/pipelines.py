from operator import itemgetter

import torch
from transformers.pipelines.base import PIPELINE_INIT_ARGS, Pipeline
from transformers.utils import add_end_docstrings

from ms_text_regress.label_dist import summarize_label_dist


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""""",
)
class MultiTaskPipelineBase(Pipeline):
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


def output_label_dist(label_dist):
    dict_scores = sorted(
        [{"index": i, "score": score.item()} for i, score in enumerate(label_dist)],
        key=itemgetter("score"),
        reverse=True,
    )
    return {
        "scores": dict_scores,
        **{k: v.item() for k, v in summarize_label_dist(label_dist).items()},
    }


class TextRegressionPipeline(MultiTaskPipelineBase):
    """
    Text regression pipeline using `BertForMultiScaleSequenceRegression`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, model_inputs):
        model_outputs = super()._forward(model_inputs)
        return model_outputs, self.model.num_labels[model_inputs[1]]

    def postprocess(self, tpl):
        model_outputs, num_labels = tpl
        hidden = model_outputs["hidden_linear"].item()
        predictions = model_outputs["logits"][0]

        return {
            "hidden": hidden,
            "prediction": predictions.item(),
            "label": torch.clamp(predictions + 0.5, 0.0, num_labels - 1).int().item(),
        }


class TextClassificationPipeline(MultiTaskPipelineBase):
    """
    Text classification pipeline using `BertForMultiScaleSequenceClassification`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # TODO? self.check_model_type

    def postprocess(self, model_outputs):
        logits = model_outputs["logits"][0]
        label_dist = logits.softmax(-1)

        return output_label_dist(label_dist)


class OrdinalRegressionPipeline(MultiTaskPipelineBase):
    """
    Text classification pipeline using `BertForOrdinalRegression`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # TODO? self.check_model_type

    def _forward(self, model_inputs):
        model_outputs = super()._forward(model_inputs)
        # Here we get rid of the nested_tensor so that it works with Pytorch
        # 1.13 which can't copy them between devices
        model_outputs.logits = model_outputs.logits[0]
        return model_outputs

    def postprocess(self, model_outputs):
        link = self.model.link

        hidden = model_outputs["hidden_linear"].item()
        ordinal_logits = model_outputs["logits"]
        label_dist = link.label_dist_from_logits(ordinal_logits)

        return {
            "hidden": hidden,
            "el_mo_summary": link.summarize_logits(ordinal_logits),
            **output_label_dist(label_dist),
        }
