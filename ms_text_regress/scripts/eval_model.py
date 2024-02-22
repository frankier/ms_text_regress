from dataclasses import dataclass
from pprint import pprint
import torch

from ms_text_regress.scripts.train import TrainerAndEvaluator, ensure_context, HfArgumentParser, TrainingArguments, ExtraArguments


@dataclass
class LoadArguments:
    checkpoint_path: str


def main():
    training_args, extra_args, load_args = TrainerAndEvaluator.get_args(parser=HfArgumentParser((TrainingArguments, ExtraArguments, LoadArguments)))
    traineval = TrainerAndEvaluator(training_args, extra_args)
    traineval.setup_eval_buffers()
    pool = traineval.setup_pool()
    trainer = traineval.get_trainer()
    traineval.model.load_state_dict(torch.load(load_args.checkpoint_path))

    with ensure_context(pool) as pool_init:
        traineval.pool = pool_init
        metrics = trainer.evaluate(traineval.eval_dataset["validation"], metric_key_prefix="eval")
        print("Validation metrics")
        if "label_dists" in metrics:
            del metrics["label_dists"]
        pprint(metrics)

        metrics = trainer.evaluate(traineval.eval_dataset["test"], metric_key_prefix="test")
        print("Test metrics")
        if "label_dists" in metrics:
            del metrics["label_dists"]
        pprint(metrics)



if __name__ == "__main__":
    main()