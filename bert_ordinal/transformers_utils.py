from typing import List


class BertMultiLabelsMixin:
    # Overwrite num_labels <=> id2label behaviour
    @property
    def num_labels(self) -> List[int]:
        return self._num_labels

    @num_labels.setter
    def num_labels(self, num_labels: List[int]):
        self._num_labels = num_labels
