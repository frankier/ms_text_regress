import numpy as np
from torch.utils.data import Sampler

from bert_ordinal.vendor.hierarchical_sampler import safe_random_choice


def ensure_full_batches(instances, batch_size):
    remainders = len(instances) % batch_size
    if remainders == 0:
        return instances
    # Oversample just enough to make constant sized batches
    return np.concatenate(
        (instances, safe_random_choice(instances, batch_size - remainders))
    )


class TaskAtATimeSampler(Sampler):
    """
    This sampler puts a single task in each batch. In case the samplers per task
    does not divide the batch size, it slightly oversamples.
    """

    def __init__(
        self,
        task_ids,
        batch_size,
    ):
        self.task_ids = task_ids
        self.batch_size = batch_size
        self.reshuffle()

    def reshuffle(self):
        self.batches = []
        for task_id in np.unique(self.task_ids):
            instances = ensure_full_batches(
                np.where(self.task_ids == task_id)[0], self.batch_size
            )
            np.random.shuffle(instances)
            for i in range(0, len(instances), self.batch_size):
                self.batches.append(instances[i : i + self.batch_size])
        np.random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield [int(x) for x in batch]
        self.reshuffle()

    def __len__(self):
        return len(self.batches)


class LabelStratifiedTaskAtATimeSampler(Sampler):
    """
    This sampler is similar to TaskAtATimeSampler, but it ensures that each
    batch contains a stratified sample of the labels (without any regard
    considering them as ordinal).
    """

    def __init__(
        self,
        task_ids,
        labels,
        batch_size,
    ):
        self.task_ids = task_ids
        self.labels = labels
        self.batch_size = batch_size
        self.reshuffle()

    def reshuffle(self):
        self.batches = []
        for task_id in np.unique(self.task_ids):
            instances = ensure_full_batches(
                np.nonzero(self.task_ids == task_id)[0], self.batch_size
            )
            np.random.shuffle(instances)
            # Use a stable sort to preserve the shuffle order
            instances = instances[np.argsort(self.labels[instances], kind="stable")]
            num_batches = len(instances) // self.batch_size
            for i in range(num_batches):
                self.batches.append(instances[i::num_batches])
        np.random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield [int(x) for x in batch]
        self.reshuffle()

    def __len__(self):
        return len(self.batches)


class FixedTaskLabelSampler(Sampler):
    """
    This sampler has a fixed number of tasks and labels per batch.
    """

    def __init__(
        self,
        task_ids,
        labels,
        tasks_per_batch,
        labels_per_task,
        batch_size,
    ):
        self.task_ids = np.array(task_ids)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.tasks_per_batch = tasks_per_batch
        self.labels_per_task = labels_per_task
        self.reshuffle()

    def reshuffle(self):
        self.batches = []
        by_task_label = {}
        tasks_left = np.empty(self.task_ids.max() + 1, dtype=int)
        labels_left = {}
        label_sorter = np.argsort(self.labels)
        task_sorter = np.argsort(self.task_ids[label_sorter], kind="stable")
        sorter = label_sorter[task_sorter]
        unsorter = np.argsort(sorter)
        labels_sorted = self.labels[sorter]
        for task_id, start, cnt in zip(
            *np.unique(self.task_ids[sorter], return_index=True, return_counts=True)
        ):
            task_labels = labels_sorted[start : start + cnt]
            tasks_left[task_id] = cnt
            labels_left[task_id] = np.zeros(task_labels.max() + 1, dtype=np.int)
            for label, label_start, label_cnt in zip(
                *np.unique(task_labels, return_index=True, return_counts=True)
            ):
                full_start = start + label_start
                by_task_label.setdefault(task_id, {})[label] = unsorter[
                    full_start : full_start + label_cnt
                ]
                labels_left[task_id][label] = label_cnt
        subbatch_size = self.tasks_per_batch * self.labels_per_task
        while 1:
            available_tasks = np.nonzero(tasks_left > 0)[0]
            if len(available_tasks) == 0:
                break
            # TODO: Rather than using safe_random_choice, it should fall back to using non-available tasks
            chosen_tasks = safe_random_choice(available_tasks, self.tasks_per_batch)
            for task_id in chosen_tasks:
                # TODO: Rather than using safe_random_choice, it should fall back to using non-available labels
                available_labels = np.nonzero(labels_left[task_id] > 0)[0]
                if len(available_labels) == 0:
                    available_labels = np.array(list(by_task_label[task_id].keys()))
                chosen_labels = safe_random_choice(
                    available_labels, self.labels_per_task
                )
                for label in chosen_labels:
                    idx = (
                        len(by_task_label[task_id][label]) - labels_left[task_id][label]
                    )
                    full_batch = by_task_label[task_id][label]
                    batch = full_batch[idx : idx + subbatch_size]
                    tasks_left[task_id] -= len(batch)
                    labels_left[task_id][label] -= len(batch)
                    if len(batch) < subbatch_size:
                        batch = np.concatenate(
                            (
                                batch,
                                safe_random_choice(
                                    full_batch, subbatch_size - len(batch)
                                ),
                            )
                        )
                    self.batches.append(batch)

    def __iter__(self):
        for batch in self.batches:
            yield [int(x) for x in batch]
        self.reshuffle()

    def __len__(self):
        return len(self.batches)
