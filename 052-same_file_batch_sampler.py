from collections import defaultdict
import math
import random
from torch.utils.data import Sampler


class SameFileBatchSampler(Sampler):
    """Yield batches whose samples all reference one Parquet event part."""

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        if batch_size % 2 != 0:
            raise ValueError("batch_size must be even to allow for negative samples.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batch_size_with_negatives = batch_size // 2 * 3


        self.groups = dataset.get_event_file_groups()


    def __iter__(self):
        groups = [group.copy() for group in self.groups]
        if self.shuffle:
            random.shuffle(groups)

        for group in groups:
            if self.shuffle:
                random.shuffle(group)
            for start in range(0, len(group), self.batch_size_with_negatives):
                batch = group[start : start + self.batch_size_with_negatives]
                if len(batch) == self.batch_size_with_negatives or not self.drop_last:
                    yield batch

    def __len__(self):
        if self.drop_last:
            return sum(len(group) // self.batch_size_with_negatives for group in self.groups)
        return sum(math.ceil(len(group) / self.batch_size_with_negatives) for group in self.groups)