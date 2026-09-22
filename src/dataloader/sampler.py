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

        self.groups = dataset.get_event_file_groups()
        self.effective_batch_size = 3 * (batch_size // 2) 


    def __iter__(self):
        groups = [group.copy() for group in self.groups]
        if self.shuffle:
            random.shuffle(groups)

        for group in groups:
            if self.shuffle:
                random.shuffle(group)
            for start in range(0, len(group), self.effective_batch_size):
                batch = group[start : start + self.effective_batch_size]
                if len(batch) == self.effective_batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        if self.drop_last:
            return sum(len(group) // self.effective_batch_size for group in self.groups)
        return sum(math.ceil(len(group) / self.effective_batch_size) for group in self.groups)