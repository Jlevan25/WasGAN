import random
from typing import Iterator

from torch.utils.data import Sampler
from torch.utils.data.sampler import T_co


class SingleClassSampler(Sampler):
    def __init__(self, data_source, class_idx, shuffle):
        super().__init__(data_source)
        self.class_idx = class_idx
        self.shuffle = shuffle
        self.indices = [i for i, c in enumerate(data_source) if c == class_idx]

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)