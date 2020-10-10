import torch
from torch.utils.data import IterableDataset, Dataset
from typing import List, Tuple
from math import ceil


class TextBatch:
    def __init__(self, samples: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.X = torch.cat([sample[0] for sample in samples], dim=1)
        self.Y = torch.cat([sample[1] for sample in samples], dim=1)

    def pin_memory(self):
        self.X = self.X.pin_memory()
        self.Y = self.Y.pin_memory()
        return self

    @staticmethod
    def collate_wrapper(
            batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> "TextBatch":
        return TextBatch(batch)


class TextIterDataset(IterableDataset):
    '''

    '''
    def __init__(self,
                 data: torch.Tensor,
                 seq_len: int,
                 shuffle: bool = False):
        '''

        :param data: [total word len, 1]
        :param seq_len: sentence length
        '''
        self._data = data
        self._seq_len = seq_len
        self._cur_sample_idx = 0
        self._total_n_samples = len(self._data) // self._seq_len
        import numpy
        self._order = numpy.arange(self._total_n_samples)
        if shuffle:
            self._order = numpy.random.permutation(self._order)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self._cur_sample_idx = 0
            self._end_sample_idx = self._total_n_samples
        else:
            worker_id = worker_info.id
            per_worker = int(
                ceil(self._total_n_samples / float(worker_info.num_workers)))
            self._cur_sample_idx = per_worker * worker_id
            self._end_sample_idx = min(self._cur_sample_idx + per_worker,
                                       self._total_n_samples)
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        :return: ([seq len; 1], [seq len; 1])
        '''
        if self._cur_sample_idx >= self._end_sample_idx:
            raise StopIteration()

        data, target = self._get_seq(self._order[self._cur_sample_idx] *
                                     self._seq_len)
        self._cur_sample_idx += 1
        return data, target

    def _get_seq(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        extract sentence and its target from document

        :param i: cur word index
        :return: ([seq len; 1], [seq len; 1])
        '''
        seq_len = min(self._seq_len, len(self._data) - 1 - i)
        data = self._data[i:i + seq_len]
        target = self._data[i + 1:i + 1 + seq_len]
        return data, target

    def get_n_samples(self):

        return self._total_n_samples


class TextDataset(Dataset):
    def __init__(self, data: torch.Tensor, seq_len: int):
        self._data = data
        self._seq_len = seq_len

    def __len__(self):
        return self._data.size(0) // self._seq_len

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        '''

        :return: ([seq len; 1], [seq len; 1])
        '''
        if (idx * self._seq_len >= self._data.size(0) - 1):
            raise StopIteration()
        data, target = self._get_seq(idx * self._seq_len)
        return data, target

    def _get_seq(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        extract sentence and its target from document

        :param i: cur word index
        :return: ([seq len; 1], [seq len; 1])
        '''
        seq_len = min(self._seq_len, len(self._data) - 1 - i)
        data = self._data[i:i + seq_len].squeeze(1)
        target = self._data[i + 1:i + 1 + seq_len].squeeze(1)
        return data, target

    def get_n_samples(self):
        return self._data.size(0) // self._seq_len
