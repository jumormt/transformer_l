import torch
from torch.utils.data import IterableDataset, Dataset
from typing import List, Tuple, Optional
from math import ceil
from dataclasses import dataclass


@dataclass
class TextSample:

    data: torch.Tensor
    target: torch.Tensor


class TextBatch:
    def __init__(self, samples: List[TextSample]):
        self.X = torch.cat([sample.data for sample in samples], dim=1)
        self.Y = torch.cat([sample.target for sample in samples], dim=1)

    def pin_memory(self):
        self.X = self.X.pin_memory()
        self.Y = self.Y.pin_memory()
        return self

    def __len__(self) -> int:
        return self.X.size(1)

    def move_to_device(self, device: torch.device):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)


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
        data, target = self._get_seq(idx * self._seq_len)
        return TextSample(data=data, target=target)

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
        return self._data.size(0) // self._seq_len


from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from configs import TransformerConfig
import torchtext
from torchtext.data.utils import get_tokenizer


class TextDataModule(LightningDataModule):

    _train_dataset: TextDataset
    _val_dataset: TextDataset
    _test_dataset: TextDataset

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

    def _init_text(self, srcdata: str, tgtdata: str):
        self.EN_TEXT = torchtext.data.Field(tokenize=get_tokenizer(srcdata),
                                            init_token='<sos>',
                                            eos_token='<eos>',
                                            lower=True)
        self.en_train_txt, self.en_val_txt, self.en_test_txt = torchtext.datasets.WikiText2.splits(
            self.EN_TEXT)
        self.EN_TEXT.build_vocab(self.en_train_txt)
        self.envocab_l = len(self.EN_TEXT.vocab.stoi)

        self.FR_TEXT = torchtext.data.Field(tokenize=get_tokenizer(tgtdata),
                                            init_token='<sos>',
                                            eos_token='<eos>',
                                            lower=True)
        self.fr_train_txt, self.fr_val_txt, self.fr_test_txt = torchtext.datasets.WikiText2.splits(
            self.FR_TEXT)
        self.FR_TEXT.build_vocab(self.fr_train_txt)
        self.frvocab_l = len(self.FR_TEXT.vocab.stoi)

    def _narrow_doc(self, txt, bptt):
        data = self.EN_TEXT.numericalize([txt.examples[0].text])
        seq_num = data.size(0) // bptt
        data = data.narrow(0, 0, seq_num * bptt + 1)
        return data

    def prepare_data(self):
        self._init_text(self.config.srcdata, self.config.tgtdata)
        # [total token]
        self.train_data = self._narrow_doc(self.en_train_txt, 35)
        self.test_data = self._narrow_doc(self.en_test_txt, 35)
        self.val_data = self._narrow_doc(self.en_val_txt, 35)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self._train_dataset = TextDataset(self.train_data, 35)
            self._val_dataset = TextDataset(self.val_data, 35)
        else:
            self._test_dataset = TextDataset(self.test_data, 35)

    @staticmethod
    def collate_wrapper(
            batch: List[TextSample]) -> "TextBatch":
        return TextBatch(batch)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._train_dataset,
                          batch_size=20,
                          collate_fn=self.collate_wrapper,
                          num_workers=8,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._val_dataset,
                          batch_size=10,
                          num_workers=8,
                          collate_fn=self.collate_wrapper,
                          pin_memory=True)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self._test_dataset,
                          batch_size=10,
                          num_workers=8,
                          collate_fn=self.collate_wrapper,
                          pin_memory=True)

    def transfer_batch_to_device(self, batch: TextBatch,
                                 device: torch.device) -> TextBatch:
        batch.move_to_device(device)
        return batch
