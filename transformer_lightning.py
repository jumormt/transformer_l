import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
import torchtext
from torchtext.data.utils import get_tokenizer
from typing import Dict, Tuple, List
import time
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer


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


class TransformerModelLightning(pl.LightningModule):
    def __init__(self,
                 TEXT,
                 train_txt,
                 val_txt,
                 test_txt,
                 ninp,
                 nhead,
                 nhid,
                 nlayers,
                 dropout=0.5):
        super().__init__()
        self.TEXT = TEXT
        self.train_txt = train_txt
        self.val_txt = val_txt
        self.test_txt = test_txt
        self.ntoken = len(TEXT.vocab.stoi)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(self.ntoken, ninp)
        self.ninp = ninp
        self.transformer = Transformer(d_model=ninp,
                                       nhead=nhead,
                                       num_encoder_layers=nlayers,
                                       num_decoder_layers=1,
                                       dim_feedforward=nhid,
                                       dropout=dropout)
        self.out = nn.Linear(ninp, self.ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)

    # ===== OPTIMIZERS =====
    def configure_optimizers(self):
        lr = 5.0  # learning rate
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return [optimizer], [scheduler]

    # ===== PREPARE DATA =====
    def narrow_doc(self, txt, bptt):
        data = self.TEXT.numericalize([txt.examples[0].text])
        seq_num = data.size(0) // bptt
        data = data.narrow(0, 0, seq_num * bptt + 1)
        return data

    def prepare_data(self):
        # [total token]
        self.train_data = self.narrow_doc(self.train_txt, 35)
        self.test_data = self.narrow_doc(self.test_txt, 35)
        self.val_data = self.narrow_doc(self.val_txt, 35)

    # ===== DATALOADERS BLOCK =====
    def train_dataloader(self):
        return DataLoader(TextIterDataset(self.train_data, 35, True),
                          batch_size=20,
                          collate_fn=TextBatch.collate_wrapper,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(TextIterDataset(self.val_data, 35),
                          batch_size=10,
                          collate_fn=TextBatch.collate_wrapper,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(TextIterDataset(self.test_data, 35),
                          batch_size=10,
                          collate_fn=TextBatch.collate_wrapper,
                          pin_memory=True)

    # ===== STEP =====
    def training_step(self, batch: TextBatch, idx: int) -> Dict:
        data, targets = batch.X, batch.Y
        logits = self(data)
        loss = nn.CrossEntropyLoss()(logits.view(-1, self.ntoken),
                                     targets.view(-1))
        logs = {"ptl/train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch: TextBatch, idx: int) -> Dict:
        data, targets = batch.X, batch.Y
        logits = self(data)
        loss = nn.CrossEntropyLoss()(logits.view(-1, self.ntoken),
                                     targets.view(-1))
        logs = {"ptl/val_loss": loss}
        return {"val_loss": loss, "log": logs}

    def test_step(self, batch: TextBatch, batch_idx: int) -> Dict:
        result = self.validation_step(batch, batch_idx)
        result["test_loss"] = result["val_loss"]
        del result["val_loss"]
        return result

    # ===== ON EPOCH END =====
    def training_epoch_end(self, outputs: List[Dict]) -> Dict:
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"ptl/train_loss": avg_loss}

        return {"loss": avg_loss, "log": logs}

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"ptl/val_loss": avg_loss}

        return {"val_loss": avg_loss, "log": logs}

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"ptl/test_loss": avg_loss}

        return {"test_loss": avg_loss, "log": logs}

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(
                src.size(0)).to(device)
            self.src_mask = mask
        # [seq len; batch size; embdding size]
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.out(output)
        return output

    def transfer_batch_to_device(self, batch: TextBatch,
                                 device: torch.device) -> TextBatch:
        # [total word length, input size]
        batch.X = batch.X.to(device)
        batch.Y = batch.Y.to(device)
        return batch


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


if __name__ == "__main__":

    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)
    emsize = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    model = TransformerModelLightning(TEXT, train_txt, val_txt, test_txt,
                                      emsize, nhead, nhid, nlayers, dropout)
    gpu = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=3,
        show_progress_bar=False,
        gpus=gpu,
        gradient_clip_val=0.5,
        row_log_interval=200,
        check_val_every_n_epoch=1,
        reload_dataloaders_every_epoch=True,
    )
    trainer.fit(model)
