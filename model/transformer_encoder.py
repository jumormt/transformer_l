import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer
from dataset import TextBatch, TextDataset, TextIterDataset
from torchtext.data.utils import get_tokenizer
from model.modules import PositionalEncoding
import torchtext
from configs import TranformerEncoderConfig


class TransformerModelLightning(pl.LightningModule):
    def __init__(self, config: TranformerEncoderConfig):
        super().__init__()
        self.save_hyperparameters()
        TEXT = torchtext.data.Field(tokenize=get_tokenizer(config.data),
                                    init_token='<sos>',
                                    eos_token='<eos>',
                                    lower=True)
        train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(
            TEXT)
        TEXT.build_vocab(train_txt)
        self.TEXT = TEXT
        self.train_txt = train_txt
        self.val_txt = val_txt
        self.test_txt = test_txt
        self.ntoken = len(TEXT.vocab.stoi)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(config.ninp,
                                              config.dropout)
        encoder_layers = TransformerEncoderLayer(config.ninp,
                                                 config.nhead,
                                                 config.nhid,
                                                 config.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      config.nlayers)
        self.encoder = nn.Embedding(self.ntoken, config.ninp)
        self.ninp = config.ninp
        self.transformer = Transformer(d_model=config.ninp,
                                       nhead=config.nhead,
                                       num_encoder_layers=config.nlayers,
                                       num_decoder_layers=1,
                                       dim_feedforward=config.nhid,
                                       dropout=config.dropout)
        self.out = nn.Linear(config.ninp, self.ntoken)

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
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
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
