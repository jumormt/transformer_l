import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Transformer, TransformerDecoder, TransformerDecoderLayer
from dataset import TextBatch, TextDataset, TextIterDataset
from torchtext.data.utils import get_tokenizer
from model.modules import PositionalEncoding, LuongAttention
import torchtext
from configs import TransformerConfig


class TransformerModelLightning(pl.LightningModule):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self._init_text(config.srcdata, config.tgtdata)
        self.ninp = config.ninp
        self.model_type = 'Transformer'
        self._init_encoder()
        self._init_decoder()
        self.norm = nn.LayerNorm(config.ninp)
        self.out = nn.Linear(config.ninp, self.frvocab_l, bias=False)

        self._init_weights()

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

    def _init_encoder(self):
        self.src_mask = None
        self.encoder_embed = nn.Embedding(self.envocab_l, self.config.ninp)
        self.pos_encoder = PositionalEncoding(self.config.ninp,
                                              self.config.dropout)
        encoder_layers = TransformerEncoderLayer(self.config.ninp,
                                                 self.config.nhead,
                                                 self.config.nhid,
                                                 self.config.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
                                                      self.config.nlayers)

    def _init_decoder(self):
        self.tgt_mask = None
        self.decoder_embed = nn.Embedding(self.frvocab_l, self.config.ninp)
        self.pos_decoder = PositionalEncoding(self.config.ninp,
                                              self.config.dropout)
        decoder_layers = TransformerDecoderLayer(self.config.ninp,
                                                 self.config.nhead,
                                                 self.config.nhid,
                                                 self.config.dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers,
                                                      self.config.nlayers)
        self.decoder_lstm = nn.LSTM(
            self.config.ninp,
            self.config.nhid,
            num_layers=1,
            batch_first=True,
        )
        self.dropout_rnn = nn.Dropout(self.config.dropout)
        self.attention = LuongAttention(self.config.nhid)
        self.concat_layer = nn.Linear(self.config.nhid * 2,
                                      self.config.nhid,
                                      bias=False)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def _init_weights(self):
        initrange = 0.1
        self.encoder_embed.weight.data.uniform_(-initrange, initrange)
        self.decoder_embed.weight.data.uniform_(-initrange, initrange)
        # self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)

    # ===== OPTIMIZERS =====
    def configure_optimizers(self):
        lr = 5.0  # learning rate
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        return [optimizer], [scheduler]

    # ===== PREPARE DATA =====
    def _narrow_doc(self, txt, bptt):
        data = self.EN_TEXT.numericalize([txt.examples[0].text])
        seq_num = data.size(0) // bptt
        data = data.narrow(0, 0, seq_num * bptt + 1)
        return data

    def prepare_data(self):
        # [total token]
        self.train_data = self._narrow_doc(self.en_train_txt, 35)
        self.test_data = self._narrow_doc(self.en_test_txt, 35)
        self.val_data = self._narrow_doc(self.en_val_txt, 35)

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
        logits = self(data, targets)
        # loss = nn.CrossEntropyLoss()(logits.view(-1, self.ntoken),
        #                              targets.view(-1))
        loss = self._calculate_loss(logits, targets)
        logs = {"ptl/train_loss": loss}
        progress_bar = {"train/loss": loss}
        return {"loss": loss, "log": logs, "progress_bar": progress_bar}

    def validation_step(self, batch: TextBatch, idx: int) -> Dict:
        data, targets = batch.X, batch.Y
        logits = self(data)
        # loss = nn.CrossEntropyLoss()(logits.view(-1, self.ntoken),
        #                              targets.view(-1))
        loss = self._calculate_loss(logits, targets)
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
        progress_bar = {"train/loss": avg_loss}

        return {"loss": avg_loss, "log": logs, "progress_bar": progress_bar}

    def validation_epoch_end(self, outputs: List[Dict]) -> Dict:
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        progress_bar = {"val/loss": avg_loss}

        return {
            "val_loss": avg_loss,
            "log": logs,
            "progress_bar": progress_bar
        }

    def test_epoch_end(self, outputs: List[Dict]) -> Dict:
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"ptl/test_loss": avg_loss}

        return {"test_loss": avg_loss, "log": logs}

    def _calculate_loss(self, logits: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
        """Calculate cross entropy with ignoring PAD index

        :param logits: [seq length; batch size; vocab size]
        :param labels: [seq length; batch size]
        :return: [1]
        """
        batch_size = labels.shape[-1]
        # [batch size; vocab size; seq length]
        _logits = logits.permute(1, 2, 0)
        # [batch size; seq length]
        _labels = labels.permute(1, 0)
        # [batch size; seq length]
        loss = F.cross_entropy(_logits, _labels, reduction="none")
        # [batch size; seq length]
        mask = _labels != self.FR_TEXT.vocab.stoi["<pad>"]
        # [batch size; seq length]
        loss = loss * mask
        # [1]
        loss = loss.sum() / batch_size
        return loss

    def forward(
            self,
            src: torch.Tensor,  # [seq len; batch size]
            target_sequence: torch.Tensor = None,  # [seq len; batch size]
    ) -> torch.Tensor:
        batch_size = src.size(1)

        # [sen len; sen len]
        # src_mask = (src != self.EN_TEXT.vocab.stoi["<pad>"])
        # src_mask = src_mask.float().masked_fill(src_mask == 0, float('-inf')).masked_fill(
        #     src_mask == 1, float(0.0)).to(src.device)
        # if self.src_mask is None or self.src_mask.size(0) != src.size(0):
        #     self.src_mask = self._generate_square_subsequent_mask(
        #         src.size(0)).to(src.device)

        # [seq len; batch size; embdding size]
        src = self.encoder_embed(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        # [seq len; batch size; hidden]
        memory = self.transformer_encoder(src)

        if self.tgt_mask is None or self.tgt_mask.size(0) != src.size(0):
            self.tgt_mask = self._generate_square_subsequent_mask(
                src.size(0)).to(src.device)
        # [seq len; batch size; hidden]
        decoder_output = self.transformer_decoder(src, memory, self.tgt_mask)

        # [seq len; batch size; fr vovab len]
        outputs = self.out(decoder_output)

        # # [seq len; batch size]
        # outputs_tmp = src.new_zeros((35, batch_size), dtype=torch.long)
        # # [seq len; batch size; fr vocab size]
        # outputs = src.new_zeros((35, batch_size, self.frvocab_l))
        # # [1; batch size]
        # cur = src.new_full((1, batch_size),
        #                    self.FR_TEXT.vocab.stoi['<sos>'],
        #                    dtype=torch.long)
        # for step in range(35):

        #     decoder_output = self.decoder_step(cur, memory)
        #     # [batch size]
        #     outputs_tmp[step] = decoder_output[-1].argmax(-1)
        #     # [batch size; fr vocab len]
        #     outputs[step] = decoder_output[-1]
        #     # [step + 2; batch size]
        #     cur = outputs_tmp[:step + 1]

        # # [1; batch size; hidden]
        # initial_state = memory.mean(0).unsqueeze(0)
        # memory = memory.permute(1, 0, 2)
        # h_prev, c_prev = initial_state, initial_state

        # outputs = src.new_zeros((35, batch_size, self.frvocab_l))
        # # [batch size]
        # current_input = src.new_full((batch_size, ),
        #                              self.FR_TEXT.vocab.stoi['<sos>'],
        #                              dtype=torch.long)
        # for step in range(35):
        #     current_output, (h_prev, c_prev) = self.decoder_lstm_step(
        #         current_input, h_prev, c_prev, memory)
        #     outputs[step] = current_output
        #     if target_sequence is not None and torch.rand(1) < 1:
        #         current_input = target_sequence[step]
        #     else:
        #         current_input = outputs[step].argmax(dim=-1)

        return outputs

    def translate(self, src: torch.Tensor) -> torch.Tensor:
        '''
        translate src to tgt

        :param src: [seq len; batch size]
        :return: [batch size; out seq len]
        '''
        batch_size = src.size(1)

        # [seq len; batch size; embdding size]
        src = self.encoder_embed(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        # [seq len; batch size; hidden]
        memory = self.transformer_encoder(src)

        if self.tgt_mask is None or self.tgt_mask.size(0) != src.size(0):
            self.tgt_mask = self._generate_square_subsequent_mask(
                src.size(0)).to(src.device)

        # [out seq len; batch size]
        outputs_tmp = src.new_zeros((40, batch_size), dtype=torch.long)
        # [out seq len; batch size; fr vocab size]
        outputs = src.new_zeros((40, batch_size, self.frvocab_l))
        # [1; batch size]
        cur = src.new_full((1, batch_size),
                           self.FR_TEXT.vocab.stoi['<sos>'],
                           dtype=torch.long)
        for step in range(40):
            # [step + 1; batch size]
            decoder_output = self.decoder_step(cur, memory)
            # [batch size]
            outputs_tmp[step] = decoder_output[-1].argmax(-1)
            # [batch size; fr vocab len]
            outputs[step] = decoder_output[-1]
            # [step + 2; batch size]
            cur = outputs_tmp[:step + 1]

        # [batch size; out seq len; fr vocab size]
        outputs = outputs.permute(1, 0, 2)

        with torch.no_grad():

            # [batch size; out seq len]
            outidx = outputs.argmax(dim=2)
            # [batch size; out seq len]
            res = [[self.EN_TEXT.vocab.itos[w.item()] for w in sen] for sen in outidx]




        return res

    def decoder_lstm_step(
            self,
            current_input: torch.Tensor,
            h_prev: torch.Tensor,
            c_prev: torch.Tensor,
            memory: torch.Tensor  # [batch size; seq len; decoder size]
    ):
        '''
        using lstm as decoder

        :param current_input: [batch size]
        :param h_prev: [1; batch size; decoder size (hidden)]
        :param c_prev: [1; batch size; decoder size (hidden)]
        '''
        # [batch size; 1; embed size(ninp)]
        embedded = self.decoder_embed(current_input).unsqueeze(1) * math.sqrt(
            self.ninp)

        # h_prev -- [n layers(1); batch size; decoder size]
        # rnn_output -- [batch size; 1; decoder size]
        rnn_output, (h_prev,
                     c_prev) = self.decoder_lstm(embedded, (h_prev, c_prev))

        # [batch size; seq len]
        attn_weights = self.attention(h_prev[-1], memory)

        # [batch size; 1; decoder size]
        context = torch.bmm(attn_weights.unsqueeze(1), memory)

        # [batch size; 2 * decoder size]
        concat_input = torch.cat([rnn_output, context], dim=2).squeeze(1)

        # [batch size; decoder size]
        concat = self.concat_layer(concat_input)
        concat = self.norm(concat)
        concat = torch.tanh(concat)

        # [batch size; fr vocab len]
        output = self.out(concat)
        return output, (h_prev, c_prev)

    def decoder_step(self, input_tokens: torch.Tensor,
                     memory: torch.Tensor) -> torch.Tensor:

        # [step + 1; batch size; embed size]
        embeded = self.decoder_embed(input_tokens) * math.sqrt(self.ninp)
        embeded = self.pos_decoder(embeded)
        if self.tgt_mask is None or self.tgt_mask.size(0) != embeded.size(0):
            self.tgt_mask = self._generate_square_subsequent_mask(
                embeded.size(0)).to(embeded.device)
        decoder_output = self.transformer_decoder(embeded, memory,
                                                  self.tgt_mask)
        # [step + 1; batch size; fr vocab len]
        decoder_output = self.out(decoder_output)

        return decoder_output

    def transfer_batch_to_device(self, batch: TextBatch,
                                 device: torch.device) -> TextBatch:
        # [total word length, input size]
        batch.X = batch.X.to(device)
        batch.Y = batch.Y.to(device)
        return batch
