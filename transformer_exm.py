import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torchtext
from torchtext.data.utils import get_tokenizer
from typing import Dict, Tuple, List
import time


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            device = src.device
            mask = self._generate_square_subsequent_mask(
                src.size(0)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


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


class TextIterDataset(IterableDataset):
    '''

    '''
    def __init__(self, data: torch.Tensor, seq_len: int):
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
        data = self._data[i:i + seq_len].squeeze(1)
        target = self._data[i + 1:i + 1 + seq_len].squeeze(1)
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


class TextBatch:
    def __init__(self, samples: List[Tuple[torch.Tensor, torch.Tensor]]):
        self.X = torch.cat(
            [sample[0] for sample in samples if (sample[0].size(0) == 35)],
            dim=1)
        self.Y = torch.cat(
            [sample[1] for sample in samples if (sample[1].size(0) == 35)],
            dim=1)

    def pin_memory(self):
        self.X = self.X.pin_memory()
        self.Y = self.Y.pin_memory()
        return self

    @staticmethod
    def collate_wrapper(
            batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> "TextBatch":
        return TextBatch(batch)


TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 20
eval_batch_size = 10
bptt = 35


def narrow_doc(txt):
    data = TEXT.numericalize([txt.examples[0].text])
    seq_num = data.size(0) // bptt
    data = data.narrow(0, 0, seq_num * bptt + 1)
    return data


# [total token]
train_data = narrow_doc(train_txt)
test_data = narrow_doc(test_txt)
val_data = narrow_doc(val_txt)

train_dataset = TextDataset(train_data, bptt)
val_dataset = TextDataset(val_data, bptt)
test_dataset = TextDataset(test_data, bptt)

ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
emsize = 200  # embedding dimension
nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers,
                         dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train():
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     pin_memory=True,
    #     collate_fn=TextBatch.collate_wrapper,
    # )
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  shuffle=True)
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)

    for i, batch in enumerate(train_dataloader):
        # data, targets = batch.X, batch.Y
        data, targets = batch
        data = data.t().to(device)
        targets = targets.t().contiguous().view(-1).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, i,
                      train_dataset.get_n_samples() // batch_size,
                      scheduler.get_lr()[0], elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, dataset):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    # val_dataloader = DataLoader(val_dataset,
    #                             batch_size=eval_batch_size,
    #                             pin_memory=True,
    #                             collate_fn=TextBatch.collate_wrapper)
    dataloader = DataLoader(dataset,
                            batch_size=eval_batch_size,
                            pin_memory=True,
                            shuffle=True)
    with torch.no_grad():
        for batch in dataloader:
            # data, targets = batch.X, batch.Y
            data, targets = batch
            data = data.t().to(device)
            targets = targets.t().contiguous().view(-1).to(device)
            output = eval_model(data)
            _, preds = output.max(dim=2)
            res = [[TEXT.vocab.itos[w.item()] for w in sen] for sen in preds]
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (dataset.get_n_samples() * bptt // eval_batch_size - 1)


best_val_loss = float("inf")
epochs = 3  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_dataset)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

test_loss = evaluate(best_model, test_dataset)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)