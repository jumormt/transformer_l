import torch
import torchtext
from torchtext.data.utils import get_tokenizer
import pytorch_lightning as pl
from model import TransformerModelLightning
from torch.utils.data import DataLoader
from dataset import TextBatch, TextIterDataset, TextDataModule
from configs import get_transformer_config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from os.path import join


def train():
    # torch.autograd.set_detect_anomaly(True)
    config = get_transformer_config()
    model = TransformerModelLightning(config)
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateMonitor()

    tensorlogger = TensorBoardLogger("ts_logger", "transformer")
    # define early stopping callback
    early_stopping_callback = EarlyStopping(patience=3,
                                            monitor="val/loss",
                                            verbose=True,
                                            mode="min")
    # define model checkpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath=join(tensorlogger.log_dir, "{epoch:02d}-{val_loss:.4f}"),
        period=1,
    )
    trainer = pl.Trainer(max_epochs=3,
                         gpus=gpu,
                         gradient_clip_val=0.5,
                         log_every_n_steps=200,
                         check_val_every_n_epoch=1,
                         reload_dataloaders_every_epoch=True,
                         callbacks=[
                             lr_logger, early_stopping_callback,
                             model_checkpoint_callback
                         ],
                         logger=tensorlogger,
                         progress_bar_refresh_rate=1)
    trainer.fit(model, datamodule=TextDataModule(config))
    trainer.test()


def evaluate(checkpoint: str):
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModelLightning.load_from_checkpoint(
        checkpoint_path=checkpoint).to(device)
    data = TEXT.numericalize([test_txt.examples[0].text])
    seq_num = data.size(0) // 35
    data = data.narrow(0, 0, seq_num * 35 + 1)
    dataloader = DataLoader(TextIterDataset(data, 35, True),
                            batch_size=10,
                            collate_fn=TextBatch.collate_wrapper,
                            pin_memory=True)
    gpu = 1 if torch.cuda.is_available() else None

    # trainer = pl.Trainer(gpus=gpu)
    # trainer.test(model, test_dataloaders=[dataloader])
    model.eval()
    for batch in dataloader:
        X, Y = batch.X.to(device), batch.Y.to(device)
        res = model.translate(X)
        print(res)


if __name__ == "__main__":
    train()
    # evaluate("ts_logger/transformer/version_1/epoch=08-val_loss=8.1653.ckpt")
