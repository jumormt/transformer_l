import torch
import torchtext
from torchtext.data.utils import get_tokenizer
import pytorch_lightning as pl
from model import TransformerModelEncoderLightning
from torch.utils.data import DataLoader
from dataset import TextBatch, TextIterDataset
from configs import get_transformer_encoder_config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
from os.path import join


def train():
    config = get_transformer_encoder_config()
    model = TransformerModelEncoderLightning(config)
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateLogger()

    tensorlogger = TensorBoardLogger("ts_logger", "transformer_encoder")
    # define early stopping callback
    early_stopping_callback = EarlyStopping(patience=3,
                                            verbose=True,
                                            mode="min")
    # define model checkpoint callback
    model_checkpoint_callback = ModelCheckpoint(
        filepath=join(tensorlogger.log_dir, "{epoch:02d}-{val_loss:.4f}"),
        period=1,
        save_top_k=3,
    )
    trainer = pl.Trainer(max_epochs=10,
                         gpus=gpu,
                         gradient_clip_val=0.5,
                         row_log_interval=200,
                         check_val_every_n_epoch=1,
                         reload_dataloaders_every_epoch=True,
                         callbacks=[lr_logger],
                         logger=tensorlogger,
                         checkpoint_callback=model_checkpoint_callback,
                         early_stop_callback=early_stopping_callback,
                         progress_bar_refresh_rate=1)
    trainer.fit(model)


def evaluate(checkpoint: str):
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)
    train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
    TEXT.build_vocab(train_txt)
    model = TransformerModelEncoderLightning.load_from_checkpoint(
        checkpoint_path=checkpoint)
    data = TEXT.numericalize([test_txt.examples[0].text])
    seq_num = data.size(0) // 35
    data = data.narrow(0, 0, seq_num * 35 + 1)
    dataloader = DataLoader(TextIterDataset(data, 35, True),
                            batch_size=10,
                            collate_fn=TextBatch.collate_wrapper,
                            pin_memory=True)
    gpu = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(gpus=gpu)
    trainer.test(model, test_dataloaders=[dataloader])


if __name__ == "__main__":
    train()
    # evaluate("lightning_logs/version_0/checkpoints/epoch=2.ckpt")
