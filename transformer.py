import torch
import torchtext
from torchtext.data.utils import get_tokenizer
import pytorch_lightning as pl
from model import TransformerModelLightning
from torch.utils.data import DataLoader
from dataset import TextBatch, TextIterDataset
from configs import get_transformer_config
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateLogger
from os.path import join


def train():
    # torch.autograd.set_detect_anomaly(True)
    config = get_transformer_config()
    model = TransformerModelLightning(config)
    gpu = 1 if torch.cuda.is_available() else None
    # define learning rate logger
    lr_logger = LearningRateLogger()

    tensorlogger = TensorBoardLogger("ts_logger", "transformer")
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


if __name__ == "__main__":
    train()
    # evaluate("lightning_logs/version_0/checkpoints/epoch=2.ckpt")
