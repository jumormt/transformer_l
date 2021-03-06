from .transformer_encoder_config import TransformerEncoderConfig
from .transformer_config import TransformerConfig


def get_transformer_config() -> TransformerConfig:
    ninp = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    return TransformerConfig(srcdata="basic_english",
                             tgtdata="basic_english",
                             ninp=ninp,
                             nhead=nhead,
                             nhid=nhid,
                             nlayers=nlayers,
                             dropout=dropout)


def get_transformer_encoder_config() -> TransformerEncoderConfig:
    ninp = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    return TransformerEncoderConfig(data="basic_english",
                                    ninp=ninp,
                                    nhead=nhead,
                                    nhid=nhid,
                                    nlayers=nlayers,
                                    dropout=dropout)
