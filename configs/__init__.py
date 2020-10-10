from .transformer_encoder_config import TranformerEncoderConfig


def get_transformer_encoder_config() -> TranformerEncoderConfig:
    ninp = 200  # embedding dimension
    nhid = 200  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    return TranformerEncoderConfig(data="basic_english",
                                   ninp=ninp,
                                   nhead=nhead,
                                   nhid=nhid,
                                   nlayers=nlayers,
                                   dropout=dropout)
