from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerEncoderConfig:
    data: str
    ninp: int
    nhead: int
    nhid: int
    nlayers: int
    dropout: float
