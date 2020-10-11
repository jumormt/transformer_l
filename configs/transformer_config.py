from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    srcdata: str
    tgtdata: str
    ninp: int
    nhead: int
    nhid: int
    nlayers: int
    dropout: float
