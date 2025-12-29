"""
Clone-Structured Cognitive Graphs (CSCG) implementation.

Based on: https://github.com/vicariousinc/naturecomm_cscg
Paper: "Learning cognitive maps as structured graphs for vicarious evaluation"

Provides:
- CHMM: Clone-structured Hidden Markov Model
- EmbeddingTokenizer: CLIP embedding to discrete token conversion
- CSCGWorldModel: POMDP-compatible world model using CSCG
"""

from .chmm import CHMM
from .tokenizer import EmbeddingTokenizer
from .cscg_world_model import CSCGWorldModel

__all__ = ['CHMM', 'EmbeddingTokenizer', 'CSCGWorldModel']
