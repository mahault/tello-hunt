"""
Mapping module for CSCG + VBGS integration.

Provides:
- CSCG (Clone-Structured Cognitive Graphs) for topological mapping
- VBGS place models for local appearance modeling
"""

from .cscg import CHMM, EmbeddingTokenizer, CSCGWorldModel

__all__ = ['CHMM', 'EmbeddingTokenizer', 'CSCGWorldModel']
