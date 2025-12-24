"""
POMDP-based world models for Tello drone navigation.

Three layered POMDPs:
1. World Model - Learned topological map for localization
2. Human Search - Belief over person locations
3. Interaction Mode - Action selection via Expected Free Energy
"""

from .config import *
