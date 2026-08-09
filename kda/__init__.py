"""An illustrated, executable explainer for Kimi Delta Attention.

Reference: *Kimi Linear: An Expressive, Efficient Attention Architecture*,
Kimi Team, arXiv:2510.26692.
"""

from kda.recurrent import VARIANTS, linear_attn, variant

__all__ = ["linear_attn", "variant", "VARIANTS"]
__version__ = "0.1.0"
