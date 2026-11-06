"""Milestone 1: causal filters, data contracts, targets and null forecasts."""

__version__ = "0.1.0"

from .smoothing import FilterSpec, CausalSmoother, smooth
from .targets import make_targets, mature_labels

__all__ = ["FilterSpec", "CausalSmoother", "smooth", "make_targets", "mature_labels"]
