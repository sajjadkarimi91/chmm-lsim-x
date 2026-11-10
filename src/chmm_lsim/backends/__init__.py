"""Execution backends: how the inference math runs (internal module; import
from chmm_lsim). No algorithm lives here -- see `chmm_lsim.models.lsim.inference`
for that. Each backend declares what it supports in `BackendCapabilities`."""
