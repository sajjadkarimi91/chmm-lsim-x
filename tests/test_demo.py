"""`examples/demo_run_lsim.py` runs end-to-end from a
clean install of the `init` extra (it calls `initialize_lsim_params`, which
needs scikit-learn -- see that file's own module docstring). This is a
smoke test only: it asserts the demo exits zero and prints its final
"Done" line, not that it runs within any particular wall-clock budget (the
demo's own per-stage timings are for human information, not a gate here)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_PATH = REPO_ROOT / "examples" / "demo_run_lsim.py"

try:
    import sklearn  # noqa: F401

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

requires_sklearn = pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="scikit-learn (the 'init' extra) is not installed -- the demo calls initialize_lsim_params, which needs it")


@requires_sklearn
def test_demo_run_lsim_exits_zero_and_completes():
    result = subprocess.run(
        [sys.executable, str(DEMO_PATH)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"demo exited with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    for expected in ("POISE  H =", "SED    H =", "FAST   H =", "EM: log-likelihood", "Viterbi states", "Done in"):
        assert expected in result.stdout, f"missing {expected!r} in demo output:\n{result.stdout}"


@requires_sklearn
def test_demo_run_chmm_exits_zero_and_completes():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "demo_run_chmm.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"demo exited with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    for expected in ("CHMM", "LSIM", "Exact CHMM Viterbi path", "Done in"):
        assert expected in result.stdout, f"missing {expected!r} in demo output:\n{result.stdout}"
