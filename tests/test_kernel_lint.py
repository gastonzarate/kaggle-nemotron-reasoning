"""Static checks for the kernels we push to Kaggle.

Saves about 10 minutes per "push and discover a path bug" iteration. Run before every push.
"""

from pathlib import Path

import pytest

from src.eval.kernel_lint import lint_kernel

ROOT = Path(__file__).resolve().parents[1]
MY_KERNELS = [
    ROOT / "notebooks" / "my_kernels" / "eval-holdout-baseline-v1",
    ROOT / "notebooks" / "my_kernels" / "eval-holdout-structured-v1",
]


@pytest.mark.parametrize("kernel_dir", MY_KERNELS, ids=lambda p: p.name)
def test_kernel_metadata_matches_referenced_paths(kernel_dir: Path):
    issues = lint_kernel(kernel_dir)
    errors = [i for i in issues if i.severity == "error"]
    assert not errors, "Kernel-source / dataset-source mismatch:\n" + "\n".join(str(e) for e in errors)
