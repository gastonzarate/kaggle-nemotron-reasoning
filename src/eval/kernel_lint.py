"""Static validator for Kaggle kernel notebooks.

Catches the most common "I forgot a source" or "wrong mount path" bugs WITHOUT pushing the
kernel and waiting for Kaggle to fail.

Validates:
- Every `/kaggle/input/notebooks/<owner>/<slug>/...` referenced in code cells must have
  `<owner>/<slug>` in kernel_sources of the metadata.
- Every `/kaggle/usr/lib/notebooks/<owner>/<slug>/...` referenced must also have a
  matching kernel_source (Kaggle replaces hyphens in slug with underscores in the mount,
  so we normalize both).
- Every `/kaggle/input/datasets/<owner>/<slug>/...` referenced must have `<owner>/<slug>`
  in dataset_sources.
- Every `/kaggle/input/<top>/...` where `<top>` is just a slug must match either a dataset
  slug or competition slug (Kaggle's "flat" mount style for user datasets and competition
  data).

Doesn't run the notebook — pure regex over cell source.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

KAGGLE_PATH_RE = re.compile(r'["\']/(kaggle/[^"\'\s{}]+)["\']')

GLOB_CHARS = ("*", "?", "[")

# Paths that only appear inside error messages or in scorer-only runtimes;
# safe to ignore in user-pushed kernels.
WHITELIST_PREFIXES = (
    "/kaggle/input/competition_evaluation",  # only mounted server-side during scoring
)


@dataclass
class LintIssue:
    severity: str  # "error" or "warn"
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.path}: {self.message}"


def _slug_to_underscore(slug: str) -> str:
    return slug.replace("-", "_")


def extract_paths_from_notebook(nb_path: Path) -> set[str]:
    """Return every /kaggle/... string literal that appears in code cells."""
    nb = json.loads(nb_path.read_text())
    found: set[str] = set()
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        for m in KAGGLE_PATH_RE.finditer(src):
            found.add("/" + m.group(1))
    return found


def lint_kernel(kernel_dir: Path) -> list[LintIssue]:
    issues: list[LintIssue] = []

    meta_path = kernel_dir / "kernel-metadata.json"
    if not meta_path.exists():
        return [LintIssue("error", str(meta_path), "kernel-metadata.json missing")]
    meta = json.loads(meta_path.read_text())

    nb_name = meta.get("code_file")
    if not nb_name:
        return [LintIssue("error", str(meta_path), "code_file missing in metadata")]
    nb_path = kernel_dir / nb_name
    if not nb_path.exists():
        return [LintIssue("error", str(nb_path), f"code_file '{nb_name}' not found in {kernel_dir}")]

    ksrc = set(meta.get("kernel_sources", []))
    dsrc = set(meta.get("dataset_sources", []))
    csrc = set(meta.get("competition_sources", []))
    msrc = set(meta.get("model_sources", []))

    ksrc_owner_slug = {tuple(s.split("/", 1)) for s in ksrc if "/" in s}
    ksrc_owner_slug_underscored = {(o, _slug_to_underscore(s)) for o, s in ksrc_owner_slug}
    dsrc_owner_slug = {tuple(s.split("/", 1)) for s in dsrc if "/" in s}
    dsrc_slugs = {s.split("/", 1)[1] for s in dsrc if "/" in s}

    paths = extract_paths_from_notebook(nb_path)

    for p in sorted(paths):
        if any(g in p for g in GLOB_CHARS):
            continue  # glob pattern, not a real path
        if any(c in p for c in ("{", "}", "\\")):
            continue  # f-string artifact or escape sequence
        if any(p.startswith(w) for w in WHITELIST_PREFIXES):
            continue  # known scorer-only path
        if p.startswith("/kaggle/input/notebooks/"):
            parts = p[len("/kaggle/input/notebooks/"):].split("/")
            if len(parts) >= 2:
                owner, slug = parts[0], parts[1]
                if (owner, slug) not in ksrc_owner_slug:
                    issues.append(LintIssue(
                        "error", p,
                        f"references kernel {owner}/{slug} but it's not in kernel_sources={sorted(ksrc)}",
                    ))

        elif p.startswith("/kaggle/usr/lib/notebooks/"):
            parts = p[len("/kaggle/usr/lib/notebooks/"):].split("/")
            if len(parts) >= 2:
                owner, slug_us = parts[0], parts[1]
                if (owner, slug_us) not in ksrc_owner_slug_underscored:
                    issues.append(LintIssue(
                        "error", p,
                        f"references utility kernel {owner}/{slug_us} but matching kernel_source missing. Have: {sorted(ksrc)}",
                    ))

        elif p.startswith("/kaggle/input/datasets/"):
            parts = p[len("/kaggle/input/datasets/"):].split("/")
            if len(parts) >= 2:
                owner, slug = parts[0], parts[1]
                if (owner, slug) not in dsrc_owner_slug:
                    issues.append(LintIssue(
                        "error", p,
                        f"references dataset {owner}/{slug} but it's not in dataset_sources={sorted(dsrc)}",
                    ))

        elif p.startswith("/kaggle/input/models/"):
            parts = p[len("/kaggle/input/models/"):].split("/")
            if len(parts) >= 2:
                owner_model = "/".join(parts[:2])
                matched = any(s.lower().startswith(f"{owner_model.lower()}/") for s in msrc)
                if not matched:
                    issues.append(LintIssue(
                        "warn", p,
                        f"references model {owner_model} but no matching model_source. Have: {sorted(msrc)}",
                    ))

        elif p.startswith("/kaggle/input/"):
            top = p[len("/kaggle/input/"):].split("/")[0]
            if (top
                and top not in {"notebooks", "datasets", "models", "competitions"}
                and top not in dsrc_slugs
                and top not in csrc
            ):
                issues.append(LintIssue(
                    "error", p,
                    f"references /kaggle/input/{top}/ — not in dataset_sources slugs={sorted(dsrc_slugs)} "
                    f"or competition_sources={sorted(csrc)}",
                ))

        elif p.startswith("/kaggle/working/") or p.startswith("/kaggle/tmp"):
            pass  # ephemeral runtime paths, no validation
        else:
            issues.append(LintIssue("warn", p, "unrecognized /kaggle/... prefix; check manually"))

    return issues


def main(kernel_dirs: list[Path]) -> int:
    total_errors = 0
    for kdir in kernel_dirs:
        print(f"\n=== {kdir} ===")
        issues = lint_kernel(kdir)
        errors = [i for i in issues if i.severity == "error"]
        warns = [i for i in issues if i.severity == "warn"]
        for i in issues:
            print(f"  {i}")
        if not issues:
            print("  ✓ clean")
        else:
            print(f"  → {len(errors)} errors, {len(warns)} warnings")
        total_errors += len(errors)
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src.eval.kernel_lint <kernel_dir> [kernel_dir...]")
        sys.exit(2)
    sys.exit(main([Path(p) for p in sys.argv[1:]]))
