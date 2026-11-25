"""Commit and push the data branch from inside the run.

The old workflow published once, after a single short pass. The workflow now loops for hours, so each pass
publishes as it goes: the dashboard must not wait for the job to finish. Safe to call when nothing changed
(it does nothing), and safe to call twice in a row.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

IDENTITY = ["-c", "user.name=paper-trading[bot]", "-c", "user.email=paper-trading@users.noreply.github.com"]


def _git(data_dir, *args, check=True):
    return subprocess.run(["git", *IDENTITY, *args], cwd=data_dir, capture_output=True, text=True, check=check)


def publish(data_dir, branch="paper-data", remote="origin", retries=3, log=print) -> bool:
    """git add -A; commit if anything changed; push (rebasing on a rejected push). True if a commit was pushed."""
    _git(data_dir, "add", "-A")
    if _git(data_dir, "diff", "--cached", "--quiet", check=False).returncode == 0:
        return False
    _git(data_dir, "commit", "-m", f"paper data {datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}")
    for attempt in range(1, retries + 1):
        res = _git(data_dir, "push", remote, branch, check=False)
        if res.returncode == 0:
            return True
        log(f"  publish: push failed (attempt {attempt}/{retries}): {res.stderr.strip().splitlines()[-1:]}")
        _git(data_dir, "pull", "--rebase", remote, branch, check=False)
    raise RuntimeError(f"could not push {branch} after {retries} attempts")
