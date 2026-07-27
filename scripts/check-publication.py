#!/usr/bin/env python3
"""Reject credentials and machine-specific absolute paths in tracked files."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HOME_PREFIX = "/" + "home/"
MOUNT_PREFIX = "/" + "mnt/"
SECRET_PATTERNS = (
    re.compile("gh" + r"p_[A-Za-z0-9]{20,}"),
    re.compile("github" + r"_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"GITHUB_TOKEN\s*=\s*[^$\s][^\s]*"),
)


def _scan(raw: bytes, label: str, failures: list[str]) -> None:
    if b"\0" in raw:
        return
    text = raw.decode("utf-8", errors="replace")
    for number, line in enumerate(text.splitlines(), 1):
        if HOME_PREFIX in line or MOUNT_PREFIX in line:
            failures.append(f"{label}:{number}: machine-specific absolute path")
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                failures.append(f"{label}:{number}: possible credential")


def main() -> None:
    tracked = list(filter(None, subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "-z"]
    ).decode().split("\0")))
    failures: list[str] = []
    for relative in tracked:
        _scan((ROOT / relative).read_bytes(), f"index:{relative}", failures)

    revisions = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-list", "--all"]
    ).decode().splitlines()
    history_files = 0
    for revision in revisions:
        paths = filter(None, subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", "-z", revision]
        ).decode().split("\0"))
        for relative in paths:
            raw = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", f"{revision}:{relative}"]
            )
            _scan(raw, f"history:{revision[:12]}:{relative}", failures)
            history_files += 1

    if failures:
        raise SystemExit("publication check failed:\n" + "\n".join(failures))
    print(
        f"publication check passed: {len(tracked)} tracked files; "
        f"{len(revisions)} reachable commits; {history_files} historical file versions"
    )


if __name__ == "__main__":
    main()
