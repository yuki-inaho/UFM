#!/usr/bin/env python3
"""Pre-stage privacy guard for the public UFM repository.

A private paired-capture dataset and its derived artifacts (checkpoints,
media, credentials, host-local paths) must never be staged into this public
repo.
This stdlib-only CLI inspects *candidate* paths and rejects anything that
looks private.

Modes
-----
* default        inspect the current repository's staged index entries with
                 statuses A/C/M/R and read each entry's index blob via
                 ``git show :<path>``. The working-tree file is never used,
                 so a later working-tree edit cannot disguise staged content.
* ``--paths``    inspect the listed working-tree paths directly (filesystem).

Exit codes
----------
0  clean
1  at least one violation was found
2  usage / repository error (could not determine what to inspect)
"""

import argparse
import os
import re
import subprocess
import sys

EXIT_OK = 0
EXIT_VIOLATION = 1
EXIT_USAGE = 2

DEFAULT_MAX_SIZE_BYTES = 1024 * 1024  # 1 MiB

DENY_EXTENSIONS = frozenset(
    {
        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp",
        ".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx",
        ".tar", ".tgz", ".zst", ".zip", ".gz", ".bz2", ".xz", ".7z", ".rar",
        ".h5", ".hdf5", ".pkl", ".pickle", ".bin", ".dat", ".parquet",
        ".feather", ".mp4", ".mov", ".avi", ".mkv",
    }
)

PRIVATE_DIRS = frozenset({"data", "checkpoints", "outputs", "runs", "private"})

ALLOWED_EXTENSIONS = frozenset(
    {
        ".py", ".pyi", ".pyx", ".txt", ".md", ".markdown", ".rst",
        ".json", ".jsonl", ".yaml", ".yml", ".toml", ".cfg", ".ini",
        ".conf", ".sh", ".bash", ".zsh", ".fish", ".c", ".h", ".cc",
        ".cpp", ".cxx", ".hpp", ".hh", ".java", ".kt", ".go", ".rs",
        ".js", ".jsx", ".ts", ".tsx", ".css", ".scss", ".html", ".htm",
        ".xml", ".sql", ".proto", ".cmake", ".mk", ".lock",
    }
)

ALLOWED_BASENAMES = frozenset(
    {
        "LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS",
        "CONTRIBUTORS", "README", "CHANGELOG", "Makefile", "makefile",
        "GNUmakefile", "Dockerfile", "dockerfile", "CODEOWNERS",
        ".gitignore", ".gitattributes", ".editorconfig", ".dockerignore",
        ".env.example", ".flake8", ".pre-commit-config.yaml",
    }
)

# Host-local markers are assembled from fragments so this source file itself
# does not contain the raw rejected sequences.
HOST_PATH_MARKERS = (
    "/" + "home/",
    "/" + "workspace/",
    "C:" + "\\" + "Users" + "\\",
)

SECRET_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])sk" + r"-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"ghp" + r"_[A-Za-z0-9]{20,}"),
    re.compile(r"gho" + r"_[A-Za-z0-9]{20,}"),
    re.compile(r"ghs" + r"_[A-Za-z0-9]{20,}"),
    re.compile(r"xoxb" + r"-[A-Za-z0-9\-]{10,}"),
    re.compile(r"AKIA" + r"[0-9A-Z]{16}"),
    re.compile(r"OPENROUTER_API_KEY" + r"\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"ANTHROPIC_API_KEY" + r"\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
)


def _is_text(data):
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def analyze(path, size, is_symlink, content, max_size_bytes):
    """Return a list of (code, detail) violations for one candidate path."""
    violations = []

    norm = path.replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    parts = [p for p in norm.split("/") if p not in ("", ".")]
    basename = parts[-1] if parts else norm
    dirs = parts[:-1]
    lower = basename.lower()
    ext = os.path.splitext(lower)[1]

    # Privacy artifact directories are root-anchored: only a top-level
    # directory of that name is private. A nested "foo/data/x.py" is not
    # treated as a private artifact directory.
    if dirs and dirs[0].lower() in PRIVATE_DIRS:
        violations.append(
            ("PRIVATE_DIR", "lies under private artifact directory '%s'" % dirs[0])
        )

    if ext in DENY_EXTENSIONS:
        violations.append(("DENY_EXT", "extension '%s' is not allowed" % ext))
    elif ext not in ALLOWED_EXTENSIONS and basename not in ALLOWED_BASENAMES:
        violations.append(
            ("EXT_NOT_ALLOWED", "extension '%s' is not on the allowlist" % (ext or "<none>"))
        )

    if is_symlink:
        violations.append(("SYMLINK", "symbolic links are not allowed"))

    if size is not None and size > max_size_bytes:
        violations.append(
            ("OVERSIZE", "size %d bytes exceeds limit %d bytes" % (size, max_size_bytes))
        )

    if content is not None and _is_text(content):
        text = content.decode("utf-8", "replace")
        for marker in HOST_PATH_MARKERS:
            if marker in text:
                violations.append(("HOST_PATH", "contains a host-local absolute path"))
                break
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                violations.append(("SECRET", "contains a plausible secret literal"))
                break

    return violations


def _repo_root():
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _iter_staged_entries(repo_root):
    proc = subprocess.run(
        [
            "git", "-C", repo_root, "diff", "--cached", "--name-status", "-z",
            "--diff-filter=ACMR",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip())
    tokens = proc.stdout.split(b"\0")
    i = 0
    while i < len(tokens):
        if tokens[i] == b"":
            i += 1
            continue
        status = tokens[i].decode("utf-8", "replace")
        code = status[0]
        if code in ("R", "C"):
            if i + 2 >= len(tokens):
                break
            new_path = tokens[i + 2].decode("utf-8", "replace")
            i += 3
        else:
            if i + 1 >= len(tokens):
                break
            new_path = tokens[i + 1].decode("utf-8", "replace")
            i += 2
        yield code, new_path


class IndexReadError(RuntimeError):
    """The staged index blob for a candidate path could not be read."""


def _read_index_blob(repo_root, path):
    proc = subprocess.run(
        ["git", "-C", repo_root, "show", ":" + path], capture_output=True
    )
    if proc.returncode != 0:
        raise IndexReadError(
            "could not read staged index blob for '%s': %s"
            % (path, proc.stderr.decode("utf-8", "replace").strip())
        )
    return proc.stdout


def _index_is_symlink(repo_root, path):
    proc = subprocess.run(
        ["git", "-C", repo_root, "ls-files", "-s", "-z", "--", path],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return False
    head = proc.stdout.split(b"\t", 1)[0]
    mode = head.split(b" ", 1)[0].decode("ascii", "replace")
    return mode == "120000"


def _inspect_staged(repo_root, path):
    is_link = _index_is_symlink(repo_root, path)
    blob = _read_index_blob(repo_root, path)
    size = len(blob)
    content = None if is_link else blob
    return size, is_link, content


def _inspect_worktree(path):
    if os.path.islink(path):
        return os.lstat(path).st_size, True, None
    if os.path.isdir(path):
        return None, False, None
    with open(path, "rb") as handle:
        data = handle.read()
    return len(data), False, data


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reject private data/checkpoints/credentials before staging."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=None,
        help="explicit working-tree paths to inspect instead of the index",
    )
    parser.add_argument(
        "--max-size-bytes",
        type=int,
        default=DEFAULT_MAX_SIZE_BYTES,
        help="maximum allowed size for a candidate file (default 1 MiB)",
    )
    args = parser.parse_args(argv)

    if args.max_size_bytes < 0:
        parser.error("--max-size-bytes must be non-negative")

    candidates = []
    errors = []

    if args.paths:
        for path in args.paths:
            try:
                size, is_link, content = _inspect_worktree(path)
            except OSError as exc:
                errors.append((path, str(exc)))
                continue
            candidates.append((path, size, is_link, content))
    else:
        repo_root = _repo_root()
        if repo_root is None:
            print("ERROR: not inside a git repository", file=sys.stderr)
            return EXIT_USAGE
        try:
            staged = list(_iter_staged_entries(repo_root))
        except RuntimeError as exc:
            print("ERROR: git failed: %s" % exc, file=sys.stderr)
            return EXIT_USAGE
        for _code, path in staged:
            try:
                size, is_link, content = _inspect_staged(repo_root, path)
            except IndexReadError as exc:
                print("ERROR: %s" % exc, file=sys.stderr)
                return EXIT_USAGE
            candidates.append((path, size, is_link, content))

    violations = []
    for path, size, is_link, content in candidates:
        for code, detail in analyze(path, size, is_link, content, args.max_size_bytes):
            violations.append((code, path, detail))

    for code, path, detail in violations:
        print("VIOLATION %s %s: %s" % (code, path, detail), file=sys.stderr)

    for path, message in errors:
        print("ERROR %s: %s" % (path, message), file=sys.stderr)

    if violations or errors:
        return EXIT_VIOLATION
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
