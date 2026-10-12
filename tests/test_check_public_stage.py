"""Synthetic unit tests for scripts/check_public_stage.py (UFM_RGBD_STEP03).

All fixtures are synthetic. Host-local path / secret literals are assembled at
runtime from fragments so that this test file itself stays free of the raw
patterns the guard rejects (the guard is expected to pass on its own sources).
"""

import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "check_public_stage.py")
TEST_FILE = os.path.abspath(__file__)

SECRET_FIXTURE = "sk" + "-" + ("A" * 24)
GITHUB_TOKEN_FIXTURE = "ghp" + "_" + ("B" * 24)
HOST_PATH_FIXTURE = "/" + "home/" + "nobody/private/secret.txt"
WINDOWS_PATH_FIXTURE = "C:" + "\\" + "Users" + "\\" + "victim" + "\\" + "secret.txt"
ENV_NAME = "OPENROUTER" + "_API_KEY"


def run_guard(args, cwd):
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


class StagedGuardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = self._tmp.name
        self._git("init", "-q")
        self._git("config", "user.email", "tester@example.invalid")
        self._git("config", "user.name", "Tester")

    def tearDown(self):
        self._tmp.cleanup()

    def _git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True)

    def _write(self, relpath, data, binary=False):
        full = os.path.join(self.repo, relpath)
        parent = os.path.dirname(full)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if binary:
            with open(full, "wb") as fh:
                fh.write(data)
        else:
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(data)
        return full

    def _stage(self, relpath):
        r = self._git("add", "-f", relpath)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_media_extension_rejected(self):
        self._write("shot.png", b"\x89PNG\r\n\x1a\n", binary=True)
        self._stage("shot.png")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("DENY_EXT", r.stderr)

    def test_checkpoint_extension_rejected(self):
        self._write("model.pt", b"\x00\x01", binary=True)
        self._stage("model.pt")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("DENY_EXT", r.stderr)

    def test_private_top_level_dir_rejected(self):
        self._write("data/sample.txt", "hello\n")
        self._stage("data/sample.txt")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("PRIVATE_DIR", r.stderr)

    def test_nested_data_dir_is_not_root_anchored_private(self):
        self._write("foo/data/x.py", "x = 1\n")
        self._stage("foo/data/x.py")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_top_level_data_py_rejected_by_root_anchor(self):
        self._write("data/x.py", "x = 1\n")
        self._stage("data/x.py")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("PRIVATE_DIR", r.stderr)

    def test_unreadable_index_blob_fails_closed(self):
        self._write("real.txt", "ok\n")
        self._stage("real.txt")
        ghost = "0" * 39 + "1"
        r = self._git(
            "update-index", "--add", "--cacheinfo", "100644,%s,ghost.txt" % ghost
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        result = run_guard([], self.repo)
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_checkpoints_dir_rejected(self):
        self._write("checkpoints/latest.bin", b"\x00\x01\x02", binary=True)
        self._stage("checkpoints/latest.bin")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("PRIVATE_DIR", r.stderr)

    def test_symlink_rejected(self):
        self._write("target.txt", "plain\n")
        os.symlink("target.txt", os.path.join(self.repo, "link.txt"))
        self._stage("link.txt")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("SYMLINK", r.stderr)

    def test_oversize_rejected(self):
        self._write("big.txt", "A" * 64 + "\n")
        self._stage("big.txt")
        r = run_guard(["--max-size-bytes", "16"], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("OVERSIZE", r.stderr)

    def test_host_local_path_rejected(self):
        self._write("cfg.txt", "root=" + HOST_PATH_FIXTURE + "\n")
        self._stage("cfg.txt")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("HOST_PATH", r.stderr)

    def test_windows_host_path_rejected(self):
        self._write("wincfg.txt", "path=" + WINDOWS_PATH_FIXTURE + "\n")
        self._stage("wincfg.txt")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("HOST_PATH", r.stderr)

    def test_secret_literal_rejected(self):
        self._write("creds.txt", "key=" + SECRET_FIXTURE + "\n")
        self._stage("creds.txt")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("SECRET", r.stderr)

    def test_github_token_rejected(self):
        self._write("token.txt", "token=" + GITHUB_TOKEN_FIXTURE + "\n")
        self._stage("token.txt")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("SECRET", r.stderr)

    def test_env_var_name_only_allowed(self):
        self._write("README.md", "Set " + ENV_NAME + " in your environment before running.\n")
        self._stage("README.md")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_env_var_with_literal_rejected(self):
        self._write("envcfg.txt", ENV_NAME + "=" + ("C" * 32) + "\n")
        self._stage("envcfg.txt")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("SECRET", r.stderr)

    def test_clean_source_file_allowed(self):
        self._write("src/ok.py", "def add(a, b):\n    return a + b\n")
        self._stage("src/ok.py")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_extensionless_common_file_allowed(self):
        self._write("LICENSE", "BSD 3-Clause\n")
        self._stage("LICENSE")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_unknown_binary_extension_rejected(self):
        self._write("blob.xyz", b"\x00\x01\x02", binary=True)
        self._stage("blob.xyz")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("EXT_NOT_ALLOWED", r.stderr)

    def test_index_blob_inspected_not_worktree(self):
        full = self._write("state.txt", "key=" + SECRET_FIXTURE + "\n")
        self._stage("state.txt")
        with open(full, "w", encoding="utf-8") as fh:
            fh.write("clean working tree\n")
        r = run_guard([], self.repo)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("SECRET", r.stderr)


class PathsModeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name, data, binary=False):
        full = os.path.join(self.dir, name)
        if binary:
            with open(full, "wb") as fh:
                fh.write(data)
        else:
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(data)
        return full

    def test_paths_mode_rejects_secret(self):
        full = self._write("creds.txt", "key=" + SECRET_FIXTURE + "\n")
        r = run_guard(["--paths", full], self.dir)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("SECRET", r.stderr)

    def test_paths_mode_rejects_media(self):
        full = self._write("pic.jpg", b"\xff\xd8\xff", binary=True)
        r = run_guard(["--paths", full], self.dir)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("DENY_EXT", r.stderr)

    def test_paths_mode_allows_clean(self):
        full = self._write("ok.md", "# docs\n")
        r = run_guard(["--paths", full], self.dir)
        self.assertEqual(r.returncode, 0, r.stderr)


class SelfCheckTests(unittest.TestCase):
    def test_guard_accepts_own_sources(self):
        r = run_guard(["--paths", SCRIPT, TEST_FILE], REPO_ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_guard_accepts_gitignore_self_check(self):
        r = run_guard(["--paths", ".gitignore"], REPO_ROOT)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
