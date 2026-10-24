"""Tests for the Phase 0 command-line foundation."""

from __future__ import annotations

import ast
import subprocess
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPOSITORY_ROOT / "main.py"
ARTIFACTS_DIRECTORY = REPOSITORY_ROOT / "artifacts"


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a separate process and capture its output."""
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *arguments],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


class CliTests(unittest.TestCase):
    def test_help_command_displays_usage(self) -> None:
        result = run_cli("--help")
        normalized_output = " ".join(result.stdout.lower().split())

        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", normalized_output)
        self.assertIn("infrastructure only", normalized_output)

    def test_help_exits_successfully(self) -> None:
        result = run_cli("--help")

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")

    def test_pending_command_does_not_train_or_create_artifacts(self) -> None:
        artifacts_before = set(ARTIFACTS_DIRECTORY.iterdir())
        result = run_cli("train")
        artifacts_after = set(ARTIFACTS_DIRECTORY.iterdir())

        self.assertEqual(result.returncode, 2)
        self.assertIn("not implemented yet", result.stdout)
        self.assertEqual(artifacts_after, artifacts_before)

        syntax_tree = ast.parse(CLI_PATH.read_text(encoding="utf-8"))
        imported_roots = {
            alias.name.split(".", maxsplit=1)[0]
            for node in ast.walk(syntax_tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertTrue(
            {"sklearn", "numpy", "pandas", "joblib"}.isdisjoint(imported_roots)
        )


if __name__ == "__main__":
    unittest.main()

