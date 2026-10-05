"""Workspace confinement and the built-in tools' safety behaviour."""

from __future__ import annotations

import sys

import pytest

from hermes_agent.errors import SandboxViolationError
from hermes_agent.tools.builtins import safe_eval_math
from hermes_agent.tools.sandbox import (
    relative_to_workspace,
    resolve_in_workspace,
)


class TestPathSandboxing:
    def test_relative_path_resolves_inside(self, workspace):
        assert resolve_in_workspace("notes/a.txt").parent == workspace / "notes"

    def test_workspace_root_itself_is_allowed(self, workspace):
        assert resolve_in_workspace(".") == workspace

    @pytest.mark.parametrize(
        "path",
        [
            "../outside.txt",
            "../../outside.txt",
            "notes/../../outside.txt",
            "./../../etc/passwd",
            "a/b/c/../../../../x",
        ],
    )
    def test_traversal_is_rejected(self, path, workspace):
        with pytest.raises(SandboxViolationError):
            resolve_in_workspace(path)

    def test_absolute_path_outside_is_rejected(self, workspace, tmp_path):
        with pytest.raises(SandboxViolationError):
            resolve_in_workspace(str(tmp_path / "elsewhere.txt"))

    def test_absolute_path_inside_is_allowed(self, workspace):
        assert resolve_in_workspace(str(workspace / "ok.txt")).name == "ok.txt"

    def test_traversal_that_lands_back_inside_is_allowed(self, workspace):
        """`a/../b` is legitimate once resolved; only escaping matters."""
        assert resolve_in_workspace("a/../b.txt") == workspace / "b.txt"

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_symlink_escape_is_rejected(self, workspace, tmp_path):
        secret = tmp_path / "secret.txt"
        secret.write_text("classified")
        (workspace / "link.txt").symlink_to(secret)
        with pytest.raises(SandboxViolationError):
            resolve_in_workspace("link.txt")

    def test_must_exist_raises_for_missing(self, workspace):
        with pytest.raises(FileNotFoundError):
            resolve_in_workspace("nope.txt", must_exist=True)

    def test_nonexistent_path_allowed_for_writes(self, workspace):
        assert resolve_in_workspace("new/deep/file.txt").name == "file.txt"

    def test_relative_display_is_posix(self, workspace):
        assert relative_to_workspace(workspace / "a" / "b.txt") == "a/b.txt"


class TestFileTools:
    async def test_write_then_read(self, registry, workspace):
        written = await registry.execute(
            "write_file", {"path": "sub/a.txt", "content": "hello"}
        )
        assert written.ok
        assert (workspace / "sub" / "a.txt").read_text() == "hello"
        assert (await registry.execute("read_file", {"path": "sub/a.txt"})).content == "hello"

    async def test_append(self, registry, workspace):
        await registry.execute("write_file", {"path": "a.txt", "content": "one"})
        await registry.execute(
            "write_file", {"path": "a.txt", "content": "two", "append": True}
        )
        assert (workspace / "a.txt").read_text() == "onetwo"

    async def test_read_traversal_blocked(self, registry, workspace):
        result = await registry.execute("read_file", {"path": "../../../etc/passwd"})
        assert not result.ok and "outside the workspace" in result.error

    async def test_write_traversal_blocked(self, registry, workspace, tmp_path):
        result = await registry.execute(
            "write_file", {"path": "../escaped.txt", "content": "x"}
        )
        assert not result.ok
        assert not (tmp_path / "escaped.txt").exists()

    async def test_read_truncates_at_max_bytes(self, registry, workspace):
        (workspace / "big.txt").write_text("x" * 5000)
        result = await registry.execute("read_file", {"path": "big.txt", "max_bytes": 100})
        assert result.ok and "truncated" in result.content
        assert len(result.content) < 400

    async def test_read_directory_is_an_error(self, registry, workspace):
        (workspace / "d").mkdir()
        result = await registry.execute("read_file", {"path": "d"})
        assert not result.ok and "use list_dir" in result.error

    async def test_list_dir_recursive(self, registry, workspace):
        (workspace / "x").mkdir()
        (workspace / "x" / "y.txt").write_text("hi")
        result = await registry.execute("list_dir", {"path": ".", "recursive": True})
        assert "x/" in result.content and "x/y.txt" in result.content

    async def test_list_dir_empty(self, registry, workspace):
        assert "is empty" in (await registry.execute("list_dir", {"path": "."})).content

    async def test_destructive_write_blocked_when_disabled(self, registry, workspace):
        registry.allow_destructive = False
        result = await registry.execute("write_file", {"path": "a.txt", "content": "x"})
        assert not result.ok and "--allow-destructive" in result.error
        assert not (workspace / "a.txt").exists()


class TestRunPython:
    async def test_captures_stdout(self, registry, workspace):
        result = await registry.execute("run_python", {"code": "print(6*7)"})
        assert result.ok and "42" in result.content and "exit_code: 0" in result.content

    async def test_traceback_is_reported(self, registry, workspace):
        result = await registry.execute("run_python", {"code": "raise ValueError('nope')"})
        assert "ValueError" in result.content and "exit_code: 1" in result.content

    async def test_timeout_is_enforced(self, registry, workspace):
        result = await registry.execute(
            "run_python", {"code": "while True: pass", "timeout_s": 2}
        )
        assert "TIMEOUT" in result.content

    async def test_network_is_blocked(self, registry, workspace):
        result = await registry.execute(
            "run_python",
            {"code": "import socket; socket.create_connection(('1.1.1.1', 80))"},
        )
        assert "disabled" in result.content.lower() or "OSError" in result.content

    async def test_runs_in_the_workspace(self, registry, workspace):
        (workspace / "data.txt").write_text("from workspace")
        result = await registry.execute(
            "run_python", {"code": "print(open('data.txt').read())"}
        )
        assert "from workspace" in result.content

    async def test_no_output_hint(self, registry, workspace):
        result = await registry.execute("run_python", {"code": "x = 1"})
        assert "remember to print" in result.content


class TestCalculator:
    @pytest.mark.parametrize(
        "expression,expected",
        [
            ("2 + 2", 4),
            ("10 / 4", 2.5),
            ("2 ** 10", 1024),
            ("17 % 5", 2),
            ("-(3 + 4)", -7),
            ("sqrt(16)", 4.0),
            ("max(1, 9, 3)", 9),
            ("round(3.14159, 2)", 3.14),
        ],
    )
    def test_arithmetic(self, expression, expected):
        assert safe_eval_math(expression) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('ls')",
            "open('/etc/passwd').read()",
            "().__class__.__bases__",
            "exec('x=1')",
            "[x for x in range(10)]",
            "lambda: 1",
        ],
    )
    def test_code_execution_is_rejected(self, expression):
        with pytest.raises(ValueError):
            safe_eval_math(expression)

    def test_huge_exponent_is_refused(self):
        """Guards against 2**10**9 hanging the process on a bignum."""
        with pytest.raises(ValueError, match="Exponent too large"):
            safe_eval_math("2 ** 999999999")

    def test_unknown_name_rejected(self):
        with pytest.raises(ValueError, match="Unknown name"):
            safe_eval_math("secret + 1")

    async def test_through_the_registry(self, registry):
        result = await registry.execute("calculator", {"expression": "3 * 7"})
        assert result.ok and "21" in result.content

    async def test_division_by_zero_is_a_tool_error(self, registry):
        result = await registry.execute("calculator", {"expression": "1/0"})
        assert not result.ok and "ZeroDivisionError" in result.error


class TestWebSearchStub:
    async def test_unconfigured_backend_explains_itself(self, registry):
        result = await registry.execute("web_search", {"query": "anything"})
        assert result.ok
        assert "not configured" in result.content

    async def test_pluggable_backend(self, registry):
        from hermes_agent.tools import builtins

        builtins.set_search_backend(
            lambda q, n: [{"title": f"Result for {q}", "url": "http://x", "snippet": "s"}]
        )
        try:
            result = await registry.execute("web_search", {"query": "hermes"})
            assert "Result for hermes" in result.content
        finally:
            builtins.set_search_backend(None)
