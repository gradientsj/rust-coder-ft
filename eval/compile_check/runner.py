#!/usr/bin/env python3
"""Sandboxed cargo runner: compile / clippy / test verdicts for Rust snippets.

Verdict pipeline per candidate (each in its own temp cargo project, no deps,
`--offline` so nothing can be fetched):
  build   cargo build --message-format=json   -> compiles, error/warning counts
  clippy  cargo clippy --message-format=json  -> clippy_clean (no warnings)
  test    cargo test (only if test code given)-> tests_passed

Sandboxing layers, in order of strength actually available here:
  - cargo --offline + no dependencies        (no network at build time)
  - `unshare -rn` network namespace for `cargo test` when the kernel allows it
    (test code EXECUTES; this cuts its network), else plain timeout fallback
  - hard wall-clock timeout + own process group kill on every subprocess
The active sandbox mode is recorded in each result; full container isolation
is deliberately out of scope for self-generated local code.

Also used as the GRPO reward backend in Phase 6 — keep the API stable.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path

TEMPLATE = Path(__file__).parent / "template"
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")

_HEADER = "#![allow(dead_code, unused_variables, unused_imports, unused_mut)]\n"


@dataclass
class CheckResult:
    compiles: bool = False
    n_errors: int = 0
    n_warnings: int = 0
    clippy_clean: bool | None = None     # None = not run (build failed)
    tests_passed: bool | None = None     # None = no tests provided / not run
    error_excerpt: str = ""
    duration_s: float = 0.0
    sandbox: str = "timeout"             # "netns+timeout" when unshare works
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _have_unshare_netns() -> bool:
    try:
        r = subprocess.run(["unshare", "-rn", "true"], capture_output=True,
                           timeout=10)
        return r.returncode == 0
    except Exception:
        return False


_UNSHARE_OK = _have_unshare_netns()


def _run(cmd: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    # own process group so a timeout kills cargo AND its rustc children
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True,
                         start_new_session=True,
                         env={**os.environ, "CARGO_TERM_COLOR": "never"})
    try:
        out, err = p.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, p.returncode, out, err)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return subprocess.CompletedProcess(cmd, -9, out, "TIMEOUT\n" + (err or ""))


def _parse_cargo_json(stdout: str) -> tuple[int, int, str]:
    """(n_errors, n_warnings, first_error_text) from --message-format=json."""
    n_err = n_warn = 0
    first = ""
    for line in stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") != "compiler-message":
            continue
        level = msg["message"].get("level")
        if level == "error":
            n_err += 1
            if not first:
                first = msg["message"].get("rendered") or msg["message"]["message"]
        elif level == "warning":
            n_warn += 1
    return n_err, n_warn, first


def check(code: str, tests: str | None = None, *, clippy: bool = True,
          build_timeout: int = 60, test_timeout: int = 30,
          meta: dict | None = None) -> CheckResult:
    """Full verdict for one Rust snippet (a lib-level item or items)."""
    res = CheckResult(meta=meta or {})
    lib = _HEADER + code
    if tests:
        if tests.lstrip().startswith("#[cfg(test)"):
            lib += "\n\n" + tests + "\n"   # already a complete test module
        else:
            lib += (f"\n\n#[cfg(test)]\nmod harness_tests {{\n"
                    f"    use super::*;\n{tests}\n}}\n")

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rcft-") as td:
        proj = Path(td) / "candidate"
        shutil.copytree(TEMPLATE, proj)
        (proj / "src/lib.rs").write_text(lib, encoding="utf-8")

        r = _run([CARGO, "build", "--offline", "--message-format=json"],
                 proj, build_timeout)
        n_err, n_warn, first = _parse_cargo_json(r.stdout)
        res.n_errors, res.n_warnings = n_err, n_warn
        res.compiles = r.returncode == 0 and n_err == 0
        res.error_excerpt = first[:2000]

        if res.compiles and clippy:
            rc = _run([CARGO, "clippy", "--offline", "--message-format=json"],
                      proj, build_timeout)
            ce, cw, _ = _parse_cargo_json(rc.stdout)
            res.clippy_clean = rc.returncode == 0 and ce == 0 and cw == 0

        if res.compiles and tests:
            cmd = [CARGO, "test", "--offline", "-q"]
            if _UNSHARE_OK:
                cmd = ["unshare", "-rn"] + cmd
                res.sandbox = "netns+timeout"
            rt = _run(cmd, proj, test_timeout)
            res.tests_passed = rt.returncode == 0

    res.duration_s = round(time.monotonic() - t0, 2)
    return res


def check_many(items: list[dict], *, max_workers: int = 16,
               **kw) -> list[CheckResult]:
    """items: [{code, tests?, meta?}]. Order-preserving parallel map."""
    def one(it: dict) -> CheckResult:
        return check(it["code"], it.get("tests"), meta=it.get("meta"), **kw)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, items))


if __name__ == "__main__":  # self-test on known snippets
    cases = [
        ("good+test", "pub fn add(a: i32, b: i32) -> i32 { a + b }",
         "    #[test]\n    fn t() { assert_eq!(add(2, 2), 4); }"),
        ("clippy-dirty", "pub fn double(xs: &Vec<i32>) -> Vec<i32> {\n"
         "    xs.iter().map(|x| x * 2).collect()\n}", None),  # &Vec -> clippy warns
        ("broken", "pub fn nope() -> i32 { \"string\" }", None),
        ("failing-test", "pub fn sub(a: i32, b: i32) -> i32 { a + b }",
         "    #[test]\n    fn t() { assert_eq!(sub(5, 3), 2); }"),
    ]
    for name, code, tests in cases:
        r = check(code, tests)
        print(f"{name:14s} compiles={r.compiles!s:5s} clippy_clean={r.clippy_clean!s:5s} "
              f"tests_passed={r.tests_passed!s:5s} errs={r.n_errors} "
              f"warns={r.n_warnings} {r.duration_s}s [{r.sandbox}]")
