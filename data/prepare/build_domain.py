#!/usr/bin/env python3
"""Turn filtered .rs files into training pairs.

Reads  data/work/filtered_manifest.jsonl
Writes data/work/domain_pairs.jsonl
       {id, pair_type, prompt, response, crate, file, license}

Pair types per documented public function:
  sig_doc_impl     (doc + signature + todo!() stub)  -> full implementation
  instruction_code (NL instruction from the doc)     -> full implementation

Extraction uses a character-level scanner that masks out comments, strings
(incl. raw/byte strings) and char literals so brace matching never trips on
a `{` inside a string or comment. Nested block comments (legal in Rust) and
lifetime-vs-char-literal disambiguation are handled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

MIN_DOC_CHARS = 30
MIN_BODY_LINES = 3
MAX_PAIR_CHARS = 12000

_FN_RE = re.compile(
    r"^[ \t]*(?:pub(?:\([^)]*\))?\s+)?(?:default\s+)?(?:const\s+)?"
    r"(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]*\"\s+)?fn\s+(\w+)",
    re.M,
)
_CHAR_LIT_RE = re.compile(r"'(?:[^'\\\n]|\\.(?:[0-9a-fA-F]{1,6})?)'")


def code_mask(text: str) -> list[bool]:
    """mask[i] is True iff text[i] is real code (not comment/string/char)."""
    n = len(text)
    mask = [True] * n
    i = 0
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":                      # line comment (also ///, //!)
            j = text.find("\n", i)
            j = n if j == -1 else j
            for k in range(i, j):
                mask[k] = False
            i = j
        elif c == "/" and nxt == "*":                    # block comment, nestable
            depth, j = 1, i + 2
            while j < n and depth:
                if text[j:j + 2] == "/*":
                    depth, j = depth + 1, j + 2
                elif text[j:j + 2] == "*/":
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            for k in range(i, j):
                mask[k] = False
            i = j
        elif c in "br" or c == '"' or c == "'":
            # raw/byte string prefixes: r", r#", b", br#"...
            m = re.match(r'(?:b?r)(#*)"', text[i:])
            if m:
                hashes = m.group(1)
                close = '"' + hashes
                j = text.find(close, i + len(m.group(0)))
                j = n if j == -1 else j + len(close)
                for k in range(i, j):
                    mask[k] = False
                i = j
            elif c == "b" and nxt == '"' or c == '"':    # plain/byte string
                j = i + (2 if c == "b" else 1)
                while j < n:
                    if text[j] == "\\":
                        j += 2
                    elif text[j] == '"':
                        j += 1
                        break
                    else:
                        j += 1
                for k in range(i, j):
                    mask[k] = False
                i = j
            elif c == "'":
                m2 = _CHAR_LIT_RE.match(text, i)
                if m2:                                   # char literal
                    for k in range(i, m2.end()):
                        mask[k] = False
                    i = m2.end()
                else:                                    # lifetime: skip quote only
                    i += 1
            else:                                        # bare b/r identifier char
                i += 1
        else:
            i += 1
    return mask


def find_fn_body(text: str, mask: list[bool], sig_start: int) -> tuple[int, int] | None:
    """Return (open_brace_idx, end_idx_exclusive) of the fn body, or None."""
    i, n = sig_start, len(text)
    while i < n:
        if mask[i]:
            if text[i] == "{":
                break
            if text[i] == ";":                           # trait method decl, no body
                return None
        i += 1
    else:
        return None
    depth = 0
    for j in range(i, n):
        if not mask[j]:
            continue
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return i, j + 1
    return None


def doc_block_above(lines: list[str], fn_line: int) -> tuple[int, list[str], list[str]]:
    """Walk upward over #[attrs] then /// docs. Returns (start_line, docs, attrs)."""
    attrs, docs = [], []
    i = fn_line - 1
    while i >= 0 and re.match(r"^\s*#\[.*\]\s*$", lines[i]):
        attrs.insert(0, lines[i])
        i -= 1
    while i >= 0 and re.match(r"^\s*///(?!/)", lines[i]):
        docs.insert(0, lines[i])
        i -= 1
    return i + 1, docs, attrs


def doc_summary(docs: list[str]) -> str:
    text = " ".join(re.sub(r"^\s*///\s?", "", l) for l in docs).strip()
    text = re.sub(r"`([^`]*)`", r"\1", text)             # unwrap inline code
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    return (m.group(1) if m else text).strip()


def extract_pairs(text: str, meta: dict) -> list[dict]:
    mask = code_mask(text)
    lines = text.splitlines()
    line_start = [0]
    for l in lines:
        line_start.append(line_start[-1] + len(l) + 1)

    pairs = []
    for m in _FN_RE.finditer(text):
        if not mask[m.start(1)]:
            continue  # fn keyword inside comment/string
        name = m.group(1)
        if name == "main":
            continue
        fn_line = text.count("\n", 0, m.start())
        _, docs, attrs = doc_block_above(lines, fn_line)
        if sum(len(d) for d in docs) < MIN_DOC_CHARS:
            continue
        if any("#[test]" in a or "#[cfg(test)]" in a for a in attrs):
            continue

        body = find_fn_body(text, mask, m.end())
        if body is None:
            continue
        open_idx, end_idx = body
        body_text = text[open_idx:end_idx]
        if body_text.count("\n") + 1 < MIN_BODY_LINES + 1:
            continue
        if "todo!" in body_text or "unimplemented!" in body_text:
            continue

        indent = re.match(r"[ \t]*", lines[fn_line]).group(0)
        signature = text[m.start():open_idx].rstrip()
        doc_text = "\n".join(docs)
        attr_text = "\n".join(attrs)
        header = "\n".join(x for x in (doc_text, attr_text) if x)
        full_fn = (header + "\n" if header else "") + text[m.start():end_idx]
        if len(full_fn) > MAX_PAIR_CHARS:
            continue

        base = {**meta, "fn": name}
        stub = (f"{header}\n{signature} {{\n{indent}    todo!()\n{indent}}}"
                if header else f"{signature} {{\n{indent}    todo!()\n{indent}}}")
        pairs.append({
            **base, "pair_type": "sig_doc_impl",
            "prompt": ("Implement the following Rust function. Replace the "
                       "`todo!()` with a complete, idiomatic implementation.\n\n"
                       f"```rust\n{stub}\n```"),
            "response": f"```rust\n{full_fn}\n```",
        })
        summary = doc_summary(docs)
        if len(summary) >= 20:
            s = summary[0].lower() + summary[1:]
            pairs.append({
                **base, "pair_type": "instruction_code",
                "prompt": (f"Write a Rust function `{name}` that "
                           f"{s.rstrip('.')}." ),
                "response": f"```rust\n{full_fn}\n```",
            })
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workdir", type=Path, default=Path("data/work"))
    args = ap.parse_args()

    seen: set[str] = set()
    n_in, n_out = 0, 0
    by_type: dict[str, int] = {}
    with open(args.workdir / "domain_pairs.jsonl", "w") as out:
        for line in (args.workdir / "filtered_manifest.jsonl").open():
            rec = json.loads(line)
            n_in += 1
            text = Path(rec["path"]).read_text(encoding="utf-8")
            meta = {"crate": rec["crate"], "file": rec["path"],
                    "license": rec["license"]}
            for p in extract_pairs(text, meta):
                h = hashlib.sha256(
                    (p["pair_type"] + p["response"]).encode()).hexdigest()
                if h in seen:        # same fn vendored in examples/tests
                    continue
                seen.add(h)
                p["id"] = f"domain-{h[:16]}"
                out.write(json.dumps(p) + "\n")
                n_out += 1
                by_type[p["pair_type"]] = by_type.get(p["pair_type"], 0) + 1
    print(json.dumps({"files_in": n_in, "pairs_out": n_out,
                      "by_type": by_type}, indent=2))


if __name__ == "__main__":
    main()
