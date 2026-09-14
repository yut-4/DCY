#!/usr/bin/env python3
"""Assemble a C/C++ corpus of a target token size using a real tokenizer.

Corpus scale claims in DCY are stated in tokens under a fixed tokenizer, not in
bytes or files, so this counts with tiktoken rather than estimating from byte
counts. Files are taken in a deterministic order and the run stops as soon as
the target is reached, so the same inputs always produce the same corpus.
"""

import argparse
import json
import shutil
from pathlib import Path

import tiktoken

SOURCE_SUFFIXES = {".c", ".h", ".cc", ".cpp", ".hpp", ".cxx", ".hh"}
SKIP_PARTS = {".git", "test", "tests", "testdata", "fixtures", "third_party",
              "vendor", "build", "node_modules", "target"}


def eligible(root):
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SOURCE_SUFFIXES or not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, nargs="+", required=True,
                        help="repository roots to draw files from, in priority order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=1_000_000)
    parser.add_argument("--encoding", default="cl100k_base")
    parser.add_argument("--max-file-tokens", type=int, default=40_000,
                        help="skip single files larger than this (amalgamations skew a corpus)")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    encoder = tiktoken.get_encoding(args.encoding)
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    total_tokens, total_bytes, records = 0, 0, []
    for source in args.sources:
        if total_tokens >= args.target_tokens:
            break
        label = source.name
        for path in eligible(source):
            if total_tokens >= args.target_tokens:
                break
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            tokens = len(encoder.encode(text, disallowed_special=()))
            if tokens == 0 or tokens > args.max_file_tokens:
                continue
            relative = path.relative_to(source)
            destination = args.out / label / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            total_tokens += tokens
            total_bytes += len(text.encode())
            records.append({"repo": label, "path": str(Path(label) / relative),
                            "tokens": tokens, "bytes": len(text.encode())})

    summary = {
        "encoding": args.encoding,
        "target_tokens": args.target_tokens,
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "files": len(records),
        "per_repo": {},
    }
    for record in records:
        entry = summary["per_repo"].setdefault(record["repo"], {"files": 0, "tokens": 0})
        entry["files"] += 1
        entry["tokens"] += record["tokens"]
    if args.manifest:
        args.manifest.write_text(json.dumps({"summary": summary, "files": records}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
