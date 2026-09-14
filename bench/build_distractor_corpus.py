#!/usr/bin/env python3
"""Build larger indexes by preserving a fixed base and adding distractors.

The base tree-sitter files are copied byte-for-byte at every scale. Only files
from unrelated repositories are added, so gold files, goals and gold source
spans remain constant while VT grows.
"""

import argparse
import json
import shutil
from pathlib import Path
import tiktoken

SUFFIXES = {".c", ".h", ".cc", ".cpp", ".hpp", ".cxx", ".hh"}
SKIP = {".git", "test", "tests", "testdata", "fixtures", "third_party", "vendor", "build"}


def files(root):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUFFIXES and not any(x in SKIP for x in p.parts):
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--distractors", type=Path, nargs="+", required=True)
    ap.add_argument("--targets", type=int, nargs="+", required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--encoding", default="cl100k_base")
    args = ap.parse_args()
    enc = tiktoken.get_encoding(args.encoding)

    base_records = []
    base_tokens = 0
    for p in files(args.base):
        rel = p.relative_to(args.base)
        dest_rel = Path(args.base.name) / rel
        dest = args.out_root / f"base-{args.targets[0]}" / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        text = p.read_text(encoding="utf-8", errors="ignore")
        n = len(enc.encode(text, disallowed_special=()))
        base_tokens += n
        base_records.append({"path": str(dest_rel), "tokens": n, "bytes": len(text.encode())})

    if base_tokens > min(args.targets):
        raise SystemExit(f"base has {base_tokens} tokens, above smallest target; use --targets >= {base_tokens}")

    for target in args.targets:
        out = args.out_root / f"distractor-{target}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        # Re-copy base into the target-specific directory.
        for p in files(args.base):
            dest = out / args.base.name / p.relative_to(args.base)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        total = base_tokens
        selected = list(base_records)
        for source in args.distractors:
            if total >= target:
                break
            for p in files(source):
                if total >= target:
                    break
                text = p.read_text(encoding="utf-8", errors="ignore")
                n = len(enc.encode(text, disallowed_special=()))
                if not n:
                    continue
                rel = Path(source.name) / p.relative_to(source)
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
                total += n
                selected.append({"path": str(rel), "tokens": n, "bytes": len(text.encode())})
        (out / "manifest.json").write_text(json.dumps({
            "encoding": args.encoding, "target_tokens": target,
            "base_tokens": base_tokens, "total_tokens": total,
            "files": len(selected), "base_files": len(base_records),
        }, indent=2))
        print(json.dumps({"target": target, "actual_tokens": total,
                          "base_tokens": base_tokens, "files": len(selected)}))


if __name__ == "__main__":
    main()
