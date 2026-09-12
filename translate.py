"""Translate Arabic files in the training_data format into English files of the same format, using Claude with the
approved corpus as context (see claude_backend.py).

    python translate.py ar-123.txt                    -> writes entranslation-123.txt next to it
    python translate.py some_folder -o out_folder     -> every ar-*.txt in the folder
    python translate.py ar-1.txt ar-2.txt --show      -> also print AR / EN side by side

Transport (how Claude is reached)
    --transport auto|api|claude-code   api = Anthropic SDK with ANTHROPIC_API_KEY (Console, pay-as-you-go);
                                       claude-code = headless local Claude Code on your claude.ai login;
                                       auto (default) = api if ANTHROPIC_API_KEY is set, else claude-code
    --model claude-fable-5-1  --effort low|medium|high|xhigh|max (default xhigh)
    --no-review   skip the second pass in which Claude checks its draft against the Arabic (halves the calls)
    --claude-exe  path to claude.exe if it is not auto-detected

Corpus and format
    --corpus training_data   folder of approved ar-N.txt / entranslation-N.txt pairs (or a jsonl from prepare_data.py)
    --no-tm       do not reuse approved human translations for lines already in the corpus
    --fuzzy 0.95  also reuse human translations for near-identical lines (0 = exact only)
    --no-copy-div do not copy the ÷ marker from the Arabic line into the English line
    --batch-lines 60  how many lines go into one call
    --overwrite   replace an existing output file
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from collections import Counter

from claude_backend import ClaudeEngine, load_corpus_pairs, make_transport
from common import TranslationMemory, output_path_for, read_lines, write_lines


def collect_inputs(paths: list[str]) -> list[str]:
    files: list[str] = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "ar-*.txt"))))
        elif os.path.isfile(p):
            files.append(p)
        else:
            files.extend(sorted(glob.glob(p)))
    seen, out = set(), []
    for f in files:
        key = os.path.abspath(f)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def build_engine(args) -> ClaudeEngine:
    pairs = load_corpus_pairs(args.corpus)
    tm = None if args.no_tm else TranslationMemory.build(pairs)
    transport = make_transport(args.transport, args.model, args.effort, claude_exe=args.claude_exe)
    if not args.quiet:
        exe = f" exe={transport.exe}" if hasattr(transport, "exe") else ""
        print(f"claude: {transport.name} model={args.model} effort={args.effort} review={not args.no_review} "
              f"corpus={len(pairs)} lines{exe}", flush=True)
    return ClaudeEngine(pairs, transport, tm=tm, copy_div=not args.no_copy_div, fuzzy=args.fuzzy,
                        batch_lines=args.batch_lines, review=not args.no_review, verbose=not args.quiet)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="ar-*.txt files, folders, or globs")
    ap.add_argument("-o", "--out-dir", default=None)
    ap.add_argument("--transport", choices=["auto", "api", "claude-code"], default="auto")
    ap.add_argument("--model", default="claude-fable-5-1")
    ap.add_argument("--effort", default="xhigh", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--claude-exe", default=None)
    ap.add_argument("--corpus", default="training_data")
    ap.add_argument("--batch-lines", type=int, default=60)
    ap.add_argument("--no-tm", action="store_true")
    ap.add_argument("--fuzzy", type=float, default=0.0)
    ap.add_argument("--no-copy-div", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    files = collect_inputs(args.inputs)
    if not files:
        print("no input files found", file=sys.stderr)
        return 1
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    engine = build_engine(args)
    total = Counter()
    failed_files = 0
    for path in files:
        out_path = output_path_for(path, args.out_dir)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"skip {path}: {out_path} exists (use --overwrite)")
            continue
        tf = read_lines(path)
        t0 = time.time()
        results = engine.translate_lines(tf.lines)
        en_lines = [r.en for r in results]
        assert len(en_lines) == len(tf.lines)
        write_lines(out_path, en_lines, newline=tf.newline, trailing_newline=tf.trailing_newline)
        counts = Counter(r.source for r in results)
        total.update(counts)
        n_fail = counts.get("failed", 0)
        failed_files += bool(n_fail)
        print(f"{os.path.basename(path)} -> {out_path}  lines={len(tf.lines)} "
              f"tm={counts['tm'] + counts['tm-fuzzy']} claude={counts['claude'] + counts['claude-split']}"
              f"{f' FAILED={n_fail}' if n_fail else ''} ({time.time() - t0:.1f}s)")
        if n_fail:
            print("   untranslated (left blank) lines:", [i + 1 for i, r in enumerate(results) if r.source == "failed"])
        if args.show:
            for raw, r in zip(tf.lines, results):
                print(f"  AR: {raw}\n  EN: {r.en}   [{r.source}]")
    tr = engine.transport
    extra = f"  cost=${tr.cost_usd:.3f}" if getattr(tr, "cost_usd", None) else ""
    print(f"done: {dict(total)}  usage={dict(tr.usage)}{extra}")
    return 1 if failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
