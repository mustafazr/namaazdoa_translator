"""Score the Claude engine on held-out files against the approved human translations.

    python prepare_data.py                       # once: file-level split into data/train.jsonl + data/test.jsonl
    python evaluate.py                           # examples and TM restricted to the training files -> honest score
    python evaluate.py --limit-files 3           # quick check

Reports corpus BLEU and chrF++ (sacrebleu) and '=' structure accuracy, for Claude alone and for translation-memory +
Claude. Writes reports/<name>_side_by_side.txt and reports/<name>_scores.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import sacrebleu

from claude_backend import ClaudeEngine, load_corpus_pairs, make_transport
from common import Pair, TranslationMemory, load_jsonl, STANZA_BREAK


def score(hyps, refs, srcs) -> dict:
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    with_eq = [(h, s) for h, s in zip(hyps, srcs) if STANZA_BREAK in s]
    eq_ok = sum(1 for h, s in with_eq if h.count(STANZA_BREAK) == s.count(STANZA_BREAK))
    return {"bleu": round(bleu.score, 2), "chrf++": round(chrf.score, 2),
            "stanza_break_kept": f"{eq_ok}/{len(with_eq)}", "lines": len(hyps)}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["auto", "api", "claude-code"], default="auto")
    ap.add_argument("--model", default="claude-fable-5-1")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--no-review", action="store_true")
    ap.add_argument("--claude-exe", default=None)
    ap.add_argument("--corpus", default="data/train.jsonl", help="examples come from here; keep it to train.jsonl for an honest score")
    ap.add_argument("--test-file", default="data/test.jsonl")
    ap.add_argument("--tm-from", default="data/train.jsonl", help="build the TM from this file ('none' to skip)")
    ap.add_argument("--fuzzy", type=float, default=0.0)
    ap.add_argument("--limit-files", type=int, default=0)
    ap.add_argument("--out", default="reports")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    rows = load_jsonl(args.test_file)
    by_file: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_file[r["file_id"]].append(r)
    if args.limit_files:
        by_file = dict(list(by_file.items())[: args.limit_files])

    pairs = load_corpus_pairs(args.corpus)
    transport = make_transport(args.transport, args.model, args.effort, claude_exe=args.claude_exe)
    engine = ClaudeEngine(pairs, transport, tm=None, copy_div=False, fuzzy=args.fuzzy, review=not args.no_review)
    name = args.name or f"{args.model}-{args.effort}{'' if not args.no_review else '-noreview'}"
    print(f"claude: {transport.name} model={args.model} effort={args.effort} examples from {len(pairs)} lines; "
          f"{len(by_file)} test files, {sum(len(v) for v in by_file.values())} lines")

    hyps, refs, srcs, tags = [], [], [], Counter()
    t0 = time.time()
    for fid, frows in by_file.items():
        results = engine.translate_lines([r["ar_raw"] for r in frows])
        for r, res in zip(frows, results):
            hyps.append(res.en)
            refs.append(r["en"])
            srcs.append(r["ar_src"])
            tags[res.source] += 1
        print(f"  file {fid}: {len(frows)} lines done ({time.time() - t0:.0f}s so far)", flush=True)

    report = {"model": args.model, "effort": args.effort, "review": not args.no_review, "transport": transport.name,
              "corpus": args.corpus, "usage": dict(transport.usage)}
    if getattr(transport, "cost_usd", None):
        report["cost_usd"] = round(transport.cost_usd, 4)
    report["claude_only"] = score(hyps, refs, srcs) | {"tags": dict(tags), "seconds": round(time.time() - t0)}
    print("Claude only    :", report["claude_only"])

    hyps_tm = hyps
    if args.tm_from and args.tm_from.lower() != "none":
        tm = TranslationMemory.build(Pair(**r) for r in load_jsonl(args.tm_from))
        hyps_tm, tm_tags, i = [], Counter(), 0
        for frows in by_file.values():
            for r in frows:
                hit, sc = tm.lookup(r["ar_raw"], args.fuzzy)
                hyps_tm.append(hit if hit is not None else hyps[i])
                tm_tags["tm" if hit is not None else "claude"] += 1
                i += 1
        report["tm_plus_claude"] = score(hyps_tm, refs, srcs) | {"tags": dict(tm_tags)}
        print("TM + Claude    :", report["tm_plus_claude"])

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}_side_by_side.txt"), "w", encoding="utf-8") as fh:
        for s, r, h in zip(srcs, refs, hyps_tm):
            fh.write(f"AR : {s}\nREF: {r}\nHYP: {h}\n\n")
    json.dump(report, open(os.path.join(args.out, f"{name}_scores.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"wrote {args.out}/{name}_side_by_side.txt and {name}_scores.json")


if __name__ == "__main__":
    main()
