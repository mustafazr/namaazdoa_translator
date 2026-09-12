"""Optional: split training_data into training / held-out files so evaluate.py can score the engine honestly.

Usage:  python prepare_data.py [--data-dir training_data] [--out data] [--test-files 20] [--seed 42]

Writes data/all.jsonl, data/train.jsonl, data/test.jsonl and data/split.json. translate.py does not need this; it reads
training_data directly.
"""
import argparse
import json
import os
import random

from common import load_corpus, dump_jsonl, pair_to_dict, tm_key, STANZA_BREAK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="training_data")
    ap.add_argument("--out", default="data")
    ap.add_argument("--test-files", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pairs = load_corpus(args.data_dir)
    file_ids = sorted({p.file_id for p in pairs}, key=lambda s: (len(s), s))
    rng = random.Random(args.seed)
    eligible = [f for f in file_ids if sum(1 for p in pairs if p.file_id == f) >= 2]
    test_ids = set(rng.sample(eligible, args.test_files))
    train_ids = [f for f in file_ids if f not in test_ids]
    train = [p for p in pairs if p.file_id not in test_ids]
    test = [p for p in pairs if p.file_id in test_ids]

    dump_jsonl(os.path.join(args.out, "all.jsonl"), map(pair_to_dict, pairs))
    dump_jsonl(os.path.join(args.out, "train.jsonl"), map(pair_to_dict, train))
    dump_jsonl(os.path.join(args.out, "test.jsonl"), map(pair_to_dict, test))
    with open(os.path.join(args.out, "split.json"), "w", encoding="utf-8") as fh:
        json.dump({"train_files": train_ids, "test_files": sorted(test_ids, key=lambda s: (len(s), s))}, fh, indent=1)

    print(f"files: {len(file_ids)}  pairs: {len(pairs)}  unique sources: {len({tm_key(p.ar_raw) for p in pairs})}")
    print(f"train files: {len(train_ids)} ({len(train)} lines)   test files: {len(test_ids)} ({len(test)} lines)")
    train_keys = {tm_key(p.ar_raw) for p in train}
    covered = sum(1 for p in test if tm_key(p.ar_raw) in train_keys)
    print(f"test lines with an exact match in train: {covered}/{len(test)}")
    with_eq = [p for p in pairs if STANZA_BREAK in p.ar_src]
    same = sum(1 for p in with_eq if p.ar_src.count(STANZA_BREAK) == p.en.count(STANZA_BREAK))
    print(f"lines with '=': {len(with_eq)}; English keeps the same '=' count in {same}")


if __name__ == "__main__":
    main()
