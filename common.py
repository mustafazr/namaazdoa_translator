"""Shared helpers for the namaazdoa Arabic -> English translator.

Corpus conventions (training_data/ar-N.txt <-> training_data/entranslation-N.txt):
  * files are line-aligned: Arabic line k  <->  English line k
  * '='  breaks between lines within a qasida stanza; PRESERVED in the English (same count/positions)
  * '÷'  a line-break marker used by the app; copied to the English at the same position (start/end of line)
  * '*'  separates phrases inside an Arabic line; normally dropped in the English
  * instruction lines are written in Lisan al-Dawat (Gujarati in Arabic script)
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Iterable

# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #
# Tashkeel, Quranic annotation marks, superscript alef, small high marks.
_DIACRITICS = re.compile(
    r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]"
)
_TATWEEL = "ـ"
_BIDI = re.compile(r"[‎‏‪-‮⁦-⁩﻿]")
_WS = re.compile(r"[ \t ]+")
DIV_MARK = "÷"  # ÷
STANZA_BREAK = "="

# Alef variants -> bare alef, used ONLY for translation-memory keys.
_ALEF_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_KEY_PUNCT = re.compile(
    "[*،,.:;!?()\\[\\]{}\"'‘’“”«»﴾﴿\\-–—]"
)


def strip_diacritics(text: str) -> str:
    return _DIACRITICS.sub("", text).replace(_TATWEEL, "")


def split_div_marks(text: str) -> tuple[str, str, str]:
    """Return (leading ÷ marks, core text, trailing ÷ marks).

    A ÷ inside the line (very rare in the corpus) is removed from the core and
    counted as trailing, so that it is re-attached at the end of the English line.
    """
    text = _BIDI.sub("", text)
    stripped = text.strip()
    lead = ""
    while stripped.startswith(DIV_MARK):
        lead += DIV_MARK
        stripped = stripped[1:].lstrip()
    trail = ""
    while stripped.endswith(DIV_MARK):
        trail = DIV_MARK + trail
        stripped = stripped[:-1].rstrip()
    inner = stripped.count(DIV_MARK)
    if inner:
        stripped = stripped.replace(DIV_MARK, " ")
        trail += DIV_MARK * inner
    return lead, stripped, trail


def clean_source(text: str, keep_diacritics: bool = False) -> str:
    """Arabic line as it is fed to the neural model. ÷ marks are removed here."""
    _, text, _ = split_div_marks(text)
    if not keep_diacritics:
        text = strip_diacritics(text)
    text = _WS.sub(" ", text).strip()
    # tidy spacing around structural separators
    text = re.sub(r"\s*\*\s*", " * ", text)
    text = re.sub(r"\s*=\s*", "=", text)
    text = text.strip(" *")
    return text


def tm_key(text: str) -> str:
    """Aggressive normalisation for exact translation-memory matching."""
    text = clean_source(text)
    text = text.translate(_ALEF_MAP)
    text = _KEY_PUNCT.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def clean_target(text: str) -> str:
    text = _BIDI.sub("", text).replace("\t", " ")
    _, text, _ = split_div_marks(text)  # the approved English never carries ÷; be safe anyway
    return _WS.sub(" ", text).strip()


def reattach_div_marks(ar_raw: str, en: str, copy_div: bool = True) -> str:
    if not copy_div or not en:
        return en
    lead, _, trail = split_div_marks(ar_raw)
    return f"{lead}{en}{trail}"


def hemistichs(text: str) -> list[str]:
    return [h.strip() for h in text.split(STANZA_BREAK)]


# --------------------------------------------------------------------------- #
# File IO that preserves the corpus conventions
# --------------------------------------------------------------------------- #
@dataclass
class TextFile:
    path: str
    lines: list[str]
    newline: str  # "\r\n" or "\n"
    trailing_newline: bool


def read_lines(path: str) -> TextFile:
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8-sig")
    newline = "\r\n" if b"\r\n" in raw else "\n"
    trailing = text.endswith(("\n", "\r"))
    return TextFile(path=path, lines=text.splitlines(), newline=newline, trailing_newline=trailing)


def write_lines(path: str, lines: Iterable[str], newline: str = "\r\n", trailing_newline: bool = True) -> None:
    body = newline.join(lines)
    if trailing_newline and body:
        body += newline
    with open(path, "wb") as fh:
        fh.write(body.encode("utf-8"))


def pair_id_from_path(path: str) -> str | None:
    """'.../ar-123.txt' -> '123'."""
    m = re.match(r"^ar-(.+)\.txt$", os.path.basename(path), flags=re.IGNORECASE)
    return m.group(1) if m else None


def output_path_for(ar_path: str, out_dir: str | None = None) -> str:
    pid = pair_id_from_path(ar_path)
    base = f"entranslation-{pid}.txt" if pid else os.path.splitext(os.path.basename(ar_path))[0] + ".en.txt"
    return os.path.join(out_dir or os.path.dirname(os.path.abspath(ar_path)), base)


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #
@dataclass
class Pair:
    file_id: str
    line_idx: int
    ar_raw: str
    ar_src: str
    en: str
    prev_ar: str = ""
    prev_en: str = ""


def load_corpus(data_dir: str, log=print) -> list[Pair]:
    pairs: list[Pair] = []
    ids = sorted(
        (pid for pid in (pair_id_from_path(f) for f in os.listdir(data_dir)) if pid),
        key=lambda s: (len(s), s),
    )
    for pid in ids:
        ar_path = os.path.join(data_dir, f"ar-{pid}.txt")
        en_path = os.path.join(data_dir, f"entranslation-{pid}.txt")
        if not os.path.exists(en_path):
            log(f"[skip] {pid}: no English file")
            continue
        ar = read_lines(ar_path).lines
        en = read_lines(en_path).lines
        while ar and not ar[-1].strip():
            ar.pop()
        while en and not en[-1].strip():
            en.pop()
        if len(ar) != len(en):
            if abs(len(ar) - len(en)) > 1:
                log(f"[skip] {pid}: line count mismatch ar={len(ar)} en={len(en)}")
                continue
            log(f"[warn] {pid}: line count mismatch ar={len(ar)} en={len(en)}; using first {min(len(ar), len(en))}")
        prev_ar = prev_en = ""
        for idx, (a, e) in enumerate(zip(ar, en)):
            src = clean_source(a)
            tgt = clean_target(e)
            if not src or not tgt:
                continue
            pairs.append(Pair(pid, idx, a, src, tgt, prev_ar, prev_en))
            prev_ar, prev_en = src, tgt
    return pairs


# --------------------------------------------------------------------------- #
# Translation memory
# --------------------------------------------------------------------------- #
class TranslationMemory:
    def __init__(self, entries: dict[str, str] | None = None):
        self.entries: dict[str, str] = entries or {}

    @classmethod
    def build(cls, pairs: Iterable[Pair]) -> "TranslationMemory":
        votes: dict[str, Counter] = defaultdict(Counter)
        for p in pairs:
            votes[tm_key(p.ar_raw)][p.en] += 1
        return cls({k: c.most_common(1)[0][0] for k, c in votes.items() if k})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, path: str) -> "TranslationMemory":
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh))

    def lookup(self, ar_line: str, fuzzy_threshold: float = 0.0) -> tuple[str | None, float]:
        """Return (english, score). score 1.0 = exact key match, else fuzzy ratio."""
        key = tm_key(ar_line)
        if not key:
            return None, 0.0
        hit = self.entries.get(key)
        if hit is not None:
            return hit, 1.0
        if fuzzy_threshold and fuzzy_threshold < 1.0 and len(key) >= 12:
            try:
                from rapidfuzz import process, fuzz
            except ImportError:
                return None, 0.0
            best = process.extractOne(
                key, self.entries.keys(), scorer=fuzz.ratio, score_cutoff=fuzzy_threshold * 100
            )
            if best:
                return self.entries[best[0]], best[1] / 100.0
        return None, 0.0


def dump_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pair_to_dict(p: Pair) -> dict:
    return asdict(p)
