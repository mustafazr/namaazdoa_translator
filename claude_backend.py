"""Claude-powered translation engine.

Claude (default model claude-fable-5-1) translates each file with the approved corpus as context:
  * translation memory: lines already in training_data are reused verbatim (never sent to the model)
  * poem exemplar: for qasida/marsiya input, an excerpt of the closest approved poem in the same form
  * style retrieval: the most similar approved Arabic->English lines are shown as examples for every new line
  * house_style.md (editable) + method/format rules in a stable system prompt (prompt-cached across calls)
  * an optional review pass in which the model checks its draft against the Arabic line by line

Two transports:
  api          Anthropic Python SDK, needs ANTHROPIC_API_KEY (Console, pay-as-you-go)
  claude-code  runs the local Claude Code executable headless (`claude -p`), uses the claude.ai subscription login
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from common import (STANZA_BREAK, Pair, TranslationMemory, clean_source, hemistichs, load_corpus,
                    reattach_div_marks, tm_key)

DEFAULT_MODEL = "claude-fable-5-1"
HERE = os.path.dirname(os.path.abspath(__file__))

METHOD = """\
You translate Arabic supplications (dua, munajaat), qasida and marsiya verses, and short Lisan al-Dawat instructions
(Gujarati written in Arabic script) into English for the Fatemi Dawat app, in the approved house style shown by the
examples: dignified, plain, concise English with the vocabulary of the house glossary.

Method. First read the whole passage and work out what it is (dua, munajaat, qasida or marsiya, instruction), who
speaks and who is addressed in each line, and which persons, events or Quranic phrases are alluded to (for a marsiya:
the Karbala narrative - Abbas and the water, Ali Akbar's thirst, Ali Asghar and the arrow, Qasim the bridegroom,
Zaynab, Zayn al-Abidin and the Nass, Dhu'l-Janah, the tents, the afternoon of Ashura). Then translate each hemistich
faithfully: keep the grammatical person, number and tense of the Arabic; render every phrase (no shortening, no
padding); where a word has several senses choose the one that fits the narrative and the approved examples; keep the
Arabic's images (ark, musk, spring, sun) rather than explaining them, but render idioms, letter-play and formulae by
their sense in natural English, never letter by letter. The Arabic is given with its vowel marks - use them to
disambiguate.
Prefer the wording of the closest approved examples, and reuse an example's English verbatim when its Arabic is the
same line.

Output format (strict):
- Output exactly one line per numbered input line, as "N: translation", in the same order, and nothing else.
- Keep every '=' break: the output line must contain the same number of '=' as the input line, each hemistich
  translated in place and joined with '=' (no spaces around '='). Never merge or split hemistichs.
- Never output the characters '*' or '÷'. '*' in the Arabic only separates phrases; translate the phrases in sequence.
- A refrain repeated in several stanzas must be translated identically every time it appears.
- No explanations, footnotes, bracketed transliterations or commentary unless the examples do so for that phrase.
- Quranic verses marked with ﴿ ﴾ are rendered in « » with the reference in parentheses, as in the examples.
- Instruction lines in Lisan al-Dawat become concise English instructions ("Then recite this dua:", "Pray in the
  first rakaat alhamd and ...").
"""

FALLBACK_GLOSSARY = """\
Allah | Rasulullah | the Prophet | Muhammad | Ali | Fatema | Hasan | Husain | Zaynab | Abbas | Ali Akbar | Ali Asghar |
Qasim | Zayn al-Abidin | Haydar | Maula / Maulana | Imam | Dai, Dais | Wasi | Mumineen | namaaz | shower salawaat on |
salaam | dua | tawfeeq | Nass | Kawthar | Karbala | Taff | Ashura. Plain transliteration, no macrons or dots."""


def load_house_style() -> str:
    p = os.path.join(HERE, "house_style.md")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read()
    return FALLBACK_GLOSSARY


def build_system_prompt() -> str:
    return METHOD + "\nHouse style and glossary:\n" + load_house_style()


@dataclass
class LineResult:
    en: str
    source: str  # "blank" | "tm" | "tm-fuzzy" | "claude" | "claude-split" | "failed"
    seconds: float = 0.0


def show_ar(text: str) -> str:
    """Arabic as shown to Claude: vowel marks kept, ÷ removed, separators tidied."""
    return clean_source(text, keep_diacritics=True)


# --------------------------------------------------------------------------- #
# Retrieval over the approved corpus
# --------------------------------------------------------------------------- #
def _trigrams(text: str) -> set[str]:
    t = re.sub(r"[^\w\s=]", " ", text)
    t = re.sub(r"\s+", " ", t).strip()
    t = f" {t} "
    return {t[i:i + 3] for i in range(len(t) - 2)}


class StyleIndex:
    def __init__(self, pairs: list[Pair]):
        self.pairs = pairs
        self.grams = [_trigrams(p.ar_src) for p in pairs]
        self.inv: dict[str, list[int]] = defaultdict(list)
        for i, g in enumerate(self.grams):
            for tri in g:
                self.inv[tri].append(i)
        self.by_file: dict[str, list[Pair]] = defaultdict(list)
        for p in pairs:
            self.by_file[p.file_id].append(p)
        self.by_breaks: dict[int, list[int]] = defaultdict(list)
        for i, p in enumerate(pairs):
            n = p.ar_src.count(STANZA_BREAK)
            if n and p.en.count(STANZA_BREAK) == n:
                self.by_breaks[n].append(i)

    def similar(self, ar_src: str, k: int = 3, exclude_keys: set[str] | None = None) -> list[tuple[float, Pair]]:
        g = _trigrams(ar_src)
        if not g:
            return []
        counts: dict[int, int] = defaultdict(int)
        for tri in g:
            for i in self.inv.get(tri, ()):
                counts[i] += 1
        scored = []
        for i, c in counts.items():
            p = self.pairs[i]
            if exclude_keys and tm_key(p.ar_raw) in exclude_keys:
                continue
            scored.append((c / (len(g) + len(self.grams[i]) - c), p))  # Jaccard
        scored.sort(key=lambda x: -x[0])
        out, seen = [], set()
        for s, p in scored:
            if p.en in seen:
                continue
            seen.add(p.en)
            out.append((s, p))
            if len(out) >= k:
                break
        return out

    def form_exemplars(self, n_breaks: int, k: int = 6) -> list[Pair]:
        idx = self.by_breaks.get(n_breaks) or []
        if not idx:  # nearest available structure
            cands = sorted(self.by_breaks, key=lambda n: abs(n - n_breaks))
            idx = self.by_breaks[cands[0]] if cands else []
        out, files = [], set()
        for i in idx:
            p = self.pairs[i]
            if p.file_id in files:
                continue
            files.add(p.file_id)
            out.append(p)
            if len(out) >= k:
                break
        return out

    def poem_exemplar(self, srcs: list[str], exclude_keys: set[str] | None = None, max_lines: int = 30) -> list[Pair]:
        """An excerpt of the approved poem whose form and wording are closest to the input (same '=' count)."""
        breaks = Counter(s.count(STANZA_BREAK) for s in srcs if STANZA_BREAK in s)
        if not breaks:
            return []
        n = breaks.most_common(1)[0][0]
        score: dict[str, float] = defaultdict(float)
        for i in self.by_breaks.get(n, []):
            score[self.pairs[i].file_id] += 0.02
        for s in srcs[:60]:
            for sc, p in self.similar(s, 5, exclude_keys):
                if p.ar_src.count(STANZA_BREAK) == n:
                    score[p.file_id] += sc
        if not score:
            return []
        best = max(score, key=score.get)
        lines = [p for p in self.by_file[best] if not (exclude_keys and tm_key(p.ar_raw) in exclude_keys)]
        return lines[:max_lines]


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
class AnthropicTransport:
    name = "api"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "xhigh", log=print):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort, self.log = model, effort, log
        self.usage = defaultdict(int)

    def complete(self, system: str, user: str, max_tokens: int = 32000) -> str:
        import anthropic
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            with self.client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        except anthropic.APIStatusError as e:
            raise RuntimeError(f"Claude API error {e.status_code}: {e.message}") from e
        u = msg.usage
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            self.usage[k] += getattr(u, k, 0) or 0
        if msg.stop_reason == "refusal":
            det = getattr(msg, "stop_details", None)
            raise RuntimeError(f"Claude declined this request ({getattr(det, 'category', None)}): {getattr(det, 'explanation', '')}")
        if msg.stop_reason == "max_tokens":
            raise RuntimeError("Claude output was cut off by max_tokens; use a smaller --batch-lines")
        return "".join(b.text for b in msg.content if b.type == "text")


def find_claude_exe(explicit: str | None = None) -> str | None:
    """Locate a Claude Code executable: explicit path, CLAUDE_EXE, PATH, standalone install, or the desktop app's bundle."""
    import glob
    cands = [explicit, os.environ.get("CLAUDE_EXE"), os.environ.get("CLAUDE_CODE_EXECPATH"),
             shutil.which("claude"), shutil.which("claude.exe"), shutil.which("claude.cmd"),
             os.path.expanduser("~/.local/bin/claude.exe"), os.path.expanduser("~/.local/bin/claude")]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        roots = glob.glob(os.path.join(local, "Packages", "Claude_*", "LocalCache", "Roaming", "Claude", "claude-code"))
        roots += [os.path.join(os.environ.get("APPDATA", ""), "Claude", "claude-code")]
        bundled = [p for r in roots for p in glob.glob(os.path.join(r, "**", "claude.exe"), recursive=True)]

        def ver(p: str):
            # version folder is the first path component under claude-code, e.g. 2.1.286
            rel = p.split("claude-code", 1)[-1].strip("\\/").split(os.sep)[0].replace("/", "\\").split("\\")[0]
            return tuple(int(x) if x.isdigit() else 0 for x in rel.split("."))

        cands += sorted(bundled, key=lambda p: (ver(p), os.path.getmtime(p)), reverse=True)
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


class ClaudeCodeTransport:
    """Headless Claude Code (`claude -p`) - runs on the claude.ai subscription login of this machine."""
    name = "claude-code"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "xhigh", exe: str | None = None, log=print):
        self.exe = find_claude_exe(exe)
        if not self.exe:
            raise RuntimeError("Claude Code executable not found. Install the CLI (PowerShell: irm https://claude.ai/install.ps1 | iex) "
                               "or pass --claude-exe PATH (or set CLAUDE_EXE).")
        self.model, self.effort, self.log = model, effort, log
        self.usage = defaultdict(int)
        self.cost_usd = 0.0

    def complete(self, system: str, user: str, max_tokens: int = 32000) -> str:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}  # allow running from inside Claude Code
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(system)
            sys_path = fh.name
        cmd = [self.exe, "-p", "--model", self.model, "--output-format", "json", "--tools", "",
               "--no-session-persistence", "--system-prompt-file", sys_path]
        if self.effort:
            cmd += ["--effort", self.effort]
        try:
            proc = subprocess.run(cmd, input=user.encode("utf-8"), capture_output=True, env=env, timeout=3600)
        finally:
            try:
                os.unlink(sys_path)
            except OSError:
                pass
        out = proc.stdout.decode("utf-8", errors="replace")
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            raise RuntimeError(f"claude -p returned non-JSON (exit {proc.returncode}): {out[:500]} {proc.stderr.decode('utf-8', 'replace')[:500]}")
        if data.get("is_error"):
            msg = str(data.get("result"))
            if "not logged in" in msg.lower():
                raise RuntimeError("Claude Code CLI is not logged in. Open a terminal, run `claude`, type `/login` and sign in "
                                   "with your claude.ai account once; then re-run. (Or use --transport api with ANTHROPIC_API_KEY.)")
            raise RuntimeError(f"claude -p error: {msg[:500]}")
        self.cost_usd += float(data.get("total_cost_usd") or 0)
        for k, v in (data.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                self.usage[k] += v
        return data.get("result") or ""


def make_transport(kind: str, model: str, effort: str, log=print, claude_exe: str | None = None):
    if kind == "auto":
        kind = "api" if os.environ.get("ANTHROPIC_API_KEY") else "claude-code"
    if kind == "api":
        return AnthropicTransport(model, effort, log)
    if kind == "claude-code":
        return ClaudeCodeTransport(model, effort, exe=claude_exe, log=log)
    raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
_NUM_LINE = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(.*)$")


def parse_numbered(text: str, expected: list[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in text.replace("\r", "").split("\n"):
        m = _NUM_LINE.match(line)
        if m:
            n = int(m.group(1))
            if n in expected and n not in out:
                out[n] = m.group(2).strip()
        elif out and line.strip():  # continuation of the previous line (should not happen; keep it)
            last = max(out)
            out[last] = (out[last] + " " + line.strip()).strip()
    return out


def _clean_en(en: str) -> str:
    en = en.replace("*", " ").replace("÷", " ")
    en = re.sub(r"\s*=\s*", "=", en)
    en = re.sub(r"[ \t]+", " ", en).strip()
    return en


class ClaudeEngine:
    def __init__(self, corpus_pairs: list[Pair], transport, tm: TranslationMemory | None = None, copy_div: bool = True,
                 fuzzy: float = 0.0, examples_per_line: int = 3, max_examples: int = 40, batch_lines: int = 60,
                 review: bool = True, exclude_keys: set[str] | None = None, verbose: bool = True):
        self.index = StyleIndex(corpus_pairs)
        self.transport = transport
        self.tm = tm
        self.copy_div = copy_div
        self.fuzzy = fuzzy
        self.k = examples_per_line
        self.max_examples = max_examples
        self.batch_lines = batch_lines
        self.review = review
        self.exclude_keys = exclude_keys
        self.verbose = verbose
        self.system_prompt = build_system_prompt()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ---- prompt ---------------------------------------------------------- #
    @staticmethod
    def _fmt_pairs(pairs: list[Pair]) -> list[str]:
        return [f"AR: {show_ar(p.ar_raw)}\nEN: {p.en}\n" for p in pairs]

    def _examples_for(self, srcs: list[str]) -> tuple[list[Pair], list[Pair]]:
        poem = self.index.poem_exemplar(srcs, self.exclude_keys)
        seen = {p.en for p in poem}
        chosen: list[Pair] = []
        breaks = {s.count(STANZA_BREAK) for s in srcs if STANZA_BREAK in s}
        for n in sorted(breaks):
            for p in self.index.form_exemplars(n):
                if p.en not in seen:
                    seen.add(p.en)
                    chosen.append(p)
        for s in srcs:
            for _, p in self.index.similar(s, self.k, self.exclude_keys):
                if p.en not in seen:
                    seen.add(p.en)
                    chosen.append(p)
        return poem, chosen[: self.max_examples]

    def _user_prompt(self, numbered: list[tuple[int, str]], poem: list[Pair], examples: list[Pair],
                     context: list[tuple[int, str, str]]) -> str:
        parts: list[str] = []
        if poem:
            parts += ["Approved translation of a poem in the same form (excerpt) - match its register, line length and "
                      "the way it handles refrains:", ""]
            parts += self._fmt_pairs(poem)
        if examples:
            parts += ["Approved translations of similar lines (follow their wording and term choices):", ""]
            parts += self._fmt_pairs(examples)
        if context:
            parts += ["Lines of the same file that are already translated (for coherence; do not output these):", ""]
            for n, a, e in context[-12:]:
                parts.append(f"{n}. AR: {a}\n   EN: {e}\n")
        parts += ["Translate the following lines. Output exactly one line per input line as \"N: translation\".", ""]
        for n, s in numbered:
            parts.append(f"{n}: {s}")
        return "\n".join(parts)

    def _review_prompt(self, numbered: list[tuple[int, str]], draft: dict[int, str], poem: list[Pair]) -> str:
        parts = ["Review this draft translation against the Arabic, line by line. Correct: misread words, wrong "
                 "speaker, person or tense, dropped or added phrases, images paraphrased away, glossary spellings, "
                 "refrains that differ between stanzas, and any line whose '=' count differs from the Arabic. Keep "
                 "lines that are already right exactly as they are. Output every line as \"N: translation\".", ""]
        if poem:
            parts += ["Approved poem in the same form (excerpt), for register:", ""]
            parts += self._fmt_pairs(poem[:12])
        parts += ["Arabic and draft:", ""]
        for n, s in numbered:
            parts.append(f"{n}: AR: {s}\n{n}: EN: {draft.get(n, '')}\n")
        return "\n".join(parts)

    # ---- translation ------------------------------------------------------ #
    def _call(self, numbered: list[tuple[int, str]], poem: list[Pair], examples: list[Pair], context) -> dict[int, str]:
        user = self._user_prompt(numbered, poem, examples, context)
        text = self.transport.complete(self.system_prompt, user)
        return parse_numbered(text, [n for n, _ in numbered])

    def _fix_structure(self, shown: str, en: str, poem, examples, context) -> tuple[str, str]:
        """Ensure '=' count matches; retry the single line, then translate hemistich by hemistich."""
        n = shown.count(STANZA_BREAK)
        if en and en.count(STANZA_BREAK) == n:
            return en, "claude"
        for _ in range(2):
            got = _clean_en(self._call([(1, shown)], poem, examples, context).get(1, ""))
            if got and got.count(STANZA_BREAK) == n:
                return got, "claude"
        parts = hemistichs(shown)
        numbered = [(i + 1, h) for i, h in enumerate(parts) if h]
        got = self._call(numbered, poem, examples, context)
        pieces = [_clean_en(got.get(i + 1, "")).replace(STANZA_BREAK, " ") if h else "" for i, h in enumerate(parts)]
        return STANZA_BREAK.join(pieces), "claude-split"

    def translate_lines(self, raw_lines: list[str]) -> list[LineResult]:
        t0 = time.time()
        results: list[LineResult | None] = [None] * len(raw_lines)
        srcs = [clean_source(r) for r in raw_lines]          # unvocalised: retrieval + TM
        shown = [show_ar(r) for r in raw_lines]              # vocalised: what Claude sees
        todo: list[int] = []
        context: list[tuple[int, str, str]] = []
        for i, (raw, src) in enumerate(zip(raw_lines, srcs)):
            if not src:
                results[i] = LineResult("", "blank")
                continue
            if self.tm is not None:
                hit, score = self.tm.lookup(raw, self.fuzzy)
                if hit is not None:
                    results[i] = LineResult(hit, "tm" if score >= 1.0 else "tm-fuzzy")
                    context.append((i + 1, shown[i], hit))
                    continue
            todo.append(i)

        for start in range(0, len(todo), self.batch_lines):
            batch = todo[start:start + self.batch_lines]
            numbered = [(i + 1, shown[i]) for i in batch]
            poem, examples = self._examples_for([srcs[i] for i in batch])
            ctx = sorted(context)[-12:]
            try:
                got = self._call(numbered, poem, examples, ctx)
            except RuntimeError as e:
                self._log(f"[error] {e}")
                got = {}
            got = {n: _clean_en(v) for n, v in got.items()}
            if self.review and got:
                try:
                    rev = parse_numbered(self.transport.complete(self.system_prompt, self._review_prompt(numbered, got, poem)),
                                         [n for n, _ in numbered])
                    changed = 0
                    for n, s in numbered:
                        r = _clean_en(rev.get(n, ""))
                        if r and r.count(STANZA_BREAK) == s.count(STANZA_BREAK) and r != got.get(n):
                            got[n] = r
                            changed += 1
                    self._log(f"review pass: {changed}/{len(numbered)} lines revised")
                except RuntimeError as e:
                    self._log(f"[review skipped] {e}")
            for i in batch:
                en = got.get(i + 1, "")
                if not en:
                    try:  # one retry for a missing line
                        en = _clean_en(self._call([(i + 1, shown[i])], poem, examples, ctx).get(i + 1, ""))
                    except RuntimeError as e:
                        self._log(f"[error] line {i + 1}: {e}")
                if not en:
                    results[i] = LineResult("", "failed")
                    continue
                try:
                    en, tag = self._fix_structure(shown[i], en, poem, examples, ctx)
                except RuntimeError as e:
                    self._log(f"[error] line {i + 1}: {e}")
                    tag = "claude"
                results[i] = LineResult(en, tag)
                context.append((i + 1, shown[i], en))
        out: list[LineResult] = []
        for raw, r in zip(raw_lines, results):
            assert r is not None
            if r.source not in ("blank", "failed"):
                r.en = reattach_div_marks(raw, r.en, self.copy_div)
            out.append(r)
        if out:
            out[-1].seconds = time.time() - t0
        return out


def load_corpus_pairs(corpus: str) -> list[Pair]:
    """corpus = a training_data folder or a jsonl produced by prepare_data.py"""
    if corpus.lower().endswith(".jsonl"):
        from common import load_jsonl
        return [Pair(**r) for r in load_jsonl(corpus)]
    return load_corpus(corpus, log=lambda *_: None)
