# namaazdoa translator (Arabic -> English, house style)

Translates `ar-N.txt` files into `entranslation-N.txt` files of the same format, using Claude (Fable 5.1) with the
approved human translations in `training_data/` as context, so the output follows the established vocabulary and
phrasing of those translations.

## How it works

Every file is translated in one or two Claude calls whose prompt contains:

1. **Method and format rules** (`claude_backend.py`): read the whole passage first, identify speaker, addressee and
   allusions, keep person and tense, render every phrase, keep `=` breaks one-for-one, no `*` or `÷`, identical refrains.
2. **`house_style.md`**: the glossary of standing renderings (Allah, Husain, Maula, Fatema, salawaat, Dai, Nass ...)
   and style preferences. Edit this file to steer the output; no code change needed.
3. **Approved examples from the corpus**: an excerpt (up to 30 lines) of the approved poem whose form is closest to the
   input, plus the 3 most similar approved lines for each new line (character-trigram retrieval; `--max-examples`, default 1000, caps the total).
4. **The Arabic with its vowel marks**, numbered, plus the already-translated lines of the same file for coherence.

A second **review pass** has Claude check its draft against the Arabic line by line (`--no-review` skips it).
Lines that already exist in `training_data` are returned verbatim from a translation memory and never sent to the
model. Output formatting is enforced: same line count, blank lines and line endings as the input; `=` counts
identical (a line whose count differs is re-translated, then hemistich by hemistich); `÷` copied from the Arabic
line; `*` dropped.

## Data format

* `ar-N.txt` and `entranslation-N.txt` are line-aligned: Arabic line k <-> English line k.
* `=`  breaks between lines within a qasida stanza (preserved in English).
* `÷`  line-break marker used by the app (copied to the English line at the same position; `--no-copy-div` to stop).
* `*`  phrase separator inside an Arabic line (dropped in the English).
* Instruction lines (Lisan al-Dawat, Gujarati in Arabic script) are translated into English instructions.

## Setup

```bash
pip install -r requirements.txt
```

Then pick how Claude is reached:

**A. claude.ai subscription through the Claude Code CLI (default, no API billing).** Log the CLI in once: run
`claude` (or the desktop app's bundled `claude.exe`, which `translate.py` auto-detects), type `/login`, sign in with
your claude.ai account, `/exit`. Each translated file is one or two headless `claude -p` calls that count against
your plan's usage limits like any Claude Code session.

**B. Anthropic Console API key (pay-as-you-go, separate from the subscription).** Create a key at
console.anthropic.com and set `ANTHROPIC_API_KEY`; `translate.py` then uses the SDK automatically (`--transport api`
to force it). Roughly $0.50 to $3 per 50-stanza poem depending on effort and review.

## Translate

```bash
python translate.py content\ar-252.txt                 # writes content\entranslation-252.txt
python translate.py content -o out                     # every ar-*.txt in a folder
python translate.py content\ar-252.txt --show          # print AR / EN side by side
```

Options: `--effort high|xhigh|max` (default xhigh; `high` is about 4x cheaper and faster, `max` for the hardest
poetry), `--no-review`, `--no-tm`, `--fuzzy 0.95` (reuse approved lines that are near-identical; below 0.9 it starts
picking wrong variants), `--no-copy-div`, `--batch-lines 60`, `--overwrite`, `--claude-exe PATH`, `--model`,
`--corpus`.

## Evaluate

```bash
python prepare_data.py          # 185 training files / 20 held-out files (seed 42) into data/
python evaluate.py              # BLEU, chrF++ and '=' accuracy against the approved translations of the 20 files
```

`evaluate.py` restricts the examples and translation memory to the training files, so the score reflects unseen text.
Side-by-side output lands in `reports/`.

## Adding approved translations

Drop new `ar-N.txt` / `entranslation-N.txt` pairs into `training_data/`. They are picked up on the next run: exact
lines are reused verbatim, and the new lines become retrieval examples for everything similar.

## Repository layout

| File | Purpose |
|---|---|
| `translate.py` | command-line entry point |
| `claude_backend.py` | prompt construction, corpus retrieval, transports (Anthropic SDK / headless Claude Code), review pass |
| `common.py` | file format handling, normalisation, translation memory |
| `house_style.md` | editable glossary and style preferences appended to every prompt |
| `prepare_data.py`, `evaluate.py` | held-out split and scoring |
| `training_data/` | the approved corpus (205 file pairs, 3.5k aligned lines) |
| `content/` | working translations made with the tool; ignored by git |
