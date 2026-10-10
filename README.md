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
3. **Approved examples selected from `training_data`** (see the next section for exactly how).
4. **The Arabic with its vowel marks**, numbered, plus the already-translated lines of the same file for coherence.

A second **review pass** has Claude check its draft against the Arabic line by line (`--no-review` skips it).
Lines that already exist in `training_data` are returned verbatim from a translation memory and never sent to the
model. Output formatting is enforced: same line count, blank lines and line endings as the input; `=` counts
identical (a line whose count differs is re-translated, then hemistich by hemistich); `÷` copied from the Arabic
line; `*` dropped.

## How `training_data` is used

The whole corpus (205 file pairs, 3,519 aligned lines) is loaded and searched on every run, but it is **not** pasted
into the prompt in full: that would be roughly 400,000 to 600,000 tokens per call, sent twice per file (translation
and review). Instead the corpus is used in two ways.

### 1. Exact reuse (translation memory)

Every Arabic line of the input is normalised (vowel marks, `÷`, `*`, punctuation and alef variants removed) and
looked up among all 3,519 approved lines. If it already exists, its approved English is used verbatim and the line is
never sent to Claude. `--fuzzy 0.95` extends this to near-identical lines; `--no-tm` turns it off.

### 2. Examples placed in the prompt

For the lines that are new, three independent searches run over the whole corpus. Their results are merged and
duplicates removed:

| Source | What is picked | How much |
|---|---|---|
| **A. Poem excerpt** | The one approved poem closest to the input: same number of `=` breaks per line, scored by how many of the input's lines it has close matches for. Its first lines are included in order, to show register, line length and how refrains are handled. | up to 30 lines |
| **B. Verse-form samples** | Approved lines with the same number of `=` breaks as the input, one per file, so Claude sees how stanzas of that shape were laid out in English. | up to 6 lines per break count |
| **C. Similar lines** | For **each** new line separately, the 3 approved lines anywhere in the corpus that share the most 3-character sequences with it (Jaccard similarity of character trigrams on the unvocalised Arabic). These usually come from many different files, not from the poem in A. | 3 per new line, total capped by `--max-examples` (default 1000) |

The numbers 30, 6 and 3 are hand-picked defaults ("enough examples without a huge prompt"), not tuned values.

### How the cap of 1000 was chosen

The cap on source C was originally 40. Because the per-line picks are taken in line order, a cap of 40 ran out after
about the first 13 new lines, so in a long poem every later stanza went to Claude with no examples of its own.

Test on `content/ar-711.txt` (52 stanzas, 6 hemistichs each, no lines already in the corpus):

* No approved poem has the same form, so source A was empty and source B found only 2 lines.
* Source C produced 156 picks from 70 files. The best matches were weak (similarity around 0.11 on a 0 to 1 scale).
* With cap 40, 40 of those were sent. With cap 1000, all of them were sent.

Both versions were translated with identical settings (Fable 5.1, xhigh effort, review on):

| | Cap 40 | Cap 1000 |
|---|---|---|
| Time | 7:25 | 6:31 |
| Plan 5-hour usage after the run | 3% | 5% |
| Lines revised by the review pass | 9 | 13 |
| `=` breaks kept | 52/52 | 52/52 |

The two outputs shared 106 of 312 hemistichs word for word. The differences were almost all synonym choices ("doom"
vs "death" of enemies, "Verily" vs "Truly"); house terms (Allah, Ahmad, Mustafa, Fatema, salawaat ...) were identical.
Stanzas 1 to 13, which had the same examples in both runs, differed about as much as stanzas 14 to 52, so most of the
variation is run-to-run wording rather than the extra examples. For a poem with no close relatives in the corpus the
cap barely matters.

The cap was set to 1000 anyway: it costs a little more usage and guarantees that every new line, up to about 330 per
batch, gets its own examples, which matters for poems that do have close matches in the corpus. Use
`--max-examples 40` to reproduce the old behaviour.

### Implications

* Renderings that exist in the corpus but are not retrieved for a given line never reach Claude. Standing terms that
  must always be followed belong in `house_style.md`, which is sent in full every time.
* Retrieval is by shared letters, not meaning, so for unusual poems the "similar" lines can be only loosely related.
* Adding approved files to `training_data/` improves both exact reuse and the quality of the examples.

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
picking wrong variants), `--no-copy-div`, `--batch-lines 60`, `--max-examples 1000`, `--overwrite`,
`--claude-exe PATH`, `--model`, `--corpus`.

While it runs, a status line per Claude call shows the phase (waiting, thinking, writing), elapsed time, lines written
with an ETA, tokens used and your plan's 5-hour usage. At xhigh effort Claude thinks silently for a few minutes before
writing, so a 50-stanza poem takes about 6 to 8 minutes. If the Claude login has expired, the run stops at once with
instructions to `/login` again and writes no output file.

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
