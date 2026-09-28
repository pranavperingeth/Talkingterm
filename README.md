# TalkingTerm 🗣️💻

**Type plain English. Get a shell command.**

TalkingTerm is a small **natural-language → shell-command translator** built from scratch as a
sequence-to-sequence (encoder–decoder) neural network. No LLM API, no `transformers`, no
rule-based parser — just a word-level vocabulary, two stacked LSTM layers and a linear
projection, trained on a generated parallel corpus of English phrases and their equivalent
Unix commands.

It ships with an interactive REPL (`tt>` prompt) that translates your sentence, shows the
command it suggests, asks for confirmation, and only then runs it.

```
$ python3 shell.py
TalkingTerm  —  type 'exit' to quit

tt> find python files
  Suggested: find . -name "*.py"
  Execute? (y/n): y
./train.py
./talkingterm/translator.py
./talkingterm/shell.py

tt> delete notes.txt
  Suggested: rm notes.txt
  Execute? (y/n): n
  Skipped.

tt> exit
Bye!
```

---

## Table of contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Safety model](#safety-model)
- [Project status](#project-status)
- [Known issues & limitations](#known-issues--limitations)
- [Roadmap](#roadmap)
- [License](#license)

---

## How it works

The whole pipeline is a classic (pre-attention) seq2seq model, written with raw PyTorch in
[`train.py`](train.py):

```
  "find python files"
          │
          ▼
   tokenize + lowercase + pad/truncate to 20 tokens
          │
          ▼
   ┌───────────────┐        final (hidden, cell)
   │   ENCODER     │ ─────────────────────────────┐
   │  Embedding    │                              │
   │  LSTM(256)    │                              ▼
   └───────────────┘                    ┌────────────────────┐
                                        │     DECODER        │  greedy, one token
   "<SOS>" ────────────────────────────▶│  Embedding         │  at a time until
                                        │  LSTM(256)         │  "<EOS>" or 20 steps
                                        │  Linear → 422      │
                                        └────────────────────┘
                                                  │
                                                  ▼
                                    find . -name "*.py"
```

* **Vocabulary** – built from both sides of the corpus, so 422 tokens including the four
  specials `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`. Unknown words map to `<UNK>` instead of crashing.
* **Encoder** – `nn.Embedding(128)` → `nn.LSTM(128 → 256)`. Only the final hidden/cell state
  is kept; it becomes the decoder's initial state (that state *is* the "meaning" of your sentence).
* **Decoder** – `nn.Embedding(128)` → `nn.LSTM(256)` → `nn.Linear(256 → vocab)`. It is fed the
  previous token plus the carried-over state, and emits a distribution over the vocabulary.
* **Inference** – greedy `argmax` decoding, stopping at `<EOS>` (see `translate()` in
  [`talkingterm/translator.py`](talkingterm/translator.py)).

### Training configuration

| Hyper-parameter | Value |
| --- | --- |
| Embedding dim | 128 |
| Hidden dim | 256 |
| Max sequence length | 20 tokens |
| Batch size | 32 |
| Epochs | 50 |
| Optimizer | Adam, lr = 1e-3 |
| LR schedule | `ReduceLROnPlateau` (patience 5, factor 0.5) |
| Loss | `CrossEntropyLoss(ignore_index=0)` — `<PAD>` is ignored |
| Gradient clipping | 1.0 (`clip_grad_norm_`) |
| Trainable parameters | **1,007,014** |

Teacher forcing is used during training: the decoder input is the target sequence shifted right
and prefixed with `<SOS>`, while the label is the target sequence ending in `<EOS>`.

---

## Repository layout

```
TalkingTerm/
├── train.py                    # vocabulary, dataset, Seq2Seq model, training loop, save/load
├── talkingterm/
│   ├── translator.py           # loads checkpoint, exposes translate(sentence) -> str, CLI
│   └── shell.py                # interactive REPL: translate → confirm → execute
├── dataset/
│   └── dataset.csv             # 20,360 English → shell-command pairs
├── models/
│   └── model.pth               # trained checkpoint (state_dict + vocab, ~4 MB)
├── README.md
├── .gitignore
└── LICENSE
```

The checkpoint is self-contained: alongside the weights it stores `word2idx`, `idx2word` and
`vocab_size`, so the inference code can rebuild the exact same vocabulary the model was trained
with. That is why `translator.py` can load a model without touching `train.py`.

---

## Dataset

`dataset/dataset.csv` is a generated parallel corpus: **20,360 pairs**, **607 unique input
phrasings**, **155 unique shell commands** — roughly 33 paraphrases per command. That redundancy
is the point: it is what lets a word-level model learn that "locate", "search for" and "where are"
are interchangeable.

| Command family | Unique outputs | Notes |
| --- | --- | --- |
| `grep "<pattern>" <file>` | 103 | 103 combinations drawn from 11 patterns × 13 targets (`error`, `TODO`, `fixme`, `timeout`, …) |
| `find . -name "*.<ext>"` | 10 | `py`, `js`, `csv`, `log`, `md`, `json`, `html`, `css`, `java`, `txt` |
| `find . -<predicate>` | 5 | `-mtime -1`, `-size +100M`, `-user john`, plus two **malformed** outputs — `find . -name ".java"` / `find . -name ".py"` (the `*` wildcard is missing; a data-generation bug that survived into the corpus) |
| `tar -czvf <name>.tar <name>` | 11 | 10 archives (`project`, `backup`, `src`, `docs`, `logs`, `data`, `images`, `build`, `temp`, `archive`) plus one `tar -xzvf archive.tar.gz` extract |
| `ls -a` / `ls -l` / `ls -S` | 3 | show hidden, long format, sort by size |
| `chmod`, `chown`, `cp`, `mkdir`, `mv`, `rm`, `rmdir` | 10 | file operations, including `rm -rf folder` (which the REPL refuses to run) |
| system & network | 13 | `ps aux`, `top`, `kill 1234`, `df -h`, `free -m`, `uptime`, `uname -a`, `last`, `clear`, `pwd`, `ifconfig`, `netstat -an`, `ping google.com` |
| **Total** | **155** | across 20,360 pairs |

Five real rows, copied verbatim from the file:

```csv
input,output
find md files,"find . -name ""*.md"""
search for the word error in log.txt,grep "error" log.txt
give execute permission to script.sh,chmod +x script.sh
search for the word function in server.log,"grep ""function"" server.log"
show running processes,ps aux
```

Quoting is inconsistent — some rows wrap the command in quotes and double the inner ones, others
leave bare `"` characters in an unquoted field. `pandas.read_csv` / `csv.reader` handle both
identically, but a naive `line.split(",")` does not.

Two data bugs made it into the corpus and are worth knowing about:

* `find . -name ".py"` / `find . -name ".java"` — 19 rows where the wildcard was lost
  (should be `"*.py"` / `"*.java"`). The model learned to produce the correct command anyway,
  because the other ~12,000 `find` rows are well formed.
* **One input is genuinely ambiguous**: `search for java files` appears mapped to both
  `find . -name "*.java"` and `find . -name ".java"`. It is the only such collision — every other
  input maps to exactly one output — but it is a reminder that the corpus was generated, not
  curated.

There is **no train/validation/test split** — the model is trained on all 20,360 rows, so the
accuracy number below is effectively training accuracy and measures *memorisation of the
paraphrase patterns*, not generalisation to brand-new phrasings.

---

## Getting started

### 1. Requirements

* Python 3.9+ (developed on 3.14)
* PyTorch, pandas

No `requirements.txt` exists yet (see [Known issues](#known-issues--limitations)) — install the
two dependencies directly:

```bash
pip install torch pandas
```

### 2. Train the model

Run from the repository root, since the paths inside `train.py` are relative:

```bash
python3 train.py
```

```
Using device: cpu
Vocabulary size: 422
Epoch   1/50 | Loss: 0.8103 | LR: 0.001000
Epoch   5/50 | Loss: 0.0085 | LR: 0.001000
Epoch  10/50 | Loss: 0.0024 | LR: 0.001000
...
Epoch  50/50 | Loss: 0.0001 | LR: 0.000500

Model saved → models/model.pth

Test translation
  Input : find python files
  Output: find . -name "*.py"
```

(Real output, reproduced on an Apple-silicon laptop: a full 50-epoch run takes roughly 8 minutes
on CPU. The learning rate halves to `5e-4` around epoch 35 once the loss stops improving, courtesy
of `ReduceLROnPlateau`.)

Note how fast the loss collapses — down to `0.0001` by epoch 20. This model is not "learning
English to shell"; it is **memorising all 155 command templates** and the paraphrases attached to
them. That is why it is near-perfect on corpus-like phrasing and unreliable on anything else.

### 3. Run the REPL

> ⚠️ **One blocker first.** `talkingterm/translator.py` still points at the absolute path from
> the Windows machine this was written on:
>
> ```python
> MODEL_PATH = r"C:\Users\prana\talkingterm\models\model.pth"
> ```
>
> Change it to the checkpoint's real location, e.g.
>
> ```python
> MODEL_PATH = "models/model.pth"   # or an absolute path such as /path/to/TalkingTerm/models/model.pth
> ```
>
> Until that line is edited, both entry points fail with
> `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\prana\\...'`.

Then, from the `talkingterm/` directory (the REPL imports `translator` as a sibling module):

```bash
cd talkingterm
python3 shell.py
```

### 4. Translate without executing anything

`translator.py` is also runnable on its own for a dry run:

```bash
cd talkingterm
python3 translator.py
# Model loaded. Type 'quit' to exit.
# Input: compress folder images
# Output: tar -czvf images.tar images
```


---

## Usage

### Phrasings that were in the training corpus

| Input phrase | Model output | Matches the expected command? |
| --- | --- | --- |
| `search for the word error in log.txt` | `grep "error" log.txt` | ✅ exact (this is literally a corpus row) |
| `make script.sh executable` | `chmod +x script.sh` | ✅ exact |
| `delete notes.txt` | `rm notes.txt` | ✅ exact |
| `compress folder project` | `tar -czvf project.tar project` | ✅ exact |
| `show current directory` | `pwd` | ✅ exact |
| `find python files` | `find . -name "*.py"` | ✅ — and note the corpus label for this row is `find . -name ".py"` (wildcard missing, a data typo), so the model emits the *better* command than the one it was trained on |

Measured on 300 rows drawn at random from the corpus: **292/300 exact-string matches (97.3 %)**.

### Phrasings that never appear in the corpus

None of these inputs exist anywhere in the training set — this is the honest picture of what the
model actually generalised to:

| Input phrase | Model output | Verdict |
| --- | --- | --- |
| `show disk usage` | `df -h` | ✅ correct command |
| `what is my ip` | `ifconfig` | ✅ correct command |
| `show me running processes` | `ps aux` | ✅ correct command |
| `list every file` | `find . -name "*.java"` | ❌ invented `.java` from nowhere |
| `count lines in report.md` | `grep "import" report.md` | ❌ fell back to the nearest template |
| `say hello world` | `chmod 777 file.txt` | ❌ confidently nonsensical |

So the 97.3 % figure is **memorisation of paraphrase patterns, not language understanding**: the
model is near-perfect on corpus-like phrasing, hits the target roughly half the time on unseen
phrasing, and otherwise produces a syntactically valid command that does something entirely
different. Treat it as a from-scratch seq2seq demo, not as a shell copilot.

### Tuning a prediction

Greedy decoding is used everywhere and there is **no beam search and no attention**, so the
first token the model commits to pins the rest of the command. The knobs live in
`talkingterm/translator.py`:

* `MAX_LEN = 20` – maximum tokens the decoder may generate.
* `EMBED_DIM` / `HIDDEN_DIM` – must stay in sync with `train.py`, otherwise
  `load_state_dict` raises a shape mismatch.

### Retraining on your own phrases

Append rows to `dataset/dataset.csv` (`input,output`, quoted when the command contains commas or
double quotes) and re-run `python3 train.py`. The vocabulary and checkpoint are rebuilt from
scratch each time, so no migration is needed — but the old `models/model.pth` is overwritten.

---

## Safety model

Generated commands are executable, so `shell.py` puts two guards in front of `subprocess.run`:

1. **Blocklist** (`BLOCKED` in [`talkingterm/shell.py`](talkingterm/shell.py)) — recursively
   destructive or system-level commands are refused outright, before the confirmation prompt:

   ```python
   BLOCKED = [
       "rm -rf", "rmdir", "del /f", "format", "mkfs",
       "dd if=", ":(){:|:&};:", "shutdown", "reboot",
       "reg delete", "rd /s", "powershell -enc",
   ]
   ```

2. **Explicit confirmation** — nothing runs without a `y` at the `Execute? (y/n):` prompt.
   Anything else skips the command.

Supporting details: commands run through the shell with a **15-second timeout** (a hung command
is killed and reported rather than freezing the REPL), `stdout`/`stderr` are captured and echoed
back, non-zero exit codes are surfaced as `[exited with code N]`, and `Ctrl-C` / `Ctrl-D` exit
cleanly.

**This is a demo, not a hardened sandbox.** The blocklist is a naive substring match (easily
bypassed by quoting or rewriting a command), the REPL runs with your own privileges on your own
machine, and the model can fabricate a destructive-looking command from an innocent prompt —
which is exactly why the confirmation step exists. Run it in a scratch directory or a container.


---

## Project status

Where the project currently stands:

- [x] Parallel corpus generated — 20,360 pairs / 155 unique commands
- [x] Word-level vocabulary with `<PAD>` / `<SOS>` / `<EOS>` / `<UNK>` and padding
- [x] Seq2Seq LSTM encoder–decoder implemented in raw PyTorch (1.0 M parameters)
- [x] Training loop — teacher forcing, loss masking on `<PAD>`, gradient clipping, LR decay
- [x] Checkpointing that bundles the weights **and** the vocabulary together
- [x] Greedy inference exposed as a reusable `translate(sentence)` function
- [x] Interactive REPL with confirmation prompt, blocklist and command timeout
- [x] Model trained and committed (`models/model.pth`, 4 MB) — 97.3 % exact match on a 300-row sample
- [ ] Checkpoint path is still hardcoded to the original author's Windows machine
- [ ] No `requirements.txt`, no packaging, no installable entry point
- [ ] No validation split, so the accuracy figure is training accuracy
- [ ] No tests
- [x] Repo hygiene — `.gitignore` added and the stray `__pycache__/*.pyc` untracked

In short: **the model and the training pipeline are finished and working; the packaging and the
path plumbing around them are not.** Everything needed to reproduce the model is in the repo.

---

## Known issues & limitations

| # | Issue | Where | Impact |
| --- | --- | --- | --- |
| 1 | `MODEL_PATH` hardcoded to `C:\Users\prana\talkingterm\models\model.pth` | `talkingterm/translator.py:12` | Importing the module raises `FileNotFoundError` on any machine but the original |
| 2 | Modules import each other as top-level siblings (`from translator import translate`) | `talkingterm/shell.py:1` | Must `cd talkingterm` first; `python3 -m talkingterm.shell` fails |
| 3 | No `__init__.py`, no `pyproject.toml` / `requirements.txt` | repo root | Dependencies must be guessed; not pip-installable |
| 4 | Checkpoint loaded eagerly at import time | `talkingterm/translator.py:53` | The module cannot be imported without the weights present (no lazy loading) |
| 5 | No attention and no beam search | `talkingterm/translator.py` | The whole input is compressed into one fixed-size state and decoding is myopic |
| 6 | Can emit a plausible-but-wrong command | model | `say hello world` → `chmod 777 file.txt` |
| 7 | Encoder consumes `<PAD>` tokens (no `pack_padded_sequence`) | `train.py` `Encoder.forward` | Minor quality loss on short sentences |
| 8 | Truncation to 20 tokens happens silently | `train.py`, `translator.py` | Long sentences are cut off with no warning |
| 9 | Blocklist is substring-based | `talkingterm/shell.py` | Trivially bypassed; real safety depends on the user confirming |
| 10 | Accuracy is measured on training data only | — | Overstates real-world performance |

---

## Roadmap

1. **Make it run anywhere** — resolve `MODEL_PATH` relative to the file
   (`Path(__file__).resolve().parents[1] / "models" / "model.pth"`), fix the import style,
   add `__init__.py`.
2. **Add `requirements.txt` / `pyproject.toml`** and a console entry point so `talkingterm`
   is a command instead of a directory you have to `cd` into.
3. **Add attention** (Bahdanau or Luong) — the single biggest quality win for a model this size.
4. **Beam search** instead of greedy `argmax`.
5. **Split the corpus** into train/val/test and report a real held-out accuracy.
6. **Slot generalisation** — let the model copy parameters out of the input (`*.py`, folder names,
   search patterns) instead of memorising 155 exact commands, so unseen phrasings work.
7. **Better safety** — replace substring matching with argument-aware parsing
   (`shlex` + an allowlist of binaries) and show the fully resolved command before execution.
8. **Tests** — round-trip vocabulary encoding, checkpoint save/load, `is_dangerous` cases and a
   handful of golden translations.

---

## License

MIT © 2026 [pranavperingeth](https://github.com/pranavperingeth) — see [LICENSE](LICENSE).

Built as a from-scratch deep-learning exercise: every layer, the vocabulary, the training loop and
the REPL are hand-written, with no pretrained weights and no NLP framework.

