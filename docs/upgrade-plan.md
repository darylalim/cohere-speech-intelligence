# Plan: upgrade to speech intelligence

Status: **drafted 2026-11-17, revised the same day for Cohere models only. Nothing started.**

**How it was written.** A planning session read `granite-speech-intelligence` (v0.19.0: its
`CLAUDE.md`, `docs/PLAN.md`, `streamlit_app.py`, `evals/` and tests) and this repo, and decided
what carries over. A second pass then replaced that repo's Granite 4.2 LLM with **Cohere Labs Tiny
Aya Global**. Its tokenizer, chat template and cache facts were checked against the real files
(see Model).

**Where it applies.** Implementation happens in later sessions, on the MacBook Pro. CLAUDE.md is
the description of what ships. Where it and this file disagree once work lands, CLAUDE.md and the
code win. Update this status line as rows land.

## Goal

Add **transcript analysis**:

- **Summary card.** Topic, two-to-four-sentence summary, key topics and action items, as JSON
  parsed with `json.loads`.
- **Question board.** Editable yes/no questions, each scored by P(yes) read from the next-token
  logits rather than generated, and ranked.
- **One shared prefill.** Both read one prefilled KV cache of the finished transcript, so adding
  a question costs a forward pass, not another read of the transcript.

**Cohere models only.** Cohere Transcribe stays the speech model. The analysis model is Tiny Aya
Global (3.35B, 8-bit MLX) through `mlx-lm`, with Aya Expanse 8B (4-bit MLX) evaluated beside it
as the comparison.

**Unchanged promises.** Analysis is off by default, never re-transcribes, and sends nothing off the
machine.

The design is the sibling repo's, which was measured and evaluated there. **Four of its
assumptions do not hold for Cohere's models** (see "What changes because the model is Cohere's"),
and the plan is shaped around them.

The app's new name is **open** (see Open decisions). Nothing in Tracks A–B depends on it.

## What carries over from granite-speech-intelligence

| Item | Verdict | Why |
|---|---|---|
| Summary + yes/no board from one shared prefill (`prefill_transcript`, `summarize_transcript`, `score_question`, `answer_questions`, `run_analysis`) | **Port, redesigned in three places** (B3–B4) | The shape carries. The prompt layout, the cache rewind and the length cap do not (next section). |
| Granite's prompts, P(yes) results and `MAX_ANALYSIS_TOKENS = 24_576` | **Starting point only** | Measured on Granite 4.2 3B over lowercase TurboCTC text. Every number is re-measured here, and prompts are re-chosen on the tuning half. |
| `mx.set_cache_limit(1 GiB)` once per process | **Port** (A3) | There it took a 30-min run's footprint from 8.47 to 2.49 GB, with an identical transcript. Measure here first. |
| `inference_lock` + polling `_inference_turn` | **Port** (A4) | Two tabs can call `generate` on one cached model at once today. It is mandatory once a second model shares the GPU. |
| Pinned 40-hex revisions, with a test | **Port, for two repos** (B3) | Weights and tokenizer come from different repos here (Model). Both pins vouch for code-like files: the chat template runs. |
| `load_llm` error mapping (offline, disk full) | **Port, adapted** (B3) | Plus gated access, matched by type the way `load_asr` already does. |
| `llm_is_cached` caption ("first analysis downloads ~4 GB") | **Port** (B5) | |
| Model text through `st.text`; Markdown export escaped | **Port** (B5) | This repo already has the rule, and a stronger `escape_markdown`. |
| `evals/` (AMI, 180 reviewed labels, ROC-AUC/Brier, `--rescore`) | **Port, adapted** (B6) | The LLM's quality oracle. It now also has to pick between two models. |
| "Removing the analysis model" README section | **Port** (B7) | |
| `tests/llm_fakes.py` (`FakeLlm`, `ScriptedGenerate`) | **Do not port** | A scripted `generate_step` is the mocked-`generate` shape this repo forbids. See Tests. |
| Stop hook; `select_source`; PyAV decoding; demo video; conventional commits | **Do not port / not now** | Stop hook: deleted here on purpose. `select_source`: the segmented control mounts one input at a time, so there is nothing to fix. PyAV: it needs its own plan (Out of scope). Demo video: out of scope. Conventional commits: this repo uses plain imperative subjects. |
| Video uploads (mp4/mov/mkv) | **Optional** (C3) | Cheap: the ffmpeg fallback already writes a temp file so a trailing `moov` decodes. |

## What changes because the model is Cohere's

Verified 2026-11-17 against the real tokenizer and template files and against `mlx-lm` 0.32.2's
source. Each of these breaks a sibling-repo assumption.

1. **The chat template rejects two user turns in a row.** Granite's prompt was system turn,
   transcript user turn, then a question user turn. Tiny Aya's template raises "Conversation roles
   must alternate". Two layouts were considered:
   - **Transcript in the system turn. Rejected.** The template renders the system turn as a
     "Developer Preamble" that "take[s] precedence over instructions in the default preamble and
     user prompt". Spoken words would get system-level authority, the wrong direction for prompt
     injection.
   - **Transcript and task in one user turn. Chosen.** The user turn is
     `<transcript>\n…\n</transcript>` followed by `\n\n<task>`. **The cached prefix ends right
     after `</transcript>`**, and every task's text starts with `\n\n`. This boundary was tested
     on both candidates across four transcripts × four tasks, Chinese and French included: the
     prefix tokens were identical in all 24 cases for each model. Ending the prefix after the
     `\n\n` instead failed 18 of 24 for both, because the tokenizer merges trailing newlines into
     the next word.
2. **The cache can't be rewound past 4096 tokens.** Tiny Aya is `cohere2`: three of every four
   layers use sliding-window attention with a 4096-token window. `mlx-lm` gives those layers a
   `RotatingKVCache(max_size=4096)`, whose `is_trimmable()` is `offset < max_size`. Granite rewinds
   the cache after every question with `trim_prompt_cache`, and that stops working once a
   transcript passes ~4K tokens, about 20 minutes.
   - **Replacement:** snapshot every layer's cache `state` and `meta_state` before a task, and
     restore them after it in a `finally`. MLX arrays are values, so holding the snapshot costs a
     copy only when the cache is next written to. That is ~226 MB for the 27 rotating layers at a
     full window, and 0.018 MB per token for the 9 global layers.
   - **Why snapshot rather than trim:** one mechanism serves both architectures (Aya Expanse is
     plain `cohere` with an ordinary `KVCache`), and it removes the sibling repo's one-token
     `offset` arithmetic after a summary.
   - **Before any app code:** Phase 0 must show the restore is exact.
3. **Context is 8K tokens, not 24K+.** Both model cards say 8K input. Tiny Aya's template also
   injects a ~383-token default preamble ("contextual safety mode", "Your name is Aya", …) into
   every prompt. Its own system text goes under the Developer Preamble heading.
   - **Usable transcript:** roughly 7.5K tokens, about 40–45 minutes of English speech.
   - **Over the cap:** refuse with the exact count, never truncate, as the sibling repo does.
   - **Longer meetings** need a chunked analysis, which becomes the first follow-up (Out of scope).
4. **The `mlx-community` conversion ships a broken tokenizer.** The conversion was made 2026-02-18
   with mlx-lm 0.28.3, and its `tokenizer.json` pre-tokenizer is `Digits(individual_digits=True)` +
   GPT-2 `ByteLevel`. Cohere's own is a custom regex that keeps contractions and groups digits in
   threes.
   - **The difference, measured:** `We'll` → `["We'll"]` upstream but `["We", "'ll"]` converted;
     `€25,000!` → 5 tokens against 8; `3 p.m.` splits differently.
   - **Why it matters:** the model would be reading token sequences it was never trained on,
     precisely on numbers, dates and money, which three of the nine default questions ask about.
   - **Upstream already fixed it.** Cohere's commit `af89d219` (2026-09-30) is titled "Set
     tokenizer_class to PreTrainedTokenizerFast so transformers v5 loads the pre-[tokenizer]".
     The conversion predates it. The rendered chat template is identical between the two;
     only the tokenizer differs.
   - **Decision:** load **weights from the conversion and the tokenizer from Cohere's repo**
     (Model).

Smaller facts that also differ from Granite:

- **Stop tokens.** EOS is `<|END_OF_TURN_TOKEN|>` (id 6). The generation prompt ends
  `<|START_RESPONSE|>` (261000), and `<|END_RESPONSE|>` (261001) exists. `generation_config.json`
  names `<EOS_TOKEN>` (3). Stop on all three until Phase 0 shows which the model emits.
- **No `enable_thinking`.** Tiny Aya Global is not a reasoning variant, so the answer is the first
  token after `<|START_RESPONSE|>`.
- **`Yes` / `No` are single tokens:** 15689 / 4182 for Tiny Aya, 20343 / 5520 for Aya Expanse.
- **Vocabulary is 262K entries** (Granite: ~100K). A 2048-token prefill chunk computes
  2048 × 262,144 logits it then throws away, about 1 GB in bf16. Phase 0 measures that, and tests
  calling the inner `model.model(...)` for prefill so the LM head never runs (verify the attribute
  in `mlx_lm/models/cohere2.py`).
- **Tokens per word ≈ 1** on English text with the upstream tokenizer (9 words → 9 tokens), and
  ≈ 0.7 per character on Chinese.
- **KV cache is small.** 0.074 MB per token across all 36 layers if all were global, but only the 9
  global layers grow, at 0.018 MB per token. Aya Expanse: 0.131 MB per token, all 32 layers.

## Model

| Role | Repo | Revision | Notes |
|---|---|---|---|
| Analysis weights | `mlx-community/tiny-aya-global-8bit-mlx` | `5d5dc68b477abd7d583563a515bed98604d45d52` | 3.56 GB safetensors, `cohere2`, 36 layers, 8-bit affine (group 64). Not gated. |
| Analysis tokenizer + template | `CohereLabs/tiny-aya-global` | `af89d219b53ed9b13b8a4645f9c8028973510324` | Tokenizer files only (`*.json`, `*.jinja`, `*.txt`, `tokenizer.model`), **never `*.py`**. Gated "auto": the user accepts the terms once on the Hub. This account already has. |
| Comparison (B6 only) | `mlx-community/aya-expanse-8b-4bit` | `3a7cee37dd4ab4ed76642f61f6253da4748e63ba` | 4.52 GB, `cohere`, 8K context. Its upstream `CohereLabs/aya-expanse-8b` is gated and **not yet accepted on this account**, so its tokenizer could not be compared. Do that in Phase 0 before scoring it. |

**Licence: CC-BY-NC-4.0 for all of them**, plus Cohere Labs' Acceptable Use Policy.

- **Personal use is fine.** The app downloads the weights rather than redistributing them, and its
  own code stays MIT.
- **The README must say so,** in the feature list and next to the analysis toggle's help: turning
  analysis on downloads non-commercial weights.

**Loading.** Not `mlx_lm.load(repo)`, because weights and tokenizer come from different repos.
`load_llm` makes three calls:

1. `snapshot_download(TOKENIZER_REPO, revision=TOKENIZER_REVISION, allow_patterns=[...])` with an
   explicit list that excludes `*.py`.
2. `mlx_lm.utils.load_model(weights_path, strict=True)`, where `weights_path` is the weights repo
   fetched at `WEIGHTS_REVISION`.
3. `mlx_lm.utils.load_tokenizer(local_tokenizer_path, eos_token_ids=STOP_IDS)`, given the **local
   path** from step 1.

Do not hand `load_tokenizer` a repo id. It takes no `revision`, and its download patterns include
`*.py`. Leave `trust_remote_code` at `False`; never pass `True`.

**Startup check.** A tokenizer that fails it raises `ModelWeightsError`, the same class an
incompatible checkpoint raises. Check three things:

- the pre-tokenizer keeps `We'll` as one token;
- `Yes` and `No` are single tokens;
- the rendered generation prompt ends `<|START_RESPONSE|>`.

## Order of work

Three tracks, top to bottom, one commit per row unless stated. Each row ends with **done when**.
Don't start the next row until it is met. Every row runs the four local checks: `uv run pytest`,
`uvx ruff@0.16.1 check .`, `uvx ruff@0.16.1 format --check .` and `uvx ty check .`.

**Do not bump `[project] version` until the last row.** Here a version bump is the release.

### Track A: foundations (no LLM yet)

| # | Commit | Done when |
|---|---|---|
| A0 | (no commit) Session setup on the MacBook Pro | `sysctl -n hw.memsize` and `machdep.cpu.brand_string` recorded at the top of Phase 0 results. Then `uv sync`, `uv run --frozen hf auth login`, and `uv run verify_transcription.py` green, giving the baseline. Accept the terms on `CohereLabs/aya-expanse-8b` if B6 will score it. |
| A1 | `Raise the Python floor to 3.11 for mlx-lm` | `requires-python = ">=3.11"`, `uv lock` re-run, lock diff reviewed. ty now targets 3.11: re-run it and fix or explain new diagnostics. CLAUDE.md's "ty checks against Python 3.10 … `StrEnum` degrades to `Unknown`" passage rewritten. The `toml`-not-`tomllib` note in `tests/test_smoke.py` and CLAUDE.md re-justified on its remaining ground ("`toml` is what `streamlit.config` parses with") or switched, as a written decision. |
| A2 | `Upgrade mlx-audio to 0.5.9 within the <0.6 ceiling` | Diff the 0.5.4 and 0.5.9 wheels' `stt/models/cohere_asr/`, `vad/` and `utils.py` (changelogs are not trusted here). `_segment_with_vad` and the `max_chunk_s` keyword confirmed unchanged, or the shim updated. `verify_transcription.py` green, VAD silence check included. Every "0.5.4" in CLAUDE.md and the `pyproject.toml` comments updated. Its own commit. |
| A3 | `Cap the MLX buffer cache` | Measured first, on a 30-min meeting with VAD on and off: `phys_footprint` lifetime max via `proc_pid_rusage`, with and without `mx.set_cache_limit(1 << 30)`, and a byte-identical transcript. Ship it only if it helps. Once per process via an `@st.cache_resource` wrapper, never at module scope. Numbers go in the source comment. |
| A4 | `Serialise inference across browser tabs` | A process-wide `threading.Lock` from `@st.cache_resource`, held around `model.generate` and later the analysis, never around a model load. Acquired by 0.5 s polling with an `st.*` call per poll so a waiting run can be stopped. Status text says it is waiting for another tab. Unit test: one lock per process, released on raise. |

### Track B: transcript analysis

| # | Work | Done when |
|---|---|---|
| B1 | `Add mlx-lm for transcript analysis` | `uv add 'mlx-lm>=0.32.2'`. Every package new to `uv.lock` listed and reviewed in the commit body. `uv sync --locked` passes on the ubuntu `test` job, which needs verifying rather than assuming. `verify_transcription.py` still green. That matters because B1 moves mlx (0.32.2 → 0.32.4) and transformers (5.17 → 5.23) under the speech model too. |
| B2 | Phase 0 spike (scratchpad only, no app code) | Every item in **Phase 0** has a number under "Phase 0 results", and **Decisions (end of Phase 0)** are made. |
| B3 | `Summarise transcripts with Tiny Aya` | `utils/analysis.py`: `load_llm`, the length pre-check, `prefill_transcript`, the cache snapshot/restore, `summarize_transcript`, `parse_summary`. Unit tests pure, contract tests on a tiny real `cohere2` model (Tests). |
| B4 | `Score yes/no questions from the shared prefill` | `DEFAULT_QUESTIONS`, `clean_questions`, `yes_probability`, `score_question`, `answer_questions`, the low-mass warning. The three restore-equivalence contract tests pass. |
| B5 | `Add the analysis card and question board` | UI per **UI** below. AppTest seeded tests green. Rendered with real models in both themes at 1920×839 and at a 1440-wide laptop window, checked by eye. |
| B6 | `Add the analysis evaluation` | `evals/` ported. Both candidates scored on Cohere transcripts of the AMI excerpts, plus the multilingual known-answer set. Results recorded under Evaluation, and D0 (the model) confirmed or switched with the reason written down. |
| B7 | `Document transcript analysis` | CLAUDE.md, README.md (features, licence note, how it works, memory, removing the model), and `.gitignore` for `evals/data/` and `evals/results/`. Status line updated. |

B6 can run before B5, and should if Phase 0 makes the choice between the two models look close:
there is no point building UI around the wrong model.

### Track C: release

| # | Commit | Done when |
|---|---|---|
| C1 | `Rename the app to <name>` | Name decided. Change only the app-name uses: `README.md` heading, `page_title` and `st.title`, the `get_by_role("heading", name=…)` wait in `docs/screenshots.py` (it hangs otherwise), and the `pyproject.toml` description. Uses that name the *model* stay. |
| C2 | `Recapture the README screenshots` | `docs/screenshots.py` extended to capture the analysis card. All PNGs recaptured per CLAUDE.md, Screenshots. Alt-text numbers re-read off the new files. |
| C3 | (optional) `Accept mp4, mov and mkv uploads` | A row each in `UPLOAD_TYPES` and `PREVIEW_MIME`, and a passing row in `check_decoding`. |
| C4 | `Bump version to 0.2.0` | Last. Pushing it is the release. |

A2 and B1 are each their own commit, so a dependency break is attributable to one change.

## Step 0 facts (verified 2026-11-17, re-check in A1/B1)

Dry resolves with `uv pip compile` for `aarch64-apple-darwin`, `MACOSX_DEPLOYMENT_TARGET=14.0`:

- **Python floor.** `mlx-lm` 0.32.2 (latest) requires Python ≥ 3.11, which is unsatisfiable
  against `>=3.10`. Hence A1.
- **What B1 will lock.** `mlx-audio[stt]>=0.4.4,<0.6` + `mlx-lm>=0.32.2` at 3.11 resolves to
  mlx 0.32.4, mlx-audio 0.5.9, mlx-lm 0.32.2, transformers 5.23.0, sentencepiece 0.2.3. The lock
  today holds mlx 0.32.2, mlx-audio 0.5.4, transformers 5.17.0.
- **mlx-audio 0.6.x.** Without the ceiling it resolves to 0.6.6. That stays out of scope unless
  A2's diff shows 0.6 is needed.
- **Architecture support.** `mlx-lm` 0.32.2 ships `models/cohere.py` and `models/cohere2.py`,
  which covers both candidates. `cohere2.make_cache` builds a `KVCache` on every
  `sliding_window_pattern`-th layer (4) and a `RotatingKVCache(max_size=sliding_window, keep=0)`
  on the rest.

## Phase 0: spike and measure (B2)

Scratchpad only. Third-party packages go into a scratch venv, with `HF_HOME` pointed into the
scratchpad for anything not already cached.

**Transcripts.** Use real app output: synthesize a scripted meeting with `say -v Samantha` (state
the voice), run it through `load_asr(DEFAULT_REPO).generate(...)` with VAD and punctuation on, and
tile it to length.

**Memory.** Measure `phys_footprint` lifetime max. RSS undercounts MLX.

1. **Restore is exact.** Prefill a 6K-token transcript, so the rotating layers have wrapped.
   - Snapshot, score a question, restore, then score a second question.
   - Compare against the second question on a fresh prefill: logits must match within bf16
     tolerance (the sibling saw P(yes) within 0.008 for cached vs fresh), with argmax identical.
   - Repeat after a full summary generation.
   - Record the max |Δlogit|. **If this fails, stop and rethink before B3.**
2. **Tokenizer swap is sound.** The converted weights with the upstream tokenizer give lower
   perplexity than with the converted tokenizer, measured on a few hundred tokens of the
   transcript. This confirms the upstream one is what the weights expect. For Aya Expanse (after
   A0's access step), repeat the upstream-vs-converted encoding comparison and decide its
   tokenizer source the same way.
3. **Context limit.** Find the largest transcript that fits under 8K together with the preamble,
   system text, the longest task and the summary's output budget. This sets
   `MAX_ANALYSIS_TOKENS` (D1). Check the summary's quality at the cap against one at 2K, since
   quality near the trained limit is not guaranteed.
4. **Memory with both models resident**, at 10 min, 30 min and the cap.
   - **When:** after transcription, after the LLM loads, and at the analysis peak.
   - **Variables:** with and without A3's cache limit, and with prefill through the full model vs
     `model.model(...)` without the LM head.
   - **Sizes:** chunk 512 / 1024 / 2048.
   - **Prior:** the sibling peaked at 7.71 GB at 32 min, but with a 0.88 GB ASR model. Here the ASR
     model is ~4 GB of bf16 weights.
5. **Eviction alternative.** The same series with `load_asr.clear()` before the LLM loads, plus
   the warm reload cost on the next Transcribe. This decides D2.
6. **Speed.** Prefill tok/s at 1K / 4K / 7.5K. Generation tok/s. Full analysis (prefill + summary +
   9 questions) at 30 min. Per-question cost.
7. **Stop tokens.** Which of ids 6, 261001 and 3 the model actually emits at the end of a summary,
   and whether `<|END_RESPONSE|>` precedes `<|END_OF_TURN_TOKEN|>`.
8. **Yes+No mass and summary parse rate per language.**
   - **Per language:** for each of the 14 `LANGUAGES`, take one transcript, synthesized with a
     `say` voice for that language where one exists (note any gaps).
   - **Record:** first-try JSON parse, which language the summary comes back in, and min Yes+No
     mass over the nine default questions.
   - **Then:** check the questions asked in English against a non-English transcript, and the
     questions translated.
9. **Safety preamble side effects.** Whether the default preamble makes the model refuse or hedge
   on transcripts that discuss violence, health or politics, as meetings sometimes do. Use three
   short scripted transcripts. Record any refusal text.
10. **Load failures.** Exact exception type and text for each of three cases:
    - offline with nothing cached;
    - a full disk, using a small disk image as `HF_HOME`;
    - the tokenizer repo's terms not accepted. Expect `GatedRepoError`, which maps onto the
      existing `ModelAccessError`.
11. **Concurrency.** Two analyses at once on one model, with and without A4's lock.

### Phase 0 results

_(fill in during B2)_

### Decisions (end of Phase 0)

Write each decision here with its measured reason.

- **D0. Model.** Tiny Aya Global unless item 1 or 2 fails or B6 shows Aya Expanse clearly better.
  Aya Expanse costs ~1 GB more and has no sliding window to work around. Its 8K context is the
  same.
- **D1. `MAX_ANALYSIS_TOKENS`**, from item 3. Over it, refuse with the exact count. Never
  truncate.
- **D2. Keep both models resident, or evict the ASR model before analysis**, from items 4–5. Keep
  `max_entries=1` on both loaders either way.
- **D3. Which transcript languages may be analysed.** Tiny Aya's card lists all 14 `LANGUAGES`, but
  only English is evaluated (AMI). Recommendation: allow every language passing item 8 (parse and
  mass ≥ `MIN_YES_NO_MASS`), and caption the card "not evaluated in <language>" outside English.
- **D4. Length pre-check unit**, words or characters per language, from item 3's token counts. The
  pre-check only has to refuse before a 4 GB download. The exact refusal happens after
  tokenisation.
- **D5. Question and summary language**, from item 8. Either keep questions and prompts in English
  and ask for the summary in the transcript's language, or translate the default questions too.
- **D6. Where the analysis card goes** (decided by screenshot in B5; see UI).

## Phase 1: summary card (B3)

New module `utils/analysis.py`. It follows `utils/models.py`'s rule: `mlx_lm`, `mlx.core` and
`huggingface_hub` are imported *inside* the functions that need them. The pure functions
(`parse_summary`, `clean_questions`, `yes_probability`'s arithmetic, `analysis_markdown`, the
pre-check) must import without mlx, so `tests/test_pure.py` stays runnable on ubuntu.

**Constants:**

- `WEIGHTS_REPO` / `WEIGHTS_REVISION` and `TOKENIZER_REPO` / `TOKENIZER_REVISION` (Model).
- `MAX_ANALYSIS_TOKENS` (D1).
- `ANALYSIS_MAX_NEW_TOKENS = 512`.
- `ANALYSIS_PREFILL_TOKENS` (from Phase 0 item 4; the sibling's 2048 may be too much with this
  vocabulary).
- `ANALYSIS_PROGRESS_TOKENS = 32`.
- `STOP_IDS` (item 7).

**Prompt layout** (from "What changes", item 1):

- **System turn.** Fixed instructions only: "You are a careful assistant that analyses transcripts
  of recorded speech." No transcript text.
- **User turn.** `<transcript>\n{transcript}\n</transcript>` + `\n\n` + task. The description line
  goes *inside the task*, never in the cached prefix, so a change to it doesn't invalidate the
  cache. It says whether the transcript is punctuated and in which language, and that it is ASR
  output that may contain misheard words. This needs a new `Transcript.punctuation` field. Mind
  the `Transcript` docstring: a field read at render time raises `AttributeError` in sessions
  open across a file-watcher save, a dev-only reload.
- **Cached prefix.** Everything up to and including `</transcript>`. `prefill_transcript` derives
  it by rendering the template with a sentinel task and cutting at the sentinel. A test asserts
  that every real task's ids start with it. If a template ever breaks that, `_task_cache` falls
  back to a fresh full prompt with a `RuntimeWarning` carrying **no transcript text**.
- **Delimiter safety.** Strip or neutralise any literal `</transcript>` and any special-token text
  (`<|…|>`, `<BOS_TOKEN>`, `<EOS_TOKEN>`) in the transcript *and* in questions before rendering.
  The tokenizer would otherwise turn them into real control tokens. The sibling only guarded
  questions, for `<|`.

**Functions:**

- `load_llm()`: `@st.cache_resource(max_entries=1, show_spinner=False)`. The three-call load
  (Model), then the startup tokenizer check.
  - **Gated:** `GatedRepoError` → `ModelAccessError` with the tokenizer repo's URL, matched by type
    like `load_asr`.
  - **Offline:** `LocalEntryNotFoundError` → a readable `RuntimeError`.
  - **Disk full:** `ENOSPC` by errno, *or* the exact OS phrases `No space left on device` / `os
    error 28` in `hf_xet`'s bare `RuntimeError`. Never a bare number: this repo once routed
    disk-full to "sign in" because a shard name contained `401`.
  - **Anything else** propagates.
- `check_analysis_length(transcript, language)`: the cheap pre-check (D4), before the model loads.
- `prefill_transcript(model, tokenizer, transcript)` → `TranscriptPrefill(cache, ids)`.
  - **Chunks:** `ANALYSIS_PREFILL_TOKENS` at a time, calling `on_progress(i, n, label)` before
    each.
  - **Over the cap:** raises with the exact count.
  - **Hand-written loop,** not `generate_step`'s `prompt_progress_callback`: `generate_step` holds
    back the last prompt token, so it cannot build a prefix-only cache.
- `_snapshot(cache)` / `_restore(cache, snap)`: every layer's `state` and `meta_state`. Every task
  runs inside `snap = _snapshot(c); try: … finally: _restore(c, snap)`. It replaces the sibling's
  `trim_prompt_cache` and `offset` arithmetic.
- `summarize_transcript`: greedy `generate_step` from the shared cache, stopping on `STOP_IDS`.
  - **Progress:** every 32 tokens.
  - **Out of tokens:** a "cut off" `RuntimeError` with no retry.
  - **Unparseable JSON:** one retry, showing the bad reply back, then `RuntimeError`.
- `parse_summary`: outermost braces, `json.loads` only, key and type validation. Raises
  `ValueError` / `TypeError`. Never `eval`, `ast.literal_eval` or YAML.
- `Summary` TypedDict: `topic`, `summary`, `topics`, `action_items`.

**Prompts.** Start from the sibling's `SUMMARY_PROMPT`, `SUMMARY_RETRY_PROMPT` and the `anyone`
`QUESTION_PROMPT`. Drop "lowercase, unpunctuated". Re-choose wording on B6's tuning half only. Any
later change is a change to what was scored, so re-run B6.

## Phase 2: yes/no question board (B4)

- `DEFAULT_QUESTIONS`: the nine the AMI evaluation labelled. A test keeps them equal to
  `evals/labels.json`'s. `MAX_QUESTIONS = 20`, `MAX_QUESTION_CHARS = 200`.
- `clean_questions(text)` → `CleanedQuestions(questions, too_long, special, over_limit)`. One per
  line; blanks and repeats are dropped. `special` covers Cohere's control-token spellings (Phase 1,
  delimiter safety).
- `_yes_no_ids`: `Yes` / `No` must each be one token, else `RuntimeError`.
- `yes_probability(yes_logit, no_logit)` → `(P(yes), mass)`: a softmax over the two logits alone,
  plus the mass as a diagnostic. The arithmetic must take plain floats, so `test_pure.py` can test
  it.
- `score_question`: one forward pass over the task tokens from the snapshot, then restore. Nothing
  is generated.
- `answer_questions` → `list[(question, P(yes))]` in the user's order. A `RuntimeWarning` names the
  question **number** and the mass when the mass is under `MIN_YES_NO_MASS` (the sibling's 0.98;
  re-check against Phase 0 item 8).
- `run_analysis(transcript, questions, want_summary, on_progress)`: one prefill, then the summary
  and/or the board.

Scores rank; they are not verdicts. The sibling's held-out evaluation had 26–29% of answers under
0.2 labelled yes. The board caption reads: "Sorted by P(yes), highest first. Scores rank the
questions; they aren't verdicts."

## UI (B5)

**Sidebar**, a new "Analysis" section under "Transcription":

- **"Transcript analysis" toggle**, off by default, `key=` + `bind="query-params"` like the three
  primary controls. Its help text says the first use downloads ~3.6 GB of non-commercial
  (CC-BY-NC) Cohere Labs weights and needs the Tiny Aya terms accepted on Hugging Face.
- **Questions** `st.text_area`, one per line, defaulting to `DEFAULT_QUESTIONS`. Not bound to the
  URL, so edits last for the session only.
- **Skipped-question captions** from `CleanedQuestions`.

**Input column** (left):

- **Analyse button.** Appears when a transcript exists and analysis is on but no analysis matches
  the current questions. It never re-transcribes.
- **Transcribe with analysis on** runs both in the existing `st.status` block as separate steps.
  Elapsed stays transcription-only, since Speed is RTFx.
- **Download caption.** While `llm_is_cached()` is false, a caption beside the button says the
  first analysis downloads ~3.6 GB.
- **Length refusal.** Over the cap, a caption under the button gives the transcript's length, the
  limit, and "about 40 minutes of speech" in place of the button.

**D6, where the card goes.** The reading column is the transcript alone, level with the title
(CLAUDE.md, the `layout="wide"` decision). Try the options in this order:

1. **Left column, below the downloads.** It keeps the reading column pure, and the sibling's board
   worked at 322–421 px. This is the recommendation.
2. **Right column, above the transcript.** It pushes the transcript below the fold.
3. **`st.tabs(["Transcript", "Analysis"])` in the right column.** Both tabs ship on every rerun.

**Card contents:**

- **Header row** with a "Download" button for `<stem>_analysis.md` (`on_click="ignore"`).
- **Body.** Topic, summary, key topics and action items, all **through `st.text`**. "None." for
  empty lists.
- **Board.** One `st.progress` per question, number first, sorted, caption above.
- **Language caption.** "Not evaluated in <language>" when the language is not English (D3).

**State:**

- **Session key.** `analysis` = `{summary, board, questions_key}`, tied to the `Transcript`.
- **Invalidation.** New bytes drop it with `result`, under the existing digest rule. Editing
  questions invalidates the board only.
- **Lifetime.** The browser session, by design. Nothing is written to disk.

**Errors.** The analysis has its own error boundary, so a failure never takes the transcript off
screen. Error branches pass `expanded=True` to `status.update`.

**Streamlit floor.** Re-check any new `st.*` call against clean 1.61 installs, per Conventions.

## Tests

The three-layer rule (CLAUDE.md, Testing) holds.

**`tests/test_pure.py`** (ubuntu, no model, no mlx). Each item below gets tests:

- `parse_summary`: fenced, prose-wrapped, wrong keys and wrong types.
- `clean_questions`: the full truth table, Cohere control-token spellings included.
- `yes_probability`.
- The transcript sanitiser: `</transcript>` and special-token text neutralised.
- `analysis_markdown` **byte for byte**, with and without a board, every field escaped.
- `sorted_board` stability, and the length pre-check at its boundary.
- `DEFAULT_QUESTIONS` against `evals/labels.json`.
- Both revisions are full 40-hex hashes.
- The task description reflecting `Transcript.punctuation` and the language.

**`tests/test_llm_contract.py`** (new; macOS only, on the `check` job). It uses the real installed
`mlx_lm`, **random weights and no download**: a 2-layer `mlx_lm.models.cohere2.Model` built from a
`ModelArgs` with **`sliding_window=8`, `sliding_window_pattern=2`**, so a 30-token prefix wraps the
rotating cache in CI.

- **What it asserts:**
  - the cached prefix plus a task matches the full prompt, within tolerance;
  - after a real `generate_step` summary and a restore, the next question's logits match a fresh
    prefill;
  - question B after question A and its restore equals B fresh;
  - the board after a summary equals the board fresh;
  - `RotatingKVCache.is_trimmable()` is `False` once wrapped. If that ever turns `True`, the reason
    for the snapshot design has gone and someone should notice;
  - `load_model`'s `strict` defaults to `True`.
- **The real tokenizer.** It is gated, so it is never fetched in CI. Its facts (the `We'll` split,
  `Yes` / `No` single tokens, the prefix boundary stable on the 24 cases) are asserted in
  `verify_transcription.py`'s analysis check instead.
- **Why it is allowed here.** It is not the mocked-`generate` test this repo forbids: nothing is
  scripted, so it tests cache arithmetic, not output quality.
- **How it is selected.** `pytest.mark.mlx` on the file. Run `-m "not mlx"` on ubuntu and `-m mlx`
  on macOS. Use the marker, not `importorskip`: a skipped row reads as a passing one.

**`tests/test_smoke.py`** (AppTest, never presses Transcribe or Analyse). Extend `_seeded()` with a
finished analysis. Then check:

- **Model text renders as plain text.** A summary carrying `![x](http://example.invalid/i)` and
  `[x](http://example.invalid/l)` comes back verbatim in `st.text`, and no card markdown element
  contains `](`.
- **Download button.** `dl_md` exists with `on_click="ignore"`.
- **Bound toggle.** The toggle defaults off and round-trips through the URL.
- **Board order.** The board renders sorted.
- **Question edits.** Editing questions keeps the summary and drops the board.
- **New audio.** New bytes drop the analysis.
- **Non-English transcript.** It shows the not-evaluated caption.
- **Over-cap transcript.** It shows the refusal instead of the button.

**`verify_transcription.py`** (integration, real weights). Add `check_analysis` after the
transcriptions, outside their `try`, like `check_vad_backend`. It runs in three steps:

1. **Tokenizer facts.** Assert those listed under the contract tests above, on the real tokenizer.
2. **Known-answer exchange.** Synthesize a short scripted exchange that contains a date, a price
   and a commitment, and lacks a person's name and a place. Assert:
   - P(yes) ranks the present ones above the absent ones;
   - the summary parses;
   - after the restore, the next score equals a fresh one within tolerance.
3. **French.** Run the same exchange in French with `say -v Thomas`, if that voice is installed.

It adds a ~3.6 GB download and a gated repo to the `integration` job. The repo's `HF_TOKEN` account
must have accepted the Tiny Aya terms; say so in the job's comment.

**Real-weights rule.** No unit test loads Tiny Aya. `verify_transcription.py` and `evals/` are the
only places real LLM weights run.

## Evaluation (B6)

Port the sibling's `evals/` (`prepare_ami.py`, `ami.py`, `metrics.py`, `analysis.py`,
`run_analysis_eval.py`, `labels.json`) with a `--model tiny-aya|aya-expanse` switch, and move its
pure-part tests into `tests/`.

- **AMI.** AMI Meeting Corpus (CC BY 4.0) via `FluidInference/ami-corpus-mirror`: 20 excerpts ×
  180 s, split by meeting, with a/c as the tuning half and b/d held out. Its 180 labels were
  human-reviewed in the sibling repo. They describe the audio, so they carry over unchanged.
- **Transcripts.** `prepare_ami` transcribes through `decode_to_mono16k` → `load_asr` →
  `generate`, with punctuation and VAD on.
- **Excerpt length.** Every 180 s excerpt fits easily under 8K tokens, so the eval does not
  exercise the sliding-window restore. Phase 0 item 1 and the contract tests do.
- **What to record.** Held-out pooled, macro and within-excerpt ROC-AUC, Brier, Yes+No mass and
  summary parse rate, on human and ASR transcripts, for both candidates. Put them next to the
  sibling's Granite 4.2 3B numbers (human pooled 0.78 / within-excerpt 0.79, TurboCTC ASR 0.76 /
  0.77) for context, not as a bar that must be cleared.
- **Multilingual known-answer set** (new). For each language with a `say` voice, a scripted
  exchange with known yes/no answers to the nine questions. It reports per-language
  within-excerpt AUC and mass. It is small and synthetic, so it supports D3's "allowed" list, not
  quality claims.
- **How to run it.** By hand, as modules from the repo root, never in CI. Data and results go in
  git-ignored directories. `--rescore` re-scores a saved run without the model.

## Security

Already in place: localhost bind, usage stats off, emoji favicon, strict ASR loading, locked
dependencies, the transcript through `st.text`. New rules:

1. **Model output is plain text, never Markdown.** Audio can carry instructions into the prompt,
   and a rendered Markdown image would make the browser fetch an attacker URL with transcript text
   in it. The `.md` download goes through `escape_markdown`.
2. **The transcript never goes in the system turn.** Tiny Aya's template gives that turn precedence
   over user instructions ("What changes", item 1).
3. **Control-token text is neutralised** in transcripts and questions before rendering.
4. **No code from model repos runs.** `trust_remote_code` stays `False`. The tokenizer download
   pattern list excludes `*.py`. The chat template is Jinja inside transformers' sandbox, and
   `TOKENIZER_REVISION` vouches for it, so changing that pin means reading the template diff.
5. **Model output is data:** `json.loads` only.
6. **No transcript text in logs or warnings.** Numbers and question indexes only.
7. **Review every package B1 adds to `uv.lock`.**
8. **Third-party weights.** `mlx-community` is not Cohere. The pinned `WEIGHTS_REVISION`, strict
   loading and the B6 evaluation are what vouch for the conversion. Never move the pin without
   re-running B6.
9. **When a pin moves, delete the old snapshot** from the HF cache in the same session (`hf cache
   rm … --dry-run` first; measure with `du`).
10. **CLAUDE.md's network sentence becomes:** "the only network calls are Hugging Face downloads —
    the checkpoint, 1.2 MB of Silero the first time VAD is switched on, and ~3.6 GB of Tiny Aya
    weights plus its tokenizer the first time analysis runs."

## Out of scope

- **Chunked analysis over the 8K cap.** This is the first follow-up. A cap of ~40 minutes is much
  lower than the sibling's ~2 hours, and meetings run longer. Sketch: summarise ~6K-token windows,
  then summarise the summaries; for questions, take the max P(yes) across windows. That is a new
  evaluation question, not a port.
- **PyAV decoding** in place of miniaudio + ffmpeg. It needs its own plan.
- **mlx-audio 0.6.x**, unless A2 shows it's needed.
- **Converting Tiny Aya ourselves** (`mlx_lm.convert` from the upstream bf16 weights with the fixed
  tokenizer). It would remove the mixed-repo load, but it needs either hosting a conversion or
  converting on first run (~6.7 GB download and a local write). Revisit if `mlx-community`
  re-converts or Cohere publishes MLX weights itself.
- **Command R7B** (RAG-tuned, same 8K and 4.5 GB as Aya Expanse), **North-Micro-Vision**
  (Apache 2.0, but `cohere_compass` has no `mlx-lm` implementation, and it lacks nl/el/pl/vi), and
  everything 32B and up.
- **Per-segment answers, persistence across sessions, diarization and word timestamps, the demo
  video.**

## Open decisions

- **App name** (C1). The user picks; don't derive it from the repo slug.
- **D0–D6** above.
- **RAM floor for the README**, from Phase 0 items 4–5.
- **Whether CC-BY-NC is acceptable** if the app's use ever becomes commercial. There is no
  commercially licensed Cohere text model in this size class. North-Micro-Vision is Apache 2.0
  but isn't runnable here.

## Risks

- **The snapshot/restore design is new.** The sibling's trim was measured exact. Ours is unproven
  until Phase 0 item 1 and the wrapped-window contract tests pass. That is why both precede app
  code.
- **The tokenizer swap rests on one upstream commit message and an encoding diff.** Phase 0 item 2
  is the measurement that settles it.
- **The 8K cap will refuse real meetings.** The refusal has to be clear, and the chunked follow-up
  should be planned soon after release.
- **Memory on 16 GB machines.** ~4 GB ASR + 3.6 GB LLM + a small KV cache + transient logits.
  Phase 0 measures; D2 and the prefill-without-LM-head trick are the levers.
- **Non-English analysis is barely evaluated.** D3 keeps the app honest about it.
- **Calibration.** P(yes) is a ranking. The UI never thresholds at 0.5.
- **Safety-preamble refusals** on sensitive but legitimate meeting content (Phase 0 item 9).
- **Dependency drift.** B1 moves mlx and transformers under the speech model too, so
  `verify_transcription.py` must be green after B1.
- **Template drift.** The prefix assertion, the startup tokenizer check and the mass warning are the
  tripwires. The two pins keep them quiet.
