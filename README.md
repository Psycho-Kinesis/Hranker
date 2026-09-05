# Exam-Prep Doubt-Solving Assistant (RAG)

A retrieval-augmented generation (RAG) system that answers student doubts on
competitive-exam topics (Quant, Reasoning, General Studies), grounded strictly
in a curated knowledge base — built as a demo of the kind of doubt-resolution
tooling a B2B edtech platform (e.g. one that supplies technology to coaching
institutes) could offer its partner institutes to reduce repetitive-doubt
load on faculty.

![The assistant answering a question, with the retrieved source notes and their similarity scores expanded below the answer](demo.png)

*Shown in offline fallback mode (no API key set). The "Retrieved source notes"
panel is the point: every answer can be checked against the exact notes it
came from, and their similarity scores.*

## Why this project
Coaching institutes get flooded with repetitive conceptual doubts. A grounded
RAG assistant can resolve the common ones instantly, cite the exact note it
used (so faculty can verify/trust it), and say when the knowledge base doesn't
cover a question — instead of hallucinating an answer, which is the single
biggest risk in an exam-prep context.

### Knowing when it doesn't know

This turned out to be the hardest part, and it is worth being precise about.

Asked "solve x + y = 10 and 2x = 8" — algebra, which the corpus does not
cover — the assistant confidently returned a note about percentages, scoring
**0.989**, higher than any of the twelve labelled eval questions. The cause is
not a missing threshold. The vectoriser drops single characters and stopwords,
so `x`, `y`, `+`, `=`, `2x`, `8` and `solve` all disappeared and the only
surviving term was `10`, which matched "decreases the new price by 10%". L2
normalisation then erases the difference: a query backed by one incidental
number and a query backed by five strong terms both become unit vectors, so
cosine similarity cannot express *how much of the question was understood*.

Counting matched terms doesn't separate these either — the legitimate question
"How do I quickly convert 1/7 into a percentage?" also matches exactly one.
Nor does weighting by rarity: `10` has an IDF of **3.71**, the highest of any
term in this corpus, while `percentage` has 2.46, so an IDF rule would reject
the good question and accept the broken one.

What does separate them is **whether the evidence is a number**. A bare number
turns up incidentally in notes on any subject, so matching one says nothing
about topic. So the system refuses when either holds:

- **The top score is below `MIN_USEFUL_SCORE` (0.35)** — no vocabulary overlap
  at all. Measured, not guessed: the labelled questions score 0.697 at worst,
  genuinely unrelated ones score 0.000.
- **Every matched term is numeric** — the algebra question's sole evidence was
  `10`. This fires only when numbers are the *whole* of the match, so "Article
  32 meaning" is answered normally: it matches `article` as well as `32`.

Measured on 12 labelled in-corpus questions and 8 out-of-corpus ones: **0
wrongly refused, 0 wrongly answered.**

A question that matches a single non-numeric term is still answered, but the
UI names that term above the answer rather than below it, so a thin match is
visible before the note is trusted.

Refusal in LLM mode is separate and stronger: the system prompt instructs the
model to say when the notes are insufficient. That instruction never runs
without an API key, which is exactly why the offline path needed its own
guard — the deployed demo runs in fallback mode by default.

## Architecture

```
Student question
      │
      ▼
 ┌─────────────┐      ┌──────────────────┐      ┌─────────────────┐
 │  Embed query │ ───▶ │  Retrieve top-k   │ ───▶ │  Generate answer │
 │ (same backend│      │  chunks by cosine │      │  grounded in     │
 │ as ingestion)│      │  similarity       │      │  retrieved text  │
 └─────────────┘      └──────────────────┘      └─────────────────┘
                              ▲                          │
                              │                          ▼
                      data/processed/            Answer + cited
                      (vectors + chunks)          source sections
```

- **Ingestion** (`src/ingest.py`): reads markdown notes from `data/raw/`,
  chunks each doc by its `##` sections (keeping section titles as metadata
  for citation), embeds every chunk, and persists the index.
- **Embeddings** (`src/embeddings.py`): two swappable backends behind one
  interface —
  - `TfidfEmbedder` (default): TF-IDF + truncated SVD, runs fully offline,
    no model download required. This is what `python src/ingest.py` builds
    the index with unless you change the backend.
  - `SentenceTransformerEmbedder`: real dense semantic embeddings via
    `sentence-transformers`, a one-line swap for when you have full
    internet access — recommended upgrade path, since it captures meaning
    rather than keyword overlap.
- **Retrieval** (`src/retrieve.py`): embeds the query with the same fitted
  backend and ranks chunks by cosine similarity.
- **Generation** (`src/generate.py`): builds a prompt instructing the model to
  answer *only* from the retrieved notes and to say so if they don't cover the
  question. Gemini and Claude are both supported; whichever key is present
  decides, so switching provider is a secrets change rather than a code
  change. The grounding prompt, citation format and refusal checks sit outside
  the provider call and are identical either way. With no key at all, it falls
  back to returning the top-matching note (clearly labeled), so the pipeline
  stays runnable and demoable.
- **UI** (`app.py`): a Streamlit chat-style front-end with an expandable
  "sources" panel — this is what you'd actually screen-share in an
  interview.

## Measured results (not placeholders)
Retrieval quality was evaluated on 12 hand-labeled questions spanning all 6
knowledge-base docs (`eval/eval_questions.json`), checking whether the
correct source document appears in the top-3 retrieved chunks:

```
Hit rate@3:              12/12 = 100.0%
Top-1 accuracy:          12/12 = 100.0%
Mean separation margin:  +0.815
Min separation margin:   +0.579
```
(Reproduce with `python eval/evaluate_retrieval.py`. Backend: TF-IDF + SVD.)

**Honest caveat**: this is a small, clean, 6-document corpus built to prove
the pipeline works end-to-end — 100% is expected here, not a claim of
state-of-the-art retrieval. The six topics share almost no vocabulary, so
even lexical matching separates them cleanly.

**Why the third metric exists.** Hit-rate and top-1 accuracy are both pinned
at 100%, which means neither can tell two retrievers apart — a saturated
metric carries no information, so "TF-IDF scores 100%" and "dense embeddings
score 100%" is not a comparison. The separation margin is the score gap
between the best chunk from the correct document and the best chunk from any
other document: it measures *how far* the right answer beat the field, not
just whether it did. It has headroom, so it can actually rank two backends,
and a shrinking margin is the early warning that retrieval is about to start
failing. The closest call here is "Which Indian rivers flow west into the
Arabian Sea?" at +0.579 — comfortable, but the narrowest of the twelve.

### Comparing embedding backends

Swapping backends is a command, not a code edit:

```bash
pip install sentence-transformers
python src/ingest.py sentence-transformers   # or: RAG_BACKEND=sentence-transformers python src/ingest.py
python eval/evaluate_retrieval.py
```

Because the fitted embedder is persisted next to the vectors, retrieval picks
up the new backend with no other change — that is the payoff of the shared
interface.

The dense-backend numbers are deliberately **not** quoted here: the
`all-MiniLM-L6-v2` weights could not be downloaded in the environment this
was built in, so that run has not been made, and reporting a number nobody
measured is exactly what the eval exists to avoid. Run the three commands
above to fill it in. The honest prior is that hit-rate stays at 100% (there
is nowhere to go) and the margin is where any difference will show.

## Running it

Requires **Python 3.10+** (the `anthropic` 1.x SDK dropped 3.9).

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Build the index from data/raw/*.md
python src/ingest.py

# 2. Try it from the CLI
python src/rag_pipeline.py "What is the successive percentage change formula?"

# 3. Check retrieval quality
python eval/evaluate_retrieval.py

# 4. Launch the demo UI
export GEMINI_API_KEY=your_key_here      # optional — or ANTHROPIC_API_KEY; omit for offline fallback
streamlit run app.py
```

A free Gemini key from [Google AI Studio](https://aistudio.google.com/apikey)
is enough to run generation. `GEMINI_MODEL` overrides the model if the default
isn't available on your key.

`data/processed/` holds generated artifacts (vectors, chunk metadata, and the
pickled fitted embedder) and is gitignored — step 1 regenerates it in about a
second. Rebuild it rather than reusing an index pickled by a different
scikit-learn version, or unpickling warns about version skew.

## Deployment

On Streamlit Community Cloud, point the app at `app.py` and add
`GEMINI_API_KEY` (or `ANTHROPIC_API_KEY`) under **Settings → Secrets**. Both
are optional — without either, the app runs in offline fallback mode.

Because `data/processed/` is gitignored, a fresh deploy starts with no index.
`app.py` detects that and builds one on first run, so deployment stays a
single step. The alternative — committing the index — would work at this
corpus size but doesn't generalise, and it would mean committing a pickle.

## What I'd build next
- Run the dense-embedding comparison above and record both backends' margins
  — the switch and the metric are in place, only the measurement is missing.
- Grow the corpus to 50-100+ notes with deliberate topical overlap (simple
  interest vs compound interest vs percentages all share vocabulary). That is
  what makes the eval discriminating rather than saturated, and it is where
  the dense backend should start to earn its keep.
- Add a re-ranking step (cross-encoder) for cases where top-k retrieval is
  noisy on a larger corpus.
- Track "couldn't answer" queries as a feed for identifying content gaps in
  the institute's knowledge base — turns the assistant into a content-gap
  analytics tool too, not just a doubt-solver.
- Add per-answer feedback (thumbs up/down) to build a labeled dataset for
  evaluating and eventually fine-tuning retrieval.

## Project structure
```
Hranker/
├── data/
│   ├── raw/            # source knowledge-base notes (markdown)
│   └── processed/       # generated: vectors.npy, chunks.json, embedder.pkl
├── src/
│   ├── embeddings.py    # TF-IDF / sentence-transformers backends
│   ├── ingest.py        # chunking + index building
│   ├── retrieve.py       # similarity search
│   ├── generate.py        # grounded LLM answer generation + offline fallback
│   └── rag_pipeline.py     # ties retrieval + generation together
├── eval/
│   ├── eval_questions.json      # labeled test questions
│   └── evaluate_retrieval.py    # hit-rate@k evaluation
├── app.py                # Streamlit demo UI
├── requirements.txt
└── README.md
```
