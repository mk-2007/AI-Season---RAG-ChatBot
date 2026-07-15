# AI Season RAG Chatbot — Task 3 Backend

Single-file RAG backend implementing **3 chunking methods x 2 retrieval
techniques = 6 combinations**, all run against the same query and printed
side by side, clearly labeled.

## Requirements this satisfies

- ✅ Simple RAG chatbot
- ✅ Uses the AI Season Bootcamp document as the knowledge base
- ✅ Three chunking methods: fixed-size, sentence/paragraph-based, recursive
- ✅ Two retrieval techniques: dense vector similarity, sparse BM25 keyword
- ✅ All 6 outputs displayed simultaneously for a single query
- ✅ Each output clearly labeled with its chunking method + retrieval technique

## Setup

You likely already have most packages installed from the earlier project.
In your existing venv:

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and add your real Groq key:
```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_real_key_here
```

Make sure `data/aiseason-document.txt` exists (it's included here — this
is the knowledge base file).

## Run it — command line version

```powershell
python rag_chatbot.py
```

## Run it — frontend (recommended)

```powershell
streamlit run app.py
```

Opens in your browser. The frontend imports all pipeline logic straight
from `rag_chatbot.py` — there is no duplicated logic between the two
files. It gives you:

- A **sidebar status panel**: mock/live mode, embedding model, current
  chunk sizes, and a "Rebuild knowledge base" button (use this after
  editing the source document — it wipes and rebuilds all 6 indexes).
- A **quick-glance comparison table** at the top of every query: method,
  retrieval technique, number of chunks retrieved, and an answer preview,
  all in one scannable row per pipeline.
- **Detailed cards below**, grouped by chunking method, with dense vs.
  sparse retrieval shown side by side so you can directly compare how
  the SAME chunk set behaves under two different retrieval techniques.
- Each card has an expandable "Retrieved N chunk(s)" section showing
  exactly what was fed to the LLM as context — useful for explaining
  *why* an answer came out the way it did.
- A running history of every query you've asked in the session, most
  recent on top.

```
Your question: What is the price of AI Season and who teaches it?
```

You'll see all 6 labeled outputs printed together, e.g.:

```
[chunking=FIXED      |  retrieval=DENSE]
[chunking=FIXED      |  retrieval=SPARSE]
[chunking=SENTENCE   |  retrieval=DENSE]
[chunking=SENTENCE   |  retrieval=SPARSE]
[chunking=RECURSIVE  |  retrieval=DENSE]
[chunking=RECURSIVE  |  retrieval=SPARSE]
```

Each block shows the retrieved chunk previews AND the generated answer,
so you can visually compare how chunking + retrieval choice changes what
gets surfaced and how the final answer reads.

Type `exit` to quit.

## What to look for when testing (useful for your report)

Try these queries and compare the 6 outputs:

- `"What is the price of AI Season?"` — a specific number; watch whether
  SPARSE (BM25) retrieval surfaces it more reliably than DENSE across
  chunking methods, since BM25 is strong on exact numeric/keyword matches.
- `"Who is the founder and what is his background?"` — a descriptive,
  paraphrasable question; DENSE retrieval tends to do better here since
  it matches on meaning, not exact wording.
- `"What is AI Season's mission statement?"` — appears verbatim multiple
  times in the document; compare whether FIXED-size chunking splits the
  quote awkwardly across two chunks versus SENTENCE/RECURSIVE keeping it
  intact.
- Something NOT in the document (e.g. `"Does AI Season offer a scholarship
  for international students?"`) — check that all 6 pipelines correctly
  say "I don't have enough information," rather than hallucinating.

## Notes on design decisions (for your documentation)

- Each chunking method gets its **own** vector collection and its own
  BM25 index — chunk sets are never mixed between methods.
- The script rebuilds all indexes fresh on every run (no leftover/stale
  chunks), which keeps behavior predictable for repeated testing/grading.
- Mock mode (`LLM_PROVIDER=mock`) exists so retrieval alone can be
  verified without spending API calls or requiring a key — useful while
  you're debugging chunking behavior specifically.
