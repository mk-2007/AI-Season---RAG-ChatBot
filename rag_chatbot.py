"""
rag_chatbot.py
==================================================================
AI SEASON — TASK 3: "Build and Compare a RAG Chatbot"
==================================================================

WHAT THIS FILE DOES (single file, both pipelines, as required):

  INGESTION PIPELINE (runs once at startup):
    1. Load the AI Season knowledge-base document (data/aiseason-document.txt)
    2. Split it into chunks using THREE different chunking methods
    3. Index each of the 3 chunk sets TWICE — once for dense (vector)
       retrieval, once for sparse (BM25/keyword) retrieval
       -> 3 chunking methods x 2 retrieval techniques = 6 independent
          retrieval pipelines, built once and reused for every query.

  RETRIEVAL + GENERATION PIPELINE (runs per user query):
    1. Take one user query.
    2. Run it through ALL 6 pipelines (fixed+dense, fixed+sparse,
       sentence+dense, sentence+sparse, recursive+dense, recursive+sparse).
    3. For each pipeline: retrieve top-k chunks -> build a grounded prompt
       -> call the LLM (Groq, or mock mode if no API key) -> get an answer.
    4. Print all 6 answers together, clearly labeled, so they can be
       compared side by side for the SAME query.

WHY THREE CHUNKING METHODS ARE IMPLEMENTED DIFFERENTLY (not just different
chunk_size numbers on the same splitter -- that would not be a real
comparison):

  A. FIXED-SIZE CHUNKING
     Cuts the raw text every N characters, no awareness of sentences,
     paragraphs, or structure. Fast, simple, but WILL slice a sentence or
     fact in half at chunk boundaries. This is the "naive baseline."

  B. SENTENCE/PARAGRAPH-BASED CHUNKING
     Splits on sentence boundaries first, then greedily groups whole
     sentences together until a target chunk size is reached. A chunk
     NEVER ends mid-sentence. This preserves local grammatical/semantic
     completeness.

  C. RECURSIVE (STRUCTURE-AWARE) CHUNKING
     Uses LangChain's RecursiveCharacterTextSplitter, which tries to split
     on paragraph breaks first, then lines, then sentences, then words --
     falling back to hard cuts only as a last resort. This produces the
     most structurally coherent chunks of the three, generally respecting
     this document's own SECTION/heading structure.

WHY TWO RETRIEVAL TECHNIQUES:

  A. DENSE (VECTOR / SEMANTIC) RETRIEVAL
     Embeds the query and every chunk into vectors (HuggingFace
     sentence-transformers) and finds chunks that are semantically close
     in meaning, even if they don't share exact words with the query.

  B. SPARSE (BM25 / KEYWORD) RETRIEVAL
     A statistical keyword-overlap ranking algorithm (no neural network,
     no "meaning" -- pure term-frequency statistics). Excellent at exact
     term matches (names, numbers, prices, dates) that a general-purpose
     embedding model may under-weight.

Running side by side lets a grader (or you) SEE cases where dense wins
(paraphrased questions), sparse wins (exact facts/numbers), and where
chunking method changes which specific fact gets surfaced at all.
==================================================================
"""

import os
import re
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_FILE = os.getenv("DATA_FILE", "data/aiseason-document.txt")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

FIXED_CHUNK_SIZE = int(os.getenv("FIXED_CHUNK_SIZE", 500))
SENTENCE_CHUNK_SIZE = int(os.getenv("SENTENCE_CHUNK_SIZE", 500))
RECURSIVE_CHUNK_SIZE = int(os.getenv("RECURSIVE_CHUNK_SIZE", 500))
RECURSIVE_CHUNK_OVERLAP = int(os.getenv("RECURSIVE_CHUNK_OVERLAP", 80))

TOP_K = int(os.getenv("TOP_K", 3))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def is_mock_mode() -> bool:
    return LLM_PROVIDER == "mock" or not GROQ_API_KEY


# ==================================================================
# SECTION 1: LOAD THE DOCUMENT
# ==================================================================

def load_document(path: str = DATA_FILE) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Knowledge base file not found at '{path}'. "
            "Put the AI Season document there (or set DATA_FILE in .env)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ==================================================================
# SECTION 2: THREE CHUNKING METHODS
# ==================================================================

def chunk_fixed_size(text: str, source: str) -> list[Document]:
    """METHOD A: Fixed-size chunking.
    Hard character-count cuts, no structural awareness. This is the
    'naive baseline' every comparison needs."""
    splitter = CharacterTextSplitter(
        separator="",           # no smart separator -> pure character cuts
        chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=0,
    )
    pieces = splitter.split_text(text)
    return [
        Document(page_content=p, metadata={"source": source, "method": "fixed"})
        for p in pieces
    ]


def chunk_sentence_based(text: str, source: str) -> list[Document]:
    """METHOD B: Sentence/paragraph-based chunking.
    Splits the text into sentences first (regex on '.', '?', '!' followed
    by whitespace+capital, plus hard paragraph breaks), then greedily
    packs whole sentences into a chunk until adding the next sentence
    would exceed SENTENCE_CHUNK_SIZE. A chunk NEVER cuts a sentence in
    half -- that's the entire point of this method versus Method A."""
    # Normalize paragraph breaks, then split into sentences within each.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []
    for para in paragraphs:
        # crude but effective sentence splitter for this kind of prose
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", para)
        sentences.extend(s.strip() for s in parts if s.strip())

    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= SENTENCE_CHUNK_SIZE:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)

    return [
        Document(page_content=c, metadata={"source": source, "method": "sentence"})
        for c in chunks
    ]


def chunk_recursive(text: str, source: str) -> list[Document]:
    """METHOD C: Recursive / structure-aware chunking.
    Tries paragraph breaks first, then lines, then sentence boundaries,
    then words -- only falling back to hard character cuts as a last
    resort. Generally produces the most semantically coherent chunks of
    the three, and tends to respect this document's own SECTION
    headings since those are set off by blank lines."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    return [
        Document(page_content=p, metadata={"source": source, "method": "recursive"})
        for p in pieces
    ]


CHUNKING_METHODS = {
    "fixed": chunk_fixed_size,
    "sentence": chunk_sentence_based,
    "recursive": chunk_recursive,
}


# ==================================================================
# SECTION 3: INGESTION — build 3 chunk sets, each indexed 2 ways
# ==================================================================

def build_all_indexes(text: str, source: str) -> dict:
    """Returns a nested dict:
        {
          "fixed":     {"dense": <Chroma retriever>, "sparse": <BM25Retriever>},
          "sentence":  {"dense": ..., "sparse": ...},
          "recursive": {"dense": ..., "sparse": ...},
        }
    Each chunking method gets its OWN vector collection (so chunk sets
    never mix) and its own in-memory BM25 index.

    NOTE ON PERSISTENCE: this script rebuilds all indexes from scratch on
    every run anyway (that's the whole point -- deterministic, reproducible
    results for grading/demo). Since nothing needs to survive between
    runs, we use Chroma in IN-MEMORY mode (no persist_directory) instead
    of writing to disk. This sidesteps a real Windows problem entirely:
    disk-persisted Chroma DBs are SQLite files that can get locked by
    OneDrive sync, antivirus scans, or a lingering process, causing
    PermissionErrors on the next run. An in-memory store has no file to
    lock -- it simply vanishes when the script/app process ends, which is
    exactly the lifecycle we want here.
    """
    print("Loading embedding model (first run downloads it, ~80MB)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    indexes = {}
    for method_name, chunk_fn in CHUNKING_METHODS.items():
        print(f"\n[ingest] Chunking method: {method_name}")
        chunks = chunk_fn(text, source)
        print(f"  -> {len(chunks)} chunks produced "
              f"(avg size {sum(len(c.page_content) for c in chunks)//len(chunks)} chars)")

        # --- Dense (vector) index ---
        # In-memory collection -- fresh every run, nothing written to disk,
        # so there is nothing for another process to lock.
        collection_name = f"ai_season_{method_name}"
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
        )
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

        # --- Sparse (BM25) index ---
        sparse_retriever = BM25Retriever.from_documents(chunks)
        sparse_retriever.k = TOP_K

        indexes[method_name] = {"dense": dense_retriever, "sparse": sparse_retriever}

    return indexes


# ==================================================================
# SECTION 4: GENERATION (retrieve -> prompt -> LLM)
# ==================================================================

PROMPT = ChatPromptTemplate.from_template(
    """You are the AI Season program assistant, speaking directly to a
prospective or current student. Answer the question using ONLY the facts
in the reference material below -- but answer as a knowledgeable person
would, not as a system quoting a source.

Rules:
- NEVER say phrases like "according to the document", "the context
  states", "based on the provided information", "the document says", or
  similar. Just state the fact directly, as if you simply know it.
- Answer ONLY what was asked. Do not append extra unrelated facts, extra
  Q&A pairs, or background trivia that happened to appear near the
  relevant text.
- If the material only covers a RELATED angle rather than the exact
  literal question, give that related answer directly and naturally
  (e.g. if asked about a time-based refund but only a merit-based refund
  exists, just explain the merit-based one -- don't mention that the
  exact thing asked wasn't found).
- If truly nothing relevant exists, say: "I don't have information on
  that." -- nothing more.
- Keep it to 1-3 sentences. Be specific with names, numbers, and dates
  when they're relevant.

Reference material:
{context}

Question: {question}

Answer:"""
)


def get_llm():
    if is_mock_mode():
        return None
    return ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)


def format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


def dedupe_docs(docs: list[Document]) -> list[Document]:
    """Defensive safety net: never let identical chunk text appear twice
    in the same context, regardless of what the retriever returns."""
    seen, unique = set(), []
    for d in docs:
        if d.page_content not in seen:
            seen.add(d.page_content)
            unique.append(d)
    return unique


def extract_token_usage(response) -> dict:
    """Groq (via langchain-groq) reports token usage two possible ways
    depending on library version: a standardized `.usage_metadata`
    attribute (input_tokens/output_tokens/total_tokens), or the older
    `.response_metadata['token_usage']` dict (prompt_tokens/
    completion_tokens/total_tokens). We check both so this keeps working
    regardless of which version is installed."""
    usage = getattr(response, "usage_metadata", None)
    if usage:
        return {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    meta = getattr(response, "response_metadata", {}) or {}
    token_usage = meta.get("token_usage", {})
    if token_usage:
        return {
            "input_tokens": token_usage.get("prompt_tokens"),
            "output_tokens": token_usage.get("completion_tokens"),
            "total_tokens": token_usage.get("total_tokens"),
        }

    return {"input_tokens": None, "output_tokens": None, "total_tokens": None}


def generate_answer(question: str, docs: list[Document], llm) -> dict:
    """Returns a dict: {answer, input_tokens, output_tokens, total_tokens}.
    Token counts are None in mock mode (no LLM call happens)."""
    docs = dedupe_docs(docs)
    empty_usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    if not docs:
        return {"answer": "I don't have information on that.", **empty_usage}

    context = format_context(docs)

    if is_mock_mode():
        answer = "[MOCK] " + docs[0].page_content[:250].strip() + "..."
        return {"answer": answer, **empty_usage}

    chain = PROMPT | llm
    response = chain.invoke({"context": context, "question": question})
    usage = extract_token_usage(response)
    return {"answer": response.content.strip(), **usage}


# ==================================================================
# SECTION 5: RUN ALL 6 COMBINATIONS FOR ONE QUERY
# ==================================================================

def run_all_combinations(question: str, indexes: dict, llm) -> list[dict]:
    """Runs the SAME question through all 3 chunking methods x 2
    retrieval techniques = 6 pipelines, and returns a list of results,
    each clearly labeled with its chunking method and retrieval
    technique, ready for side-by-side display."""
    results = []
    for method_name, retrievers in indexes.items():
        for retrieval_name, retriever in retrievers.items():
            docs = dedupe_docs(retriever.invoke(question))
            gen = generate_answer(question, docs, llm)
            results.append({
                "chunking_method": method_name,
                "retrieval_technique": retrieval_name,
                "num_chunks_retrieved": len(docs),
                "answer": gen["answer"],
                "input_tokens": gen["input_tokens"],
                "output_tokens": gen["output_tokens"],
                "total_tokens": gen["total_tokens"],
                "retrieved_previews": [d.page_content[:120].replace("\n", " ") + "..." for d in docs],
            })
    return results


def print_comparison(question: str, results: list[dict]):
    print("\n" + "=" * 90)
    print(f"QUERY: {question}")
    print("=" * 90)

    for r in results:
        label = f"[chunking={r['chunking_method'].upper()}  |  retrieval={r['retrieval_technique'].upper()}]"
        print("\n" + "-" * 90)
        print(label)
        print("-" * 90)
        print(f"Retrieved {r['num_chunks_retrieved']} chunk(s):")
        for i, preview in enumerate(r["retrieved_previews"], start=1):
            print(f"   [{i}] {preview}")
        print(f"\nANSWER:\n{r['answer']}")
        if r["total_tokens"] is not None:
            print(f"\n[tokens] input={r['input_tokens']}  output={r['output_tokens']}  total={r['total_tokens']}")
        else:
            print("\n[tokens] N/A (mock mode)")

    print("\n" + "=" * 90)
    print("End of comparison — 6 combinations shown above "
          "(3 chunking methods x 2 retrieval techniques).")
    print("=" * 90)


# ==================================================================
# SECTION 6: MAIN — ingest once, then loop on user queries
# ==================================================================

def main():
    print(f"LLM_PROVIDER = {LLM_PROVIDER}  |  mock_mode = {is_mock_mode()}")
    if is_mock_mode():
        print("Running in MOCK MODE — set LLM_PROVIDER=groq and GROQ_API_KEY in .env for real answers.\n")

    text = load_document()
    print(f"Loaded knowledge base: {DATA_FILE} ({len(text)} characters)\n")

    indexes = build_all_indexes(text, source=DATA_FILE)
    llm = get_llm()

    print("\nIngestion complete. All 6 pipelines (3 chunking x 2 retrieval) are ready.")
    print("Type a question about AI Season, or 'exit' to quit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        results = run_all_combinations(question, indexes, llm)
        print_comparison(question, results)


if __name__ == "__main__":
    main()