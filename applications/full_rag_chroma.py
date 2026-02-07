"""
Full RAG (ChromaDB + Hugging Face) with Re-Ranking + Strict JSON Output

Install deps:
  python -m pip install -r requirements.txt

What this app demonstrates (system design):
- Vector retrieval (ChromaDB) finds "roughly relevant" chunks fast via embeddings.
- A re-ranker (cross-encoder) improves quality by re-ordering candidates using a stronger
  question+passage scoring model (slower, but more accurate).
- Strict JSON output makes the system auditable: you can verify the answer is supported
  and see exactly which sources/evidence were used.

Why we do these in the UI:
- Retrieval-only can return near-matches that look relevant but contain wrong/partial facts.
- Re-ranking helps pick the BEST chunks among the Top-N retrieved.
- Strict JSON output prevents the model from rambling; it must produce:
  answer + citations + quoted evidence, or say UNKNOWN.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

APP_NAME = "RAG: Retrieval Augmented Generation"
APP_DESCRIPTION = (
    "Full RAG with ChromaDB vector retrieval + HF re-ranker + strict JSON output "
    "(answer + citations + quoted evidence)."
)

# -----------------------------
# Paths / constants
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = REPO_ROOT / "knowledge_base"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CHROMA_DIR = ARTIFACTS_DIR / "chroma_db"

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_GEN_MODEL = "google/flan-t5-base"  # CPU-friendly instruction-tuned

_WS_RE = re.compile(r"\s+")


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class DocChunk:
    source: str
    chunk_id: int
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    chunk_id: str
    text: str
    distance: float
    rerank_score: Optional[float] = None


# -----------------------------
# Dependency checks
# -----------------------------
def _require_deps() -> None:
    try:
        import chromadb  # noqa: F401
        import sentence_transformers  # noqa: F401
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception as exc:
        st.error("Missing required dependencies for Full RAG.")
        st.write("Install with:")
        st.code("python -m pip install -r requirements.txt")
        st.exception(exc)
        st.stop()


def _device_label() -> str:
    import torch

    if torch.cuda.is_available():
        return f"GPU (cuda:{torch.cuda.current_device()})"
    return "CPU"


# -----------------------------
# Text helpers
# -----------------------------
def _clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _chunk_text_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Transparent word-based chunking (beginner-friendly).
    chunk_size/overlap are in *words*.
    """
    words = _clean_text(text).split()
    if not words:
        return []

    chunks: List[str] = []
    i = 0
    while i < len(words):
        end = min(len(words), i + chunk_size)
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i = max(0, end - overlap)
    return chunks


def _quote_evidence(text: str, max_chars: int = 220) -> str:
    """
    Take a short quote snippet for evidence.
    We keep it short so the JSON stays readable.
    """
    t = _clean_text(text)
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


# -----------------------------
# Load documents from knowledge_base
# -----------------------------
def _load_kb_documents(kb_dir: Path) -> List[Tuple[str, str]]:
    """
    Load .txt/.md files from ./knowledge_base.
    (PDF support can be added later if needed.)
    """
    if not kb_dir.exists():
        return []

    docs: List[Tuple[str, str]] = []
    for p in sorted([x for x in kb_dir.glob("**/*") if x.is_file()]):
        if p.suffix.lower() in {".txt", ".md"}:
            docs.append((p.name, p.read_text(encoding="utf-8", errors="ignore")))
    return docs


def _build_chunks(
    docs: List[Tuple[str, str]], chunk_size: int, overlap: int
) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    for source, content in docs:
        content = _clean_text(content)
        if not content:
            continue
        parts = _chunk_text_words(content, chunk_size=chunk_size, overlap=overlap)
        for j, part in enumerate(parts):
            chunks.append(DocChunk(source=source, chunk_id=j, text=part))
    return chunks


# -----------------------------
# ChromaDB + embeddings
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _get_chroma_collection(collection_name: str):
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(name=collection_name)


def _upsert_chunks(
    *,
    collection,
    chunks: List[DocChunk],
    embedder,
    batch_size: int = 64,
) -> int:
    if not chunks:
        return 0

    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, str]] = []

    for c in chunks:
        cid = f"{c.source}::chunk_{c.chunk_id}"
        ids.append(cid)
        texts.append(c.text)
        metas.append({"source": c.source, "chunk_id": str(c.chunk_id)})

    inserted = 0
    for i in range(0, len(texts), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        batch_metas = metas[i : i + batch_size]
        embs = embedder.encode(batch_texts, show_progress_bar=False).tolist()

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embs,
            metadatas=batch_metas,
        )
        inserted += len(batch_ids)

    return inserted


def _retrieve_topn(
    *,
    collection,
    query: str,
    embedder,
    top_n: int,
) -> List[RetrievedChunk]:
    """
    Fast vector retrieval: return Top-N candidates by embedding similarity.
    Smaller distance is better in Chroma distance output (typically).
    """
    q_emb = embedder.encode([query], show_progress_bar=False).tolist()[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_n,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[RetrievedChunk] = []
    for text, meta, dist in zip(docs, metas, dists):
        out.append(
            RetrievedChunk(
                source=str(meta.get("source", "")),
                chunk_id=str(meta.get("chunk_id", "")),
                text=str(text),
                distance=float(dist),
            )
        )
    return out


# -----------------------------
# Re-ranking (cross-encoder)
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_reranker(model_name: str):
    """
    Cross-encoder scores (question, passage) pairs.
    This is slower than vector retrieval but improves ranking quality.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _rerank(
    *,
    reranker,
    question: str,
    candidates: List[RetrievedChunk],
) -> List[RetrievedChunk]:
    if not candidates:
        return []

    pairs = [(question, c.text) for c in candidates]
    scores = reranker.predict(pairs)

    rescored: List[RetrievedChunk] = []
    for c, s in zip(candidates, scores):
        rescored.append(
            RetrievedChunk(
                source=c.source,
                chunk_id=c.chunk_id,
                text=c.text,
                distance=c.distance,
                rerank_score=float(s),
            )
        )

    # Higher cross-encoder score is better
    rescored.sort(
        key=lambda x: (x.rerank_score if x.rerank_score is not None else -1e9),
        reverse=True,
    )
    return rescored


# -----------------------------
# Generation (HF pipeline)
# -----------------------------
@st.cache_resource(show_spinner=False)
def _load_generator(model_name: str):
    """
    Use an instruction-tuned seq2seq model (FLAN-T5) for better compliance.
    """
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text2text-generation", model=model_name, device=device)


def _build_strict_json_prompt(
    *,
    question: str,
    retrieved: List[RetrievedChunk],
) -> str:
    """
    Force a strict JSON output with citations + quoted evidence.

    Model must output EXACTLY one JSON object. No markdown. No extra text.
    """
    ctx_lines: List[str] = []
    for i, r in enumerate(retrieved, start=1):
        ctx_lines.append(f"[{i}] SOURCE={r.source} CHUNK={r.chunk_id}\n{r.text}")
    context = "\n\n".join(ctx_lines).strip()

    schema = {
        "answer": "string (or 'UNKNOWN' if not supported)",
        "supported_by_context": "boolean",
        "citations": [{"source": "string", "chunk_id": "string"}],
        "quoted_evidence": [
            {"source": "string", "chunk_id": "string", "quote": "string"}
        ],
        "notes": "string (optional; keep short)",
    }

    return (
        "You are a grounded assistant.\n"
        "You MUST answer using ONLY the provided context.\n"
        "If the answer is not clearly supported by the context, you MUST answer 'UNKNOWN'.\n"
        "Return ONLY a single valid JSON object. No markdown. No extra text.\n"
        "Keys must match the schema exactly.\n\n"
        f"JSON schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "JSON:"
    )


def _generate_text(gen, prompt: str, max_new_tokens: int) -> str:
    out = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=4,
    )
    return str(out[0].get("generated_text", "")).strip()


def _try_parse_json(text: str) -> Tuple[Optional[Dict], str]:
    """
    Parse JSON from model output. If the model includes extra text, attempt substring extraction.
    """
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        try:
            return json.loads(t), ""
        except Exception as exc:
            return None, f"JSON parse failed: {exc}"

    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "No JSON object detected in output."

    candidate = t[start : end + 1]
    try:
        return json.loads(candidate), ""
    except Exception as exc:
        return None, f"JSON parse failed: {exc}"


def _honest_fallback_json(
    *,
    raw_output: str,
    retrieved: List[RetrievedChunk],
) -> Dict:
    """
    If the model refuses to output JSON, we do NOT hallucinate structure.
    We return an 'honest' JSON wrapper so the UI stays usable.
    """
    citations = [{"source": r.source, "chunk_id": r.chunk_id} for r in retrieved]
    evidence = [
        {"source": r.source, "chunk_id": r.chunk_id, "quote": _quote_evidence(r.text)}
        for r in retrieved
    ]
    return {
        "answer": raw_output.strip() or "",
        "supported_by_context": False,
        "citations": citations,
        "quoted_evidence": evidence,
        "notes": "Model did not return valid JSON; this is a fallback wrapper.",
    }


# -----------------------------
# Streamlit UI
# -----------------------------
def run() -> None:
    _require_deps()

    st.markdown("### Full RAG (ChromaDB + Hugging Face)")
    st.write(
        "This is a **full RAG** pipeline: documents → embeddings → **ChromaDB** → retrieval → (optional) **re-rank** → generation.\n\n"
        "**Why re-rank?** Vector retrieval is fast but approximate. Re-ranking improves accuracy by choosing the best evidence.\n"
        "**Why strict JSON?** It forces auditability: answer + citations + quoted evidence (or `UNKNOWN`)."
    )

    colA, colB = st.columns([1.4, 1.0])
    with colA:
        collection_name = st.text_input("Chroma collection", value="llm_lab_kb")
    with colB:
        st.text_input("Device", value=_device_label(), disabled=True)

    st.markdown("---")
    st.markdown("### Models")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        embed_model = st.selectbox(
            "Embedding model",
            options=[
                DEFAULT_EMBED_MODEL,
                "sentence-transformers/all-MiniLM-L12-v2",
                "sentence-transformers/paraphrase-MiniLM-L6-v2",
            ],
            index=0,
        )
    with col2:
        rerank_model = st.selectbox(
            "Re-ranker model (cross-encoder)",
            options=[
                DEFAULT_RERANK_MODEL,
                "cross-encoder/ms-marco-TinyBERT-L-2-v2",
            ],
            index=0,
        )
    with col3:
        gen_model = st.selectbox(
            "Generator model",
            options=[
                DEFAULT_GEN_MODEL,
                "google/flan-t5-small",
            ],
            index=0,
        )

    st.markdown("---")
    st.markdown("### Indexing (Documents → Chunks → Embeddings → ChromaDB)")
    docs = _load_kb_documents(KB_DIR)

    with st.expander("Knowledge base status"):
        st.write(f"Knowledge base folder: `{KB_DIR}`")
        st.write(f"Found **{len(docs)}** document(s).")
        if not docs:
            st.info(
                "Create a `knowledge_base/` folder at repo root and add .txt files (e.g., australia.txt)."
            )
        for name, content in docs[:8]:
            st.markdown(f"**{name}**")
            st.write((content[:400] + "…") if len(content) > 400 else content)
            st.markdown("---")

    col4, col5, col6 = st.columns([1, 1, 1])
    with col4:
        chunk_size = int(st.slider("Chunk size (words)", 80, 400, 200, 20))
    with col5:
        overlap = int(st.slider("Overlap (words)", 0, 120, 40, 10))
    with col6:
        batch_size = int(st.slider("Embedding batch size", 8, 128, 64, 8))

    embedder = _load_embedder(embed_model)
    collection = _get_chroma_collection(collection_name)

    colX, colY = st.columns([1, 1])
    with colX:
        build_btn = st.button(
            "📦 Build / Update Index", type="primary", use_container_width=True
        )
    with colY:
        clear_btn = st.button("🧹 Clear Collection (danger)", use_container_width=True)

    if clear_btn:
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            client.delete_collection(name=collection_name)
            st.success("Collection deleted. Rebuild the index to recreate it.")
            st.stop()
        except Exception as exc:
            st.error("Failed to clear collection.")
            st.exception(exc)
            st.stop()

    if build_btn:
        if not docs:
            st.error("No documents found. Add .txt files to ./knowledge_base first.")
            st.stop()

        with st.spinner("Chunking and embedding documents..."):
            chunks = _build_chunks(docs, chunk_size=chunk_size, overlap=overlap)
            t0 = time.perf_counter()
            n = _upsert_chunks(
                collection=collection,
                chunks=chunks,
                embedder=embedder,
                batch_size=batch_size,
            )
            t1 = time.perf_counter()

        st.success(f"Indexed/updated **{n}** chunk(s) into ChromaDB.")
        st.caption(f"Index time: {(t1 - t0):.2f}s")

    st.markdown("---")
    st.markdown("### Ask a Question (Retrieve → Re-rank → Answer as JSON)")

    question = st.text_area(
        "Question", value="What is the capital of Australia?", height=70
    )

    colR1, colR2, colR3, colR4 = st.columns([1, 1, 1, 1])
    with colR1:
        use_rerank = st.checkbox(
            "Use re-rank",
            value=True,
            help="Re-rank improves evidence quality; costs extra time.",
        )
    with colR2:
        top_n = int(
            st.slider(
                "Retrieve Top-N",
                3,
                20,
                10,
                1,
                help="Initial candidates from Chroma vector search.",
            )
        )
    with colR3:
        top_k = int(
            st.slider(
                "Keep Top-K",
                1,
                8,
                4,
                1,
                help="Chunks passed to the generator after re-ranking.",
            )
        )
    with colR4:
        max_new_tokens = int(st.slider("Max new tokens", 64, 256, 128, 8))

    ask_btn = st.button(
        "🔎 Retrieve + Answer", type="primary", use_container_width=True
    )
    if not ask_btn:
        return

    # Retrieve
    with st.spinner("1/3 Retrieving candidates from ChromaDB..."):
        t_retr0 = time.perf_counter()
        candidates = _retrieve_topn(
            collection=collection, query=question, embedder=embedder, top_n=top_n
        )
        t_retr1 = time.perf_counter()

    if not candidates:
        st.warning(
            "No retrieved chunks. Build the index first (and ensure knowledge_base has content)."
        )
        return

    # Re-rank
    selected: List[RetrievedChunk] = candidates
    t_rr0 = t_rr1 = 0.0
    if use_rerank:
        with st.spinner("2/3 Re-ranking candidates with a cross-encoder..."):
            reranker = _load_reranker(rerank_model)
            t_rr0 = time.perf_counter()
            reranked = _rerank(
                reranker=reranker, question=question, candidates=candidates
            )
            t_rr1 = time.perf_counter()
            selected = reranked[:top_k]
    else:
        selected = candidates[:top_k]

    # Show retrieved and selected evidence
    st.markdown("#### Evidence selection")
    st.write(
        "Vector retrieval gets you **fast candidates**. Re-rank chooses the **best evidence** among them "
        "before generation."
    )

    with st.expander("Show Top-N candidates (vector retrieval)"):
        for i, r in enumerate(candidates, start=1):
            title = f"[{i}] {r.source} | chunk {r.chunk_id} | dist={r.distance:.4f}"
            st.markdown(f"**{title}**")
            st.write(r.text[:600] + ("…" if len(r.text) > 600 else ""))
            st.markdown("---")

    st.markdown("#### Final Top-K evidence passed to the model")
    for i, r in enumerate(selected, start=1):
        score_str = (
            f" | rerank={r.rerank_score:.4f}" if r.rerank_score is not None else ""
        )
        with st.expander(f"[{i}] {r.source} | chunk {r.chunk_id}{score_str}"):
            st.write(r.text)

    # Generate strict JSON
    gen = _load_generator(gen_model)
    prompt = _build_strict_json_prompt(question=question, retrieved=selected)

    with st.spinner("3/3 Generating strict JSON answer..."):
        t_gen0 = time.perf_counter()
        raw = _generate_text(gen, prompt, max_new_tokens=max_new_tokens)
        t_gen1 = time.perf_counter()

    st.markdown("---")
    st.markdown("### Output (Strict JSON)")

    parsed, err = _try_parse_json(raw)
    if parsed is None:
        st.error(f"Model did not output valid JSON ({err}). Using honest fallback.")
        parsed = _honest_fallback_json(raw_output=raw, retrieved=selected)

    st.json(parsed)

    st.markdown("### Raw model output (debug)")
    st.code(raw, language="text")

    st.markdown("---")
    st.markdown("### Timing (so you can see the cost of re-rank)")
    st.write(f"- Retrieval time: **{(t_retr1 - t_retr0):.2f}s** (fast)")
    if use_rerank:
        st.write(
            f"- Re-rank time: **{(t_rr1 - t_rr0):.2f}s** (slower but higher quality)"
        )
    st.write(f"- Generation time: **{(t_gen1 - t_gen0):.2f}s**")

    with st.expander("Show exact prompt (debug)"):
        st.code(prompt, language="text")
