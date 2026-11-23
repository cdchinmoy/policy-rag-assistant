"""Minimal local RAG for policy / FAQ documents.

Usage:
    python rag.py ingest              # index everything in ./docs
    python rag.py ask "your question" # ask a question
"""
import pathlib
import re
import sys

import chromadb
from anthropic import Anthropic

DOCS_DIR = pathlib.Path("docs")
DB_DIR = "chroma_db"
COLLECTION = "policies"
MODEL = "claude-sonnet-5-5"
TOP_K = 5
CHUNK_CHARS = 2000  # roughly 500 tokens

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.
Rules:
- If the context does not contain the answer, say you don't know and suggest contacting a person. Do not guess.
- For rules, fees, deadlines and procedures, quote the policy wording instead of paraphrasing loosely.
- Cite the passages you used like [1] or [2][3].
- Be concise."""


# ---------- loading ----------
def read_sections(path: pathlib.Path):
    """Return a list of (section_name, text) for one file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(f"page {i + 1}", page.extract_text() or "") for i, page in enumerate(reader.pages)]

    text = path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".md":
        sections, current, lines = [], "Intro", []
        for line in text.splitlines():
            if re.match(r"^#{1,4}\s", line):
                if lines:
                    sections.append((current, "\n".join(lines)))
                current, lines = line.lstrip("# ").strip(), []
            else:
                lines.append(line)
        if lines:
            sections.append((current, "\n".join(lines)))
        return sections
    return [("", text)]


# ---------- chunking ----------
def chunk_text(text: str, size: int = CHUNK_CHARS):
    """Pack paragraphs into chunks of up to `size` characters."""
    chunks, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if current and len(current) + len(para) > size:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{para}".strip()
    if current:
        chunks.append(current)
    return chunks


# ---------- indexing ----------
def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=DB_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION)


def ingest():
    col = get_collection(reset=True)
    ids, docs, metas = [], [], []
    files = [p for p in DOCS_DIR.rglob("*") if p.suffix.lower() in {".md", ".txt", ".pdf"}]
    for path in files:
        for section, text in read_sections(path):
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{path.name}-{section}-{i}-{len(ids)}")
                docs.append(chunk)
                metas.append({"source": path.name, "section": section})
    if not docs:
        sys.exit("No documents found. Put .md, .txt or .pdf files in ./docs")
    col.add(ids=ids, documents=docs, metadatas=metas)
    print(f"Indexed {len(docs)} chunks from {len(files)} files.")


# ---------- retrieval + answer ----------
def retrieve(question: str, k: int = TOP_K):
    col = get_collection()
    res = col.query(query_texts=[question], n_results=k)
    return list(zip(res["documents"][0], res["metadatas"][0]))


def ask(question: str):
    hits = retrieve(question)
    context = "\n\n".join(
        f"[{i}] ({m['source']}, {m['section']})\n{d}" for i, (d, m) in enumerate(hits, 1)
    )
    client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    msg = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return msg.content[0].text, [m for _, m in hits]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ingest":
        ingest()
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        answer, sources = ask(" ".join(sys.argv[2:]))
        print(answer)
        print("\nSources retrieved:")
        for i, m in enumerate(sources, 1):
            print(f"  [{i}] {m['source']} - {m['section']}")
    else:
        print(__doc__)
