import io
import os
import re
import hashlib
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# App settings
# -----------------------------
st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="wide")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"


# -----------------------------
# Session state
# -----------------------------
if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "index" not in st.session_state:
    st.session_state.index = None

if "document_keys" not in st.session_state:
    st.session_state.document_keys = set()


# -----------------------------
# Cached models
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(file_bytes, filename):
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "filename": filename,
                    "page": page_number,
                    "text": text.strip(),
                }
            )

    return pages


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    )

    if not text.strip():
        return []

    return [{"filename": filename, "page": None, "text": text.strip()}]


def extract_text_file(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore").strip()

    if not text:
        return []

    return [{"filename": filename, "page": None, "text": text}]


def extract_document(file_bytes, filename):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)

    if extension == ".docx":
        return extract_docx(file_bytes, filename)

    if extension in {".txt", ".md"}:
        return extract_text_file(file_bytes, filename)

    return []


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=900, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def create_chunks(extracted_pages):
    all_chunks = []

    for page_data in extracted_pages:
        pieces = chunk_text(page_data["text"])

        for piece in pieces:
            all_chunks.append(
                {
                    "text": piece,
                    "filename": page_data["filename"],
                    "page": page_data["page"],
                }
            )

    return all_chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def add_documents_to_index(new_chunks):
    if not new_chunks:
        return

    model = load_embedding_model()

    texts = [item["text"] for item in new_chunks]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    if st.session_state.index is None:
        dimension = vectors.shape[1]
        st.session_state.index = faiss.IndexFlatIP(dimension)

    st.session_state.index.add(vectors)

    st.session_state.chunks.extend(new_chunks)

    if st.session_state.embeddings is None:
        st.session_state.embeddings = vectors
    else:
        st.session_state.embeddings = np.vstack(
            [st.session_state.embeddings, vectors]
        )


# -----------------------------
# Keyword search
# -----------------------------
STOP_WORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and", "or", "of",
    "to", "in", "on", "for", "with", "what", "which", "who", "how",
    "why", "when", "where", "does", "do", "did", "can", "could",
    "please", "tell", "me", "about"
}


def important_words(question):
    words = re.findall(r"\b[a-zA-Z0-9]{3,}\b", question.lower())
    return [word for word in words if word not in STOP_WORDS]


def keyword_scores(question):
    words = important_words(question)
    scores = []

    for item in st.session_state.chunks:
        chunk_words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", item["text"].lower()))

        if not words:
            scores.append(0.0)
            continue

        matches = sum(1 for word in words if word in chunk_words)
        scores.append(matches / len(words))

    return np.array(scores, dtype="float32")


# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(question, top_k=5):
    if not st.session_state.chunks or st.session_state.index is None:
        return []

    model = load_embedding_model()

    question_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    semantic_scores, semantic_ids = st.session_state.index.search(
        question_vector, min(top_k * 3, len(st.session_state.chunks))
    )

    semantic_scores = semantic_scores[0]
    semantic_ids = semantic_ids[0]

    keyword = keyword_scores(question)

    # Candidate chunks come from semantic search plus keyword matches.
    candidate_ids = set(semantic_ids.tolist())

    keyword_ids = np.argsort(keyword)[::-1][: top_k * 3]
    candidate_ids.update(keyword_ids.tolist())

    results = []

    for idx in candidate_ids:
        if idx < 0 or idx >= len(st.session_state.chunks):
            continue

        semantic = float(
            np.dot(
                question_vector[0],
                st.session_state.embeddings[idx],
            )
        )

        key_score = float(keyword[idx])

        # 70% semantic + 30% keyword
        hybrid = 0.70 * semantic + 0.30 * key_score

        results.append(
            {
                **st.session_state.chunks[idx],
                "semantic_score": semantic,
                "keyword_score": key_score,
                "hybrid_score": hybrid,
            }
        )

    results.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return results[:top_k]


# -----------------------------
# Google Drive
# -----------------------------
def download_drive_files(url):
    temp_dir = Path(tempfile.mkdtemp())

    try:
        if "/folders/" in url:
            downloaded = gdown.download_folder(
                url,
                output=str(temp_dir),
                quiet=True,
                use_cookies=False,
                remaining_ok=True,
            )

            if not downloaded:
                return []

            files = []
            for path in temp_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append((path.name, path.read_bytes()))

            return files

        output_path = temp_dir / "drive_file"
        downloaded = gdown.download(
            url=url,
            output=str(output_path),
            quiet=True,
            fuzzy=True,
        )

        if not downloaded:
            return []

        # gdown may return a filename with the real extension.
        downloaded_path = Path(downloaded)
        if downloaded_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return []

        return [(downloaded_path.name, downloaded_path.read_bytes())]

    except Exception as exc:
        st.error(f"Google Drive loading failed: {exc}")
        return []


# -----------------------------
# Document processing
# -----------------------------
def process_file(file_bytes, filename, source="Local upload"):
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    document_key = f"{filename}:{file_hash}"

    if document_key in st.session_state.document_keys:
        return 0, 0, True

    extracted = extract_document(file_bytes, filename)
    chunks = create_chunks(extracted)

    add_documents_to_index(chunks)

    st.session_state.document_keys.add(document_key)

    st.session_state.documents.append(
        {
            "filename": filename,
            "source": source,
            "pages_or_sections": len(extracted),
            "characters": sum(len(item["text"]) for item in extracted),
            "chunks": len(chunks),
        }
    )

    return len(extracted), len(chunks), False


# -----------------------------
# Groq
# -----------------------------
def ask_groq(question, retrieved_chunks):
    if "GROQ_API_KEY" not in st.secrets:
        st.error("GROQ_API_KEY is missing. Add it to Streamlit Secrets.")
        return None

    context_parts = []

    for number, item in enumerate(retrieved_chunks, start=1):
        page = f", page {item['page']}" if item["page"] else ""
        context_parts.append(
            f"[Source {number}: {item['filename']}{page}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    system_prompt = """
You are an AI Document Assistant.

Answer the user's question ONLY from the provided document context.
Do not use outside knowledge.
If the answer is not available in the context, say:
"I could not find this information in the provided documents."

Keep the answer clear and concise.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_completion_tokens=1200,
    )

    return response.choices[0].message.content


# -----------------------------
# UI
# -----------------------------
st.title("📚 AI Document Assistant")
st.caption(
    "Upload documents or load a public Google Drive file/folder, then ask questions using hybrid semantic + keyword search."
)

with st.sidebar:
    st.header("Settings")

    chunk_size = st.slider("Chunk size", 500, 1500, 900, 100)
    chunk_overlap = st.slider("Chunk overlap", 50, 300, 150, 25)
    top_k = st.slider("Retrieved chunks", 2, 8, 5)

    st.divider()

    st.write(f"**Documents:** {len(st.session_state.documents)}")
    st.write(f"**Chunks:** {len(st.session_state.chunks)}")

    if st.button("Clear all documents"):
        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.index = None
        st.session_state.document_keys = set()
        st.rerun()


st.subheader("1. Upload local documents")

uploaded_files = st.file_uploader(
    "Choose PDF, DOCX, TXT or MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

if st.button("Process uploaded documents", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one document.")
    else:
        total_chunks = 0

        with st.spinner("Extracting, chunking and embedding documents..."):
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getvalue()
                _, chunks_created, already_exists = process_file(
                    file_bytes,
                    uploaded_file.name,
                    source="Local upload",
                )

                if already_exists:
                    st.info(f"{uploaded_file.name} was already processed.")
                else:
                    total_chunks += chunks_created

        st.success(f"Processing complete. Created {total_chunks} new chunks.")


st.subheader("2. Google Drive")

drive_url = st.text_input(
    "Paste a public Google Drive file or folder link",
    placeholder="https://drive.google.com/...",
)

if st.button("Load from Google Drive"):
    if not drive_url.strip():
        st.warning("Please paste a Google Drive link.")
    else:
        with st.spinner("Downloading and processing Google Drive files..."):
            drive_files = download_drive_files(drive_url.strip())

            if not drive_files:
                st.warning(
                    "No supported PDF, DOCX, TXT or MD files were found. "
                    "Make sure the Drive file/folder is publicly accessible."
                )
            else:
                total_chunks = 0

                for filename, file_bytes in drive_files:
                    _, chunks_created, already_exists = process_file(
                        file_bytes,
                        filename,
                        source="Google Drive",
                    )

                    if not already_exists:
                        total_chunks += chunks_created

                st.success(
                    f"Google Drive processing complete. Created {total_chunks} new chunks."
                )


st.subheader("3. Document information")

if st.session_state.documents:
    for document in st.session_state.documents:
        with st.expander(document["filename"]):
            st.write(f"**Source:** {document['source']}")
            st.write(f"**Pages/sections extracted:** {document['pages_or_sections']}")
            st.write(f"**Characters extracted:** {document['characters']}")
            st.write(f"**Chunks created:** {document['chunks']}")
else:
    st.info("No documents processed yet.")


st.subheader("4. Ask a question")

question = st.text_area(
    "Question",
    placeholder="Ask something about your uploaded documents...",
    height=100,
)

if st.button("Ask AI", type="primary"):
    if not st.session_state.chunks:
        st.warning("Please process at least one document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documents and generating answer..."):
            retrieved = hybrid_search(question.strip(), top_k=top_k)

            if not retrieved:
                st.warning("No relevant document chunks were found.")
            else:
                answer = ask_groq(question.strip(), retrieved)

                if answer:
                    st.markdown("### Answer")
                    st.write(answer)

                    st.markdown("### Retrieved Sources")

                    for number, item in enumerate(retrieved, start=1):
                        page_text = (
                            f"Page {item['page']}"
                            if item["page"]
                            else "Page not available"
                        )

                        with st.expander(
                            f"Source {number} — {item['filename']} — {page_text}"
                        ):
                            st.write(
                                f"**Hybrid score:** {item['hybrid_score']:.3f}"
                            )
                            st.write(
                                f"**Semantic score:** {item['semantic_score']:.3f}"
                            )
                            st.write(
                                f"**Keyword score:** {item['keyword_score']:.3f}"
                            )
                            st.markdown("**Retrieved text:**")
                            st.write(item["text"])
