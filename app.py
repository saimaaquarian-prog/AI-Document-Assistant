```python
import io
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


# =========================================================
# APP SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# You can change this model if your Groq account uses another model.
GROQ_MODEL = "openai/gpt-oss-20b"


# =========================================================
# SESSION STATE
# =========================================================

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


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


# =========================================================
# DOCUMENT EXTRACTION
# =========================================================

def extract_pdf(file_bytes, filename):
    """
    Extract text from PDF.

    Page number is preserved for every extracted page.
    """

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
    """
    Extract text from DOCX.

    DOCX does not naturally provide reliable page numbers,
    so page is stored as None.
    """

    document = Document(io.BytesIO(file_bytes))

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    text = "\n".join(paragraphs)

    if not text.strip():
        return []

    return [
        {
            "filename": filename,
            "page": None,
            "text": text.strip(),
        }
    ]


def extract_text_file(file_bytes, filename):
    """
    Extract TXT or Markdown text.
    """

    text = file_bytes.decode(
        "utf-8",
        errors="ignore",
    ).strip()

    if not text:
        return []

    return [
        {
            "filename": filename,
            "page": None,
            "text": text,
        }
    ]


def extract_document(file_bytes, filename):
    """
    Main extraction function.

    Selects the correct extractor based on file extension.
    """

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(
            file_bytes,
            filename,
        )

    elif extension == ".docx":
        return extract_docx(
            file_bytes,
            filename,
        )

    elif extension in {".txt", ".md"}:
        return extract_text_file(
            file_bytes,
            filename,
        )

    return []


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunk_text(
    text,
    chunk_size=900,
    overlap=150,
):
    """
    Split text into overlapping chunks.
    """

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text),
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def create_chunks(extracted_pages):
    """
    Create chunks while preserving filename and page metadata.
    """

    all_chunks = []

    for page_data in extracted_pages:

        pieces = chunk_text(
            page_data["text"]
        )

        for piece in pieces:

            all_chunks.append(
                {
                    "text": piece,
                    "filename": page_data["filename"],
                    "page": page_data["page"],
                }
            )

    return all_chunks


# =========================================================
# EMBEDDINGS + FAISS
# =========================================================

def add_documents_to_index(new_chunks):
    """
    Create embeddings for new chunks and add them
    to the existing FAISS index.

    Document embeddings are NOT recreated when the user
    asks another question.
    """

    if not new_chunks:
        return

    model = load_embedding_model()

    texts = [
        item["text"]
        for item in new_chunks
    ]

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # Create FAISS index only once.
    if st.session_state.index is None:

        dimension = vectors.shape[1]

        st.session_state.index = faiss.IndexFlatIP(
            dimension
        )

    # Add new vectors.
    st.session_state.index.add(vectors)

    # Save chunks.
    st.session_state.chunks.extend(
        new_chunks
    )

    # Save embeddings.
    if st.session_state.embeddings is None:

        st.session_state.embeddings = vectors

    else:

        st.session_state.embeddings = np.vstack(
            [
                st.session_state.embeddings,
                vectors,
            ]
        )


# =========================================================
# KEYWORD SEARCH
# =========================================================

STOP_WORDS = {
    "the",
    "is",
    "are",
    "was",
    "were",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "what",
    "which",
    "who",
    "how",
    "why",
    "when",
    "where",
    "does",
    "do",
    "did",
    "can",
    "could",
    "please",
    "tell",
    "me",
    "about",
}


def important_words(question):
    """
    Extract important words from the question.
    """

    words = re.findall(
        r"\b[a-zA-Z0-9]{3,}\b",
        question.lower(),
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
    ]


def keyword_scores(question):
    """
    Calculate keyword matching score for every chunk.
    """

    words = important_words(question)

    scores = []

    for item in st.session_state.chunks:

        chunk_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]{3,}\b",
                item["text"].lower(),
            )
        )

        if not words:

            scores.append(0.0)

            continue

        matches = sum(
            1
            for word in words
            if word in chunk_words
        )

        score = matches / len(words)

        scores.append(score)

    return np.array(
        scores,
        dtype="float32",
    )


# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(
    question,
    top_k=5,
):
    """
    Combine semantic search and keyword search.

    Semantic score = 70%
    Keyword score = 30%
    """

    if (
        not st.session_state.chunks
        or st.session_state.index is None
    ):
        return []

    model = load_embedding_model()

    # Create embedding ONLY for the question.
    question_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # ---------------------------------------------
    # Semantic search
    # ---------------------------------------------

    semantic_scores, semantic_ids = (
        st.session_state.index.search(
            question_vector,
            min(
                top_k * 3,
                len(st.session_state.chunks),
            ),
        )
    )

    semantic_scores = semantic_scores[0]

    semantic_ids = semantic_ids[0]

    # ---------------------------------------------
    # Keyword search
    # ---------------------------------------------

    keyword = keyword_scores(
        question
    )

    # ---------------------------------------------
    # Candidate chunks
    # ---------------------------------------------

    candidate_ids = set(
        semantic_ids.tolist()
    )

    keyword_ids = np.argsort(
        keyword
    )[::-1][: top_k * 3]

    candidate_ids.update(
        keyword_ids.tolist()
    )

    results = []

    # ---------------------------------------------
    # Calculate hybrid score
    # ---------------------------------------------

    for idx in candidate_ids:

        if (
            idx < 0
            or idx >= len(
                st.session_state.chunks
            )
        ):
            continue

        semantic = float(
            np.dot(
                question_vector[0],
                st.session_state.embeddings[idx],
            )
        )

        keyword_score = float(
            keyword[idx]
        )

        hybrid_score = (
            0.70 * semantic
            + 0.30 * keyword_score
        )

        results.append(
            {
                **st.session_state.chunks[idx],
                "semantic_score": semantic,
                "keyword_score": keyword_score,
                "hybrid_score": hybrid_score,
            }
        )

    # Highest hybrid score first.
    results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return results[:top_k]


# =========================================================
# GOOGLE DRIVE
# =========================================================

def download_drive_files(url):
    """
    Download a public Google Drive file or folder.

    Supported:
    PDF
    DOCX
    TXT
    MD

    The function avoids using fuzzy=True because some
    gdown versions do not support that argument.
    """

    temp_dir = Path(
        tempfile.mkdtemp()
    )

    try:

        # =================================================
        # GOOGLE DRIVE FOLDER
        # =================================================

        if "/folders/" in url:

            downloaded_files = (
                gdown.download_folder(
                    url,
                    output=str(temp_dir),
                    quiet=True,
                    use_cookies=False,
                )
            )

            if not downloaded_files:
                return []

            files = []

            for path in temp_dir.rglob("*"):

                if not path.is_file():
                    continue

                extension = (
                    path.suffix.lower()
                )

                if extension in SUPPORTED_EXTENSIONS:

                    files.append(
                        (
                            path.name,
                            path.read_bytes(),
                        )
                    )

            return files

        # =================================================
        # GOOGLE DRIVE SINGLE FILE
        # =================================================

        # Extract Google Drive file ID.

        match = re.search(
            r"/d/([a-zA-Z0-9_-]+)",
            url,
        )

        if not match:

            match = re.search(
                r"id=([a-zA-Z0-9_-]+)",
                url,
            )

        if not match:

            st.error(
                "Could not find the Google Drive file ID. "
                "Please paste a normal Google Drive file link."
            )

            return []

        file_id = match.group(1)

        # Download without fuzzy=True.
        downloaded = gdown.download(
            id=file_id,
            output=str(
                temp_dir / "drive_file"
            ),
            quiet=True,
        )

        if not downloaded:
            return []

        downloaded_path = Path(
            downloaded
        )

        if not downloaded_path.exists():
            return []

        file_bytes = (
            downloaded_path.read_bytes()
        )

        if not file_bytes:
            return []

        # ---------------------------------------------
        # Detect file type from file content.
        # This solves the problem where gdown does
        # not preserve the original file extension.
        # ---------------------------------------------

        # PDF signature
        if file_bytes.startswith(
            b"%PDF"
        ):

            filename = (
                "google_drive_document.pdf"
            )

        # DOCX is a ZIP-based Office file.
        elif file_bytes.startswith(
            b"PK"
        ):

            # Check whether it really looks like
            # an Office DOCX package.

            import zipfile

            try:

                with zipfile.ZipFile(
                    io.BytesIO(file_bytes)
                ) as zip_file:

                    names = zip_file.namelist()

                    if (
                        "[Content_Types].xml"
                        in names
                        and "word/document.xml"
                        in names
                    ):

                        filename = (
                            "google_drive_document.docx"
                        )

                    else:

                        filename = (
                            "google_drive_document.txt"
                        )

            except zipfile.BadZipFile:

                filename = (
                    "google_drive_document.txt"
                )

        # Otherwise treat it as text.
        else:

            filename = (
                "google_drive_document.txt"
            )

        return [
            (
                filename,
                file_bytes,
            )
        ]

    except Exception as exc:

        st.error(
            f"Google Drive loading failed: {exc}"
        )

        return []


# =========================================================
# PROCESS DOCUMENT
# =========================================================

def process_file(
    file_bytes,
    filename,
    source="Local upload",
):
    """
    Complete document pipeline:

    File
      ↓
    Extraction
      ↓
    Chunking
      ↓
    Embedding
      ↓
    FAISS
    """

    # Create a unique key from filename + file contents.
    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    document_key = (
        f"{filename}:{file_hash}"
    )

    # Do not process the same document twice.
    if (
        document_key
        in st.session_state.document_keys
    ):

        return (
            0,
            0,
            True,
        )

    # ---------------------------------------------
    # Extraction
    # ---------------------------------------------

    extracted = extract_document(
        file_bytes,
        filename,
    )

    # ---------------------------------------------
    # Chunking
    # ---------------------------------------------

    chunks = create_chunks(
        extracted
    )

    # ---------------------------------------------
    # Embedding + FAISS
    # ---------------------------------------------

    add_documents_to_index(
        chunks
    )

    # Remember the document.
    st.session_state.document_keys.add(
        document_key
    )

    # Save document information.
    st.session_state.documents.append(
        {
            "filename": filename,
            "source": source,
            "pages_or_sections": len(
                extracted
            ),
            "characters": sum(
                len(item["text"])
                for item in extracted
            ),
            "chunks": len(chunks),
        }
    )

    return (
        len(extracted),
        len(chunks),
        False,
    )


# =========================================================
# GROQ
# =========================================================

def ask_groq(
    question,
    retrieved_chunks,
):
    """
    Send the question and retrieved chunks
    to Groq.

    The API key is read from Streamlit Secrets.
    """

    # ---------------------------------------------
    # Check API key
    # ---------------------------------------------

    if "GROQ_API_KEY" not in st.secrets:

        st.error(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets."
        )

        return None

    # ---------------------------------------------
    # Build context
    # ---------------------------------------------

    context_parts = []

    for number, item in enumerate(
        retrieved_chunks,
        start=1,
    ):

        if item["page"]:

            location = (
                f"{item['filename']}, "
                f"page {item['page']}"
            )

        else:

            location = item["filename"]

        context_parts.append(
            f"[Source {number}: {location}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    # ---------------------------------------------
    # Groq client
    # ---------------------------------------------

    client = Groq(
        api_key=st.secrets[
            "GROQ_API_KEY"
        ]
    )

    # ---------------------------------------------
    # System prompt
    # ---------------------------------------------

    system_prompt = """
You are an AI Document Assistant.

Your job is to answer questions using ONLY
the document context provided to you.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. Do not make assumptions when the context
   does not contain the answer.
4. If the answer is not available in the
   provided context, say exactly:

"I could not find this information in the
provided documents."

5. Keep the answer clear and concise.
6. When possible, mention the relevant
   document name and page number.
"""

    # ---------------------------------------------
    # User prompt
    # ---------------------------------------------

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}


USER QUESTION:

{question}
"""

    # ---------------------------------------------
    # Call Groq
    # ---------------------------------------------

    try:

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.2,
            max_completion_tokens=1200,
        )

        return response.choices[
            0
        ].message.content

    except Exception as exc:

        st.error(
            f"Groq error: {exc}"
        )

        return None


# =========================================================
# USER INTERFACE
# =========================================================

st.title(
    "📚 AI Document Assistant"
)

st.caption(
    "Upload PDF, DOCX, TXT or MD documents "
    "or load a public Google Drive file/folder."
)


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

with st.sidebar:

    st.header(
        "⚙️ Search Settings"
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=500,
        max_value=1500,
        value=900,
        step=100,
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=50,
        max_value=300,
        value=150,
        step=25,
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=8,
        value=5,
    )

    st.divider()

    st.write(
        f"**Documents:** "
        f"{len(st.session_state.documents)}"
    )

    st.write(
        f"**Chunks:** "
        f"{len(st.session_state.chunks)}"
    )

    st.divider()

    if st.button(
        "🗑️ Clear all documents"
    ):

        st.session_state.documents = []

        st.session_state.chunks = []

        st.session_state.embeddings = None

        st.session_state.index = None

        st.session_state.document_keys = set()

        st.rerun()


# =========================================================
# LOCAL UPLOAD
# =========================================================

st.subheader(
    "1️⃣ Upload Local Documents"
)

uploaded_files = st.file_uploader(
    "Choose PDF, DOCX, TXT or MD files",
    type=[
        "pdf",
        "docx",
        "txt",
        "md",
    ],
    accept_multiple_files=True,
)


if st.button(
    "Process Uploaded Documents",
    type="primary",
):

    if not uploaded_files:

        st.warning(
            "Please upload at least one document."
        )

    else:

        total_chunks = 0

        with st.spinner(
            "Extracting, chunking and embedding documents..."
        ):

            for uploaded_file in uploaded_files:

                file_bytes = (
                    uploaded_file.getvalue()
                )

                _, chunks_created, already_exists = (
                    process_file(
                        file_bytes,
                        uploaded_file.name,
                        source="Local upload",
                    )
                )

                if already_exists:

                    st.info(
                        f"{uploaded_file.name} "
                        "was already processed."
                    )

                else:

                    total_chunks += (
                        chunks_created
                    )

        st.success(
            f"Processing complete. "
            f"Created {total_chunks} new chunks."
        )


# =========================================================
# GOOGLE DRIVE
# =========================================================

st.subheader(
    "2️⃣ Google Drive"
)

drive_url = st.text_input(
    "Paste a public Google Drive file or folder link",
    placeholder=(
        "https://drive.google.com/file/d/..."
    ),
)


if st.button(
    "Load from Google Drive"
):

    if not drive_url.strip():

        st.warning(
            "Please paste a Google Drive link."
        )

    else:

        with st.spinner(
            "Downloading and processing Google Drive files..."
        ):

            drive_files = download_drive_files(
                drive_url.strip()
            )

            if not drive_files:

                st.warning(
                    "No supported PDF, DOCX, TXT or MD "
                    "files were found. Make sure the "
                    "Drive file/folder is publicly accessible."
                )

            else:

                total_chunks = 0

                for (
                    filename,
                    file_bytes,
                ) in drive_files:

                    (
                        _,
                        chunks_created,
                        already_exists,
                    ) = process_file(
                        file_bytes,
                        filename,
                        source="Google Drive",
                    )

                    if not already_exists:

                        total_chunks += (
                            chunks_created
                        )

                st.success(
                    "Google Drive processing complete. "
                    f"Created {total_chunks} new chunks."
                )


# =========================================================
# DOCUMENT INFORMATION
# =========================================================

st.subheader(
    "3️⃣ Document Information"
)

if st.session_state.documents:

    for document in st.session_state.documents:

        with st.expander(
            document["filename"]
        ):

            st.write(
                f"**Source:** "
                f"{document['source']}"
            )

            st.write(
                f"**Pages/sections extracted:** "
                f"{document['pages_or_sections']}"
            )

            st.write(
                f"**Characters extracted:** "
                f"{document['characters']}"
            )

            st.write(
                f"**Chunks created:** "
                f"{document['chunks']}"
            )

else:

    st.info(
        "No documents processed yet."
    )


# =========================================================
# ASK QUESTION
# =========================================================

st.subheader(
    "4️⃣ Ask a Question"
)

question = st.text_area(
    "Question",
    placeholder=(
        "Ask something about your documents..."
    ),
    height=100,
)


if st.button(
    "🔎 Ask AI",
    type="primary",
):

    if not st.session_state.chunks:

        st.warning(
            "Please process at least one "
            "document first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            # -----------------------------------------
            # Hybrid retrieval
            # -----------------------------------------

            retrieved = hybrid_search(
                question.strip(),
                top_k=top_k,
            )

            if not retrieved:

                st.warning(
                    "No relevant document chunks "
                    "were found."
                )

            else:

                # -------------------------------------
                # Generate answer
                # -------------------------------------

                answer = ask_groq(
                    question.strip(),
                    retrieved,
                )

                if answer:

                    st.markdown(
                        "### 🤖 Answer"
                    )

                    st.write(
                        answer
                    )

                    # ---------------------------------
                    # Retrieved sources
                    # ---------------------------------

                    st.markdown(
                        "### 📚 Retrieved Sources"
                    )

                    for number, item in enumerate(
                        retrieved,
                        start=1,
                    ):

                        if item["page"]:

                            page_text = (
                                f"Page "
                                f"{item['page']}"
                            )

                        else:

                            page_text = (
                                "Page not available"
                            )

                        with st.expander(
                            f"Source {number} — "
                            f"{item['filename']} — "
                            f"{page_text}"
                        ):

                            st.write(
                                f"**Hybrid score:** "
                                f"{item['hybrid_score']:.3f}"
                            )

                            st.write(
                                f"**Semantic score:** "
                                f"{item['semantic_score']:.3f}"
                            )

                            st.write(
                                f"**Keyword score:** "
                                f"{item['keyword_score']:.3f}"
                            )

                            st.markdown(
                                "**Retrieved text:**"
                            )

                            st.write(
                                item["text"]
                            )
```
