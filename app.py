import io
import re
import hashlib
import tempfile
import zipfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "AI Document Assistant"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5

STOP_WORDS = {
    "the", "is", "are", "was", "were", "a", "an", "and", "or",
    "of", "to", "in", "on", "for", "from", "with", "by", "as",
    "at", "be", "this", "that", "these", "those", "it", "its",
    "what", "which", "who", "when", "where", "why", "how",
    "can", "could", "would", "should", "do", "does", "did",
    "about", "please", "tell", "me"
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Document Assistant")

st.caption(
    "Upload documents or load a public Google Drive file, "
    "then ask questions using hybrid semantic + keyword search."
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():

    defaults = {
        "documents": [],
        "chunks": [],
        "embeddings": None,
        "index": None,
        "document_keys": set(),
        "last_question": "",
        "last_answer": "",
        "last_sources": []
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


initialize_session_state()


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(file_bytes, filename):

    pages = []

    reader = PdfReader(
        io.BytesIO(file_bytes)
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            text = page.extract_text() or ""

        except Exception:

            text = ""

        if text.strip():

            pages.append(
                {
                    "filename": filename,
                    "page": page_number,
                    "text": text.strip()
                }
            )

    return pages


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(file_bytes, filename):

    document = Document(
        io.BytesIO(file_bytes)
    )

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append(text)

    text = "\n".join(
        paragraphs
    ).strip()

    if not text:

        return []

    return [
        {
            "filename": filename,
            "page": None,
            "text": text
        }
    ]


# ============================================================
# TXT / MD EXTRACTION
# ============================================================

def extract_text_file(file_bytes, filename):

    text = file_bytes.decode(
        "utf-8",
        errors="ignore"
    ).strip()

    if not text:

        return []

    return [
        {
            "filename": filename,
            "page": None,
            "text": text
        }
    ]


# ============================================================
# DOCUMENT EXTRACTION ROUTER
# ============================================================

def extract_document(file_bytes, filename):

    extension = Path(
        filename
    ).suffix.lower()

    if extension == ".pdf":

        return extract_pdf(
            file_bytes,
            filename
        )

    if extension == ".docx":

        return extract_docx(
            file_bytes,
            filename
        )

    if extension in {".txt", ".md"}:

        return extract_text_file(
            file_bytes,
            filename
        )

    raise ValueError(
        f"Unsupported file type: {extension}. "
        "Supported types are PDF, DOCX, TXT and MD."
    )


# ============================================================
# GOOGLE DRIVE FILE TYPE DETECTION
# ============================================================

def detect_file_type(file_bytes):

    # PDF
    if file_bytes[:4] == b"%PDF":

        return ".pdf"

    # DOCX
    if file_bytes[:2] == b"PK":

        try:

            with zipfile.ZipFile(
                io.BytesIO(file_bytes)
            ) as archive:

                names = set(
                    archive.namelist()
                )

            if (
                "[Content_Types].xml" in names
                and "word/document.xml" in names
            ):

                return ".docx"

        except Exception:

            pass

    # TXT / MD
    try:

        file_bytes.decode(
            "utf-8"
        )

        return ".txt"

    except UnicodeDecodeError:

        return None


def make_filename_with_extension(
    filename,
    file_bytes
):

    path = Path(filename)

    if path.suffix.lower() in SUPPORTED_EXTENSIONS:

        return path.name

    detected = detect_file_type(
        file_bytes
    )

    if detected:

        return f"{path.stem}{detected}"

    return path.name


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.replace(
        "\x00",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    text,
    chunk_size=DEFAULT_CHUNK_SIZE,
    overlap=DEFAULT_CHUNK_OVERLAP
):

    text = normalize_text(
        text
    )

    if not text:

        return []

    chunk_size = max(
        200,
        int(chunk_size)
    )

    overlap = max(
        0,
        int(overlap)
    )

    if overlap >= chunk_size:

        overlap = chunk_size // 4

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= len(text):

            break

        start = end - overlap

    return chunks


def create_chunks(
    extracted_pages,
    chunk_size,
    overlap
):

    all_chunks = []

    for item in extracted_pages:

        text_chunks = chunk_text(
            item["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in text_chunks:

            all_chunks.append(
                {
                    "filename": item["filename"],
                    "page": item["page"],
                    "text": chunk
                }
            )

    return all_chunks


# ============================================================
# EMBEDDINGS + FAISS
# ============================================================

def add_chunks_to_index(new_chunks):

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
        show_progress_bar=False
    ).astype("float32")

    if st.session_state.index is None:

        dimension = vectors.shape[1]

        st.session_state.index = faiss.IndexFlatIP(
            dimension
        )

    st.session_state.index.add(
        vectors
    )

    if st.session_state.embeddings is None:

        st.session_state.embeddings = vectors

    else:

        st.session_state.embeddings = np.vstack(
            [
                st.session_state.embeddings,
                vectors
            ]
        )

    st.session_state.chunks.extend(
        new_chunks
    )


# ============================================================
# KEYWORD SEARCH
# ============================================================

def important_words(question):

    words = re.findall(
        r"\b[a-zA-Z0-9_]+\b",
        question.lower()
    )

    return [
        word
        for word in words
        if len(word) > 2
        and word not in STOP_WORDS
    ]


def keyword_score(
    question,
    text
):

    words = important_words(
        question
    )

    if not words:

        return 0.0

    text_lower = text.lower()

    matched = sum(
        word in text_lower
        for word in words
    )

    return matched / len(words)


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    question,
    top_k=5
):

    if not st.session_state.chunks:

        return []

    if st.session_state.index is None:

        return []

    model = load_embedding_model()

    question_vector = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    total_chunks = len(
        st.session_state.chunks
    )

    search_count = min(
        max(top_k * 4, 10),
        total_chunks
    )

    distances, indices = (
        st.session_state.index.search(
            question_vector,
            search_count
        )
    )

    candidates = {}

    # Semantic candidates
    for score, index_position in zip(
        distances[0],
        indices[0]
    ):

        if index_position < 0:

            continue

        candidates[
            int(index_position)
        ] = float(score)

    # Keyword candidates
    keyword_candidates = []

    for index_position, item in enumerate(
        st.session_state.chunks
    ):

        score = keyword_score(
            question,
            item["text"]
        )

        if score > 0:

            keyword_candidates.append(
                (
                    index_position,
                    score
                )
            )

    keyword_candidates.sort(
        key=lambda item: item[1],
        reverse=True
    )

    for index_position, _ in keyword_candidates[
        :top_k * 3
    ]:

        if index_position not in candidates:

            candidates[index_position] = 0.0

    results = []

    for index_position, semantic_raw in candidates.items():

        item = st.session_state.chunks[
            index_position
        ]

        key_score = keyword_score(
            question,
            item["text"]
        )

        semantic_score = max(
            0.0,
            min(
                1.0,
                (semantic_raw + 1.0) / 2.0
            )
        )

        hybrid_score = (
            0.70 * semantic_score
            + 0.30 * key_score
        )

        results.append(
            {
                "filename": item["filename"],
                "page": item["page"],
                "text": item["text"],
                "semantic_score": semantic_score,
                "keyword_score": key_score,
                "hybrid_score": hybrid_score
            }
        )

    results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    return results[
        :top_k
    ]


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def file_hash(
    file_bytes,
    filename
):

    digest = hashlib.sha256(
        file_bytes
    ).hexdigest()

    return f"{filename}:{digest}"


def process_file(
    file_bytes,
    filename,
    chunk_size,
    overlap
):

    key = file_hash(
        file_bytes,
        filename
    )

    if key in st.session_state.document_keys:

        return (
            False,
            f"{filename} is already loaded."
        )

    extracted_pages = extract_document(
        file_bytes,
        filename
    )

    if not extracted_pages:

        return (
            False,
            f"No readable text was found in {filename}."
        )

    new_chunks = create_chunks(
        extracted_pages,
        chunk_size,
        overlap
    )

    if not new_chunks:

        return (
            False,
            f"No chunks could be created from {filename}."
        )

    add_chunks_to_index(
        new_chunks
    )

    st.session_state.document_keys.add(
        key
    )

    st.session_state.documents.append(
        {
            "filename": filename,
            "pages": len(extracted_pages),
            "chunks": len(new_chunks)
        }
    )

    return (
        True,
        f"Loaded {filename}: "
        f"{len(extracted_pages)} page/section(s), "
        f"{len(new_chunks)} chunk(s)."
    )


# ============================================================
# GOOGLE DRIVE
# ============================================================

def extract_drive_file_id(url):

    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url
        )

        if match:

            return match.group(1)

    return None


def is_drive_folder_url(url):

    return "/folders/" in url


def download_drive_files(url):

    url = url.strip()

    if not url:

        raise ValueError(
            "Please enter a Google Drive URL."
        )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="ai_document_drive_"
        )
    )

    downloaded_files = []

    # --------------------------------------------------------
    # GOOGLE DRIVE FOLDER
    # --------------------------------------------------------

    if is_drive_folder_url(url):

        try:

            result = gdown.download_folder(
                url,
                output=str(temp_dir),
                quiet=True,
                use_cookies=False
            )

        except TypeError:

            result = gdown.download_folder(
                url,
                output=str(temp_dir),
                quiet=True
            )

        except Exception as exc:

            raise RuntimeError(
                "Google Drive folder could not be downloaded. "
                "Make sure the folder is public: "
                "Anyone with the link → Viewer."
            ) from exc

        if isinstance(result, list):

            candidate_paths = [
                Path(item)
                for item in result
                if item
            ]

        else:

            candidate_paths = [
                path
                for path in temp_dir.rglob("*")
                if path.is_file()
            ]

        for path in candidate_paths:

            try:

                file_bytes = path.read_bytes()

            except Exception:

                continue

            filename = make_filename_with_extension(
                path.name,
                file_bytes
            )

            extension = Path(
                filename
            ).suffix.lower()

            if extension in SUPPORTED_EXTENSIONS:

                downloaded_files.append(
                    (
                        filename,
                        file_bytes
                    )
                )

    # --------------------------------------------------------
    # GOOGLE DRIVE SINGLE FILE
    # --------------------------------------------------------

    else:

        file_id = extract_drive_file_id(
            url
        )

        if not file_id:

            raise ValueError(
                "Could not find a Google Drive file ID. "
                "Please paste a valid Google Drive file link."
            )

        output_path = (
            temp_dir / "drive_download"
        )

        try:

            downloaded = gdown.download(
                id=file_id,
                output=str(output_path),
                quiet=True,
                use_cookies=False
            )

        except TypeError:

            downloaded = gdown.download(
                id=file_id,
                output=str(output_path),
                quiet=True
            )

        except Exception as exc:

            raise RuntimeError(
                "Google Drive file could not be downloaded. "
                "Make sure it is public: "
                "Anyone with the link → Viewer."
            ) from exc

        if not downloaded:

            raise RuntimeError(
                "Google Drive did not return a downloadable file."
            )

        downloaded_path = Path(
            downloaded
        )

        if not downloaded_path.exists():

            downloaded_path = output_path

        if not downloaded_path.exists():

            raise RuntimeError(
                "The Google Drive download could not be located."
            )

        file_bytes = downloaded_path.read_bytes()

        filename = make_filename_with_extension(
            downloaded_path.name,
            file_bytes
        )

        extension = Path(
            filename
        ).suffix.lower()

        if extension in SUPPORTED_EXTENSIONS:

            downloaded_files.append(
                (
                    filename,
                    file_bytes
                )
            )

    if not downloaded_files:

        raise RuntimeError(
            "No supported PDF, DOCX, TXT or MD files were found. "
            "Make sure the Google Drive file/folder is public and "
            "contains a supported document."
        )

    return downloaded_files


# ============================================================
# GROQ
# ============================================================

def get_groq_client():

    if "GROQ_API_KEY" not in st.secrets:

        return None

    api_key = st.secrets[
        "GROQ_API_KEY"
    ]

    if not api_key:

        return None

    return Groq(
        api_key=api_key
    )


def ask_groq(
    question,
    retrieved_chunks
):

    client = get_groq_client()

    if client is None:

        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets."
        )

    if not retrieved_chunks:

        return (
            "I could not find this information "
            "in the provided documents."
        )

    context_parts = []

    for number, item in enumerate(
        retrieved_chunks,
        start=1
    ):

        if item["page"] is not None:

            page_text = (
                f"Page {item['page']}"
            )

        else:

            page_text = (
                "Page not available"
            )

        context_parts.append(
            f"""
SOURCE {number}
Filename: {item['filename']}
{page_text}

Content:
{item['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    system_prompt = """
You are an AI Document Assistant.

Answer the user's question using ONLY the information
in the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not supported by the context, say exactly:
"I could not find this information in the provided documents."
4. Keep the answer clear and easy to understand.
5. When possible, mention the relevant filename and page number.
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
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.0
    )

    return response.choices[
        0
    ].message.content.strip()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    chunk_size = st.slider(
        "Chunk size",
        min_value=400,
        max_value=2000,
        value=DEFAULT_CHUNK_SIZE,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=DEFAULT_CHUNK_OVERLAP,
        step=50
    )

    top_k = st.slider(
        "Retrieved sources",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K
    )

    st.divider()

    if st.button(
        "🗑️ Clear all documents",
        use_container_width=True
    ):

        st.session_state.documents = []
        st.session_state.chunks = []
        st.session_state.embeddings = None
        st.session_state.index = None
        st.session_state.document_keys = set()
        st.session_state.last_question = ""
        st.session_state.last_answer = ""
        st.session_state.last_sources = []

        st.success(
            "All documents cleared."
        )

        st.rerun()


# ============================================================
# DOCUMENT INPUT
# ============================================================

st.subheader(
    "1️⃣ Add Documents"
)

tab_local, tab_drive = st.tabs(
    [
        "📁 Local Upload",
        "☁️ Google Drive"
    ]
)


# ============================================================
# LOCAL UPLOAD
# ============================================================

with tab_local:

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD files",
        type=[
            "pdf",
            "docx",
            "txt",
            "md"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button(
            "➕ Process uploaded documents",
            type="primary"
        ):

            progress = st.progress(
                0
            )

            total = len(
                uploaded_files
            )

            for position, uploaded_file in enumerate(
                uploaded_files,
                start=1
            ):

                try:

                    file_bytes = (
                        uploaded_file.getvalue()
                    )

                    success, message = process_file(
                        file_bytes,
                        uploaded_file.name,
                        chunk_size,
                        chunk_overlap
                    )

                    if success:

                        st.success(
                            message
                        )

                    else:

                        st.info(
                            message
                        )

                except Exception as exc:

                    st.error(
                        f"Error processing "
                        f"{uploaded_file.name}: {exc}"
                    )

                progress.progress(
                    position / total
                )


# ============================================================
# GOOGLE DRIVE
# ============================================================

with tab_drive:

    st.info(
        "The Google Drive file or folder must be public: "
        "Anyone with the link → Viewer."
    )

    drive_url = st.text_input(
        "Paste Google Drive file or folder link",
        placeholder=(
            "https://drive.google.com/file/d/..."
        )
    )

    if st.button(
        "☁️ Load from Google Drive",
        type="primary"
    ):

        if not drive_url.strip():

            st.warning(
                "Please paste a Google Drive link."
            )

        else:

            with st.spinner(
                "Downloading from Google Drive..."
            ):

                try:

                    drive_files = (
                        download_drive_files(
                            drive_url
                        )
                    )

                    for filename, file_bytes in drive_files:

                        try:

                            success, message = process_file(
                                file_bytes,
                                filename,
                                chunk_size,
                                chunk_overlap
                            )

                            if success:

                                st.success(
                                    message
                                )

                            else:

                                st.info(
                                    message
                                )

                        except Exception as exc:

                            st.error(
                                f"Error processing "
                                f"{filename}: {exc}"
                            )

                except Exception as exc:

                    st.error(
                        f"Google Drive loading failed: {exc}"
                    )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

st.subheader(
    "2️⃣ Document Information"
)

if st.session_state.documents:

    for document in (
        st.session_state.documents
    ):

        st.write(
            f"📄 **{document['filename']}** — "
            f"{document['pages']} page/section(s), "
            f"{document['chunks']} chunk(s)"
        )

    st.metric(
        "Total chunks",
        len(
            st.session_state.chunks
        )
    )

else:

    st.info(
        "No documents loaded yet."
    )


# ============================================================
# QUESTION
# ============================================================

st.subheader(
    "3️⃣ Ask a Question"
)

question = st.text_input(
    "Enter your question",
    placeholder=(
        "Ask something about your uploaded documents..."
    )
)


if st.button(
    "🔎 Ask Question",
    type="primary",
    disabled=not bool(
        st.session_state.chunks
    )
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching documents..."
        ):

            retrieved = hybrid_search(
                question,
                top_k=top_k
            )

        if not retrieved:

            st.warning(
                "No relevant document content was found."
            )

        else:

            with st.spinner(
                "Generating answer..."
            ):

                try:

                    answer = ask_groq(
                        question,
                        retrieved
                    )

                    st.session_state.last_question = (
                        question
                    )

                    st.session_state.last_answer = (
                        answer
                    )

                    st.session_state.last_sources = (
                        retrieved
                    )

                except Exception as exc:

                    st.error(
                        f"Groq error: {exc}"
                    )


# ============================================================
# ANSWER + SOURCES
# ============================================================

if st.session_state.last_answer:

    st.subheader(
        "4️⃣ Answer"
    )

    st.write(
        st.session_state.last_answer
    )

    st.subheader(
        "📌 Retrieved Sources"
    )

    for number, source in enumerate(
        st.session_state.last_sources,
        start=1
    ):

        if source["page"] is not None:

            page_text = str(
                source["page"]
            )

        else:

            page_text = "Not available"

        with st.expander(
            f"Source {number}: "
            f"{source['filename']} | "
            f"Page: {page_text}"
        ):

            st.write(
                f"**Filename:** "
                f"{source['filename']}"
            )

            st.write(
                f"**Page:** "
                f"{page_text}"
            )

            st.write(
                f"**Hybrid score:** "
                f"{source['hybrid_score']:.3f}"
            )

            st.write(
                f"**Semantic score:** "
                f"{source['semantic_score']:.3f}"
            )

            st.write(
                f"**Keyword score:** "
                f"{source['keyword_score']:.3f}"
            )

            st.markdown(
                "**Retrieved text:**"
            )

            st.write(
                source["text"]
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Pipeline: Document → Extraction → Chunking → "
    "Sentence Transformer → FAISS → Hybrid Search → Groq"
)
