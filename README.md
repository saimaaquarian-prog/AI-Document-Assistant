📚 AI Document Assistant

A simple Streamlit RAG application that lets users upload documents, build a searchable knowledge base, and ask questions using Groq.

Features

PDF, DOCX, TXT and MD upload

Separate extraction functions for each document type

PDF page numbers are preserved

DOCX/TXT/MD documents use Page not available

Text chunking with overlap

Sentence Transformers embeddings

FAISS vector search

Simple keyword search

Hybrid semantic + keyword ranking

Retrieved filename and page metadata

Groq-powered question answering

Answers are restricted to retrieved document context

Retrieved source chunks are displayed below every answer

Public Google Drive file/folder loading

Local uploads and Google Drive use the same pipeline

Embeddings are created only when a new document is processed

Streamlit session state keeps the FAISS index and embeddings available between questions

Embedding model is cached with st.cache_resource

Groq API key is stored in Streamlit Secrets

Architecture

Local PDF/DOCX/TXT/MD
          │
          ▼
   Text Extraction
          │
          ▼
       Chunking
          │
          ▼
Sentence Transformer
      Embeddings
          │
          ▼
       FAISS Index
          │
          │
Google Drive ──► Same Pipeline
          │
          ▼
     User Question
          │
     ┌────┴────┐
     ▼         ▼
Semantic    Keyword
 Search      Search
     └────┬────┘
          ▼
     Hybrid Ranking
          │
          ▼
   Relevant Chunks
          │
          ▼
        Groq
          │
          ▼
      Final Answer
          │
          ▼
    Retrieved Sources

Project files

ai-document-assistant/
│
├── app.py
├── requirements.txt
└── README.md

Run locally

Install the packages:

pip install -r requirements.txt

Create:

.streamlit/secrets.toml

Add your Groq key:

GROQ_API_KEY = "your-groq-api-key"

Run:

streamlit run app.py

Streamlit Cloud deployment

1. Create a GitHub repository

Create a new GitHub repository and upload:

app.py

requirements.txt

README.md

2. Create the Streamlit app

Open Streamlit Community Cloud and connect your GitHub repository.

Select:

Main file: app.py

Deploy the application.

3. Add the Groq API key

In Streamlit Cloud:

App
→ Settings
→ Secrets

Add:

GROQ_API_KEY = "your-groq-api-key"

Save the secret and restart/redeploy the app.

The API key is never hardcoded in app.py.

Google Drive

Paste a publicly accessible Google Drive file or folder link.

Supported files:

.pdf

.docx

.txt

.md

The app uses gdown to download public Drive files/folders and then sends them through the same extraction, chunking, embedding, and indexing pipeline.

Google Drive folders have practical limits when using gdown; keep folders reasonably sized for a simple Streamlit deployment.

How the RAG pipeline works

1. Extraction

Each document is converted into text.

For PDF files, the page number is stored with the extracted text.

2. Chunking

Long text is split into smaller overlapping chunks.

Default:

Chunk size: 900 characters
Overlap: 150 characters

Overlap helps preserve information that crosses chunk boundaries.

3. Embeddings

Each chunk is converted into a vector using:

sentence-transformers
all-MiniLM-L6-v2

The vectors are normalized and stored in FAISS.

4. Semantic search

The user's question is also converted into an embedding.

FAISS uses inner-product similarity to find semantically similar chunks.

5. Keyword search

Important words from the question are compared with words in each chunk.

This helps when an exact term appears in the document but semantic similarity alone might not rank it highly.

6. Hybrid search

The app combines:

70% semantic similarity
30% keyword matching

The highest-ranked chunks are sent to Groq.

7. Grounded answer

Groq receives:

the user question

the retrieved document chunks

The system prompt tells the model to answer only from that context.

If the information is not available, the assistant says:

I could not find this information in the provided documents.

Why embeddings are not recreated for every question

Document embeddings are created only when a new document is processed.

The application keeps:

document metadata

chunks

embeddings

FAISS index

in Streamlit session state.

The Sentence Transformer model is also cached with:

@st.cache_resource

When a question is asked, only the question embedding is generated. The existing document embeddings are reused.

Important Google Drive note

The Drive link must be accessible to the application, such as a publicly shared file/folder. Private files that require a Google login are not handled by this simple version.

Security

Never put your Groq API key directly in app.py.

Use Streamlit Secrets:

GROQ_API_KEY = "your-groq-api-key"

Also add .streamlit/secrets.toml to .gitignore if you run the project locally.

.gitignore

If you later add a .gitignore, include:

.streamlit/secrets.toml
__pycache__/
*.pyc
