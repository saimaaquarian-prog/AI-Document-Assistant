# AI Document Assistant

A beginner-friendly Streamlit RAG application for asking questions about PDF, DOCX, TXT and Markdown documents.

## Features

- Upload PDF documents
- Upload DOCX documents
- Upload TXT documents
- Upload Markdown files
- Load public Google Drive files
- Load public Google Drive folders
- Extract document text
- Preserve filename
- Preserve PDF page numbers
- Create overlapping text chunks
- Generate Sentence Transformer embeddings
- Store embeddings in FAISS
- Semantic vector search
- Keyword search
- Hybrid semantic + keyword search
- Groq LLM integration
- Retrieved source chunks shown after every answer
- Streamlit session-state optimization
- Duplicate document protection
- Groq API key stored only in Streamlit Secrets

## Project Structure

```text
ai-document-assistant/
│
├── app.py
├── requirements.txt
└── README.md
