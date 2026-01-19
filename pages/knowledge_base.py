"""
Knowledge Base Page for Streamlit UI
=====================================

This module provides the Knowledge Base (RAG) management tab for the Streamlit application.
Users can upload documents, view statistics, and manage the knowledge base.
"""

import streamlit as st
import os
import tempfile
from datetime import datetime

# Import RAG functions
from tools.rag_tools import (
    ingest_document_internal,
    remove_document_internal,
    _reset_rag_store,
    get_rag_store_state
)


def render_knowledge_base():
    """Render the knowledge base management section in Streamlit."""

    st.header("📚 Knowledge Base (RAG)")
    st.markdown("Upload documents to create a knowledge base. The system will automatically use relevant information from these documents when answering your queries.")

    # Get current RAG state
    rag_state = get_rag_store_state()
    rag_docs = rag_state["documents"]
    rag_metadata = rag_state["metadata"]

    # Statistics Section
    st.subheader("📊 Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", len(rag_docs))
    with col2:
        st.metric("Chunks", rag_metadata["total_chunks"])
    with col3:
        st.metric("Tokens", f"{rag_metadata['total_tokens']:,}")

    if rag_metadata["last_updated"]:
        st.caption(f"Last updated: {rag_metadata['last_updated']}")

    st.markdown("---")

    # Upload Section
    st.subheader("📤 Upload Documents")
    st.markdown("Supported formats: **PDF**, **TXT**, **DOCX**, **CSV**, **MD**, **JSON**")

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "txt", "docx", "csv", "md", "json"],
        accept_multiple_files=True,
        key="kb_uploader",
        help="Upload one or more documents to add to the knowledge base"
    )

    if uploaded_files:
        st.markdown("**Files ready for ingestion:**")

        for uploaded_file in uploaded_files:
            # Check if already in knowledge base
            already_exists = uploaded_file.name in rag_docs

            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                icon = "✅" if already_exists else "📄"
                status = " (already loaded)" if already_exists else ""
                st.markdown(f"{icon} **{uploaded_file.name}**{status}")
            with col2:
                st.caption(f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                if not already_exists:
                    if st.button("Ingest", key=f"ingest_{uploaded_file.name}", type="primary"):
                        # Save file temporarily
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=os.path.splitext(uploaded_file.name)[1]
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        # Ingest document
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            result = ingest_document_internal(tmp_path)

                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                        if result["success"]:
                            st.success(f"Successfully ingested '{uploaded_file.name}': {result['chunks']} chunks, {result['tokens']:,} tokens")
                            st.rerun()
                        else:
                            st.error(f"Failed to ingest: {result['error']}")
                else:
                    st.caption("Loaded")

        # Ingest All button
        unprocessed = [f for f in uploaded_files if f.name not in rag_docs]
        if len(unprocessed) > 1:
            st.markdown("---")
            if st.button(f"📥 Ingest All ({len(unprocessed)} files)", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                success_count = 0
                for i, uploaded_file in enumerate(unprocessed):
                    status_text.text(f"Processing {uploaded_file.name}...")

                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(uploaded_file.name)[1]
                    ) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    result = ingest_document_internal(tmp_path)

                    # Clean up
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

                    if result["success"]:
                        success_count += 1

                    progress_bar.progress((i + 1) / len(unprocessed))

                status_text.text(f"Completed: {success_count}/{len(unprocessed)} files ingested successfully")
                st.rerun()

    st.markdown("---")

    # Loaded Documents Section
    st.subheader("📁 Loaded Documents")

    if rag_docs:
        for doc_name, doc_info in rag_docs.items():
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.markdown(f"**📄 {doc_name}**")
                    st.caption(f"Type: {doc_info['doc_type'].upper()}")
                with col2:
                    st.metric("Chunks", doc_info['chunks'], label_visibility="collapsed")
                    st.caption("chunks")
                with col3:
                    st.metric("Tokens", f"{doc_info['tokens']:,}", label_visibility="collapsed")
                    st.caption("tokens")
                with col4:
                    if st.button("🗑️ Remove", key=f"remove_{doc_name}"):
                        result = remove_document_internal(doc_name)
                        if result["success"]:
                            st.success(f"Removed '{doc_name}'")
                            st.rerun()
                        else:
                            st.error(f"Failed to remove: {result['error']}")

                st.markdown("---")

        # Clear All button
        st.markdown("")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
                doc_count = len(rag_docs)
                _reset_rag_store()
                st.success(f"Cleared {doc_count} document(s) from knowledge base")
                st.rerun()
    else:
        st.info("No documents loaded yet. Upload files above to get started.")

    st.markdown("---")

    # How It Works Section
    with st.expander("💡 How It Works", expanded=False):
        st.markdown("""
        ### Automatic Context Injection

        When you ask a question in the chat:
        1. **Semantic Search**: Your query is compared against all document chunks using embeddings
        2. **Relevance Matching**: The most relevant chunks are identified based on similarity
        3. **Context Injection**: Relevant content is automatically added to the prompt as context
        4. **Smart Response**: The AI uses this context to provide more accurate, informed answers

        ### Configuration
        - **Chunk Size**: 1000 characters (with 200 char overlap)
        - **Embedding Model**: OpenAI text-embedding-3-small
        - **Top Results**: Up to 5 most relevant chunks per query
        - **Minimum Score**: Only chunks with relevance score > 0.3 are included

        ### Tips for Best Results
        - Upload documents with clear, well-structured content
        - PDFs with extractable text work best (not scanned images)
        - Break very large documents into smaller, focused files
        - Use descriptive filenames for easy management
        """)

    # Supported Formats Section
    with st.expander("📋 Supported Formats", expanded=False):
        st.markdown("""
        | Format | Extension | Notes |
        |--------|-----------|-------|
        | PDF | `.pdf` | Text-based PDFs only (not scanned images) |
        | Plain Text | `.txt` | UTF-8 encoded |
        | Word Document | `.docx` | Microsoft Word format |
        | CSV | `.csv` | Comma-separated values |
        | Markdown | `.md` | Markdown formatted text |
        | JSON | `.json` | JSON data (converted to text) |
        """)


if __name__ == "__main__":
    # For testing
    st.set_page_config(page_title="Knowledge Base", layout="wide")
    render_knowledge_base()
