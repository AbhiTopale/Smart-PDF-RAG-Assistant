import os
import time
from typing import Optional
from dotenv import load_dotenv
import streamlit as st
from PIL import Image

from rag.pipeline import RAGPipeline
from rag.pdf_processor import EmptyPDFError, ScannedPDFError, CorruptedPDFError, PDFProcessingError
from ocr.image_ocr import extract_question_from_image, is_tesseract_available
from utils.helpers import validate_pdf_file, validate_image_file, format_similarity_score

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, polished academic interface
st.markdown("""
<style>
    /* Global Typography & spacing */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .status-pill-ok {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-pill-wait {
        background-color: #F1F5F9;
        color: #64748B;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .answer-box {
        background: linear-gradient(135deg, #F0FDF4 0%, #FFFFFF 100%);
        border: 1px solid #BBF7D0;
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .source-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .rag-pipeline-step {
        display: inline-block;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        padding: 4px 10px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.85rem;
        margin: 3px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

if "processed_pdf_name" not in st.session_state:
    st.session_state.processed_pdf_name = None

if "pdf_metrics" not in st.session_state:
    st.session_state.pdf_metrics = None

if "extracted_question" not in st.session_state:
    st.session_state.extracted_question = ""

if "ocr_preview_img" not in st.session_state:
    st.session_state.ocr_preview_img = None

if "history" not in st.session_state:
    st.session_state.history = []

pipeline: RAGPipeline = st.session_state.rag_pipeline


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")

    st.markdown("#### 1. RAG Hyperparameters")
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=200,
        max_value=2000,
        value=800,
        step=50,
        help="Target length of each document segment."
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=500,
        value=150,
        step=25,
        help="Character overlap between consecutive chunks to preserve contextual continuity."
    )
    top_k = st.slider(
        "Top K Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of most similar chunks retrieved from FAISS."
    )

    st.markdown("#### 2. LLM Model Settings")
    default_groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    available_models = [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "allam-2-7b"
    ]
    if default_groq_model not in available_models:
        available_models.insert(0, default_groq_model)

    selected_model = st.selectbox(
        "Groq LLM Model",
        options=available_models,
        index=0,
        help="Active model on your Groq API account."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values produce more deterministic, factual answers based strictly on context."
    )

    st.markdown("---")
    st.markdown("#### 3. System Status")

    # API Key check (without displaying key value)
    has_api_key = pipeline.llm.is_configured()
    tesseract_available = is_tesseract_available()

    col_s1, col_s2 = st.columns([3, 2])
    with col_s1:
        st.write("Groq API Key:")
    with col_s2:
        if has_api_key:
            st.markdown('<span class="status-pill-ok">Configured ✓</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill-wait">Missing ✗</span>', unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.write("Tesseract OCR:")
    with col_t2:
        if tesseract_available:
            st.markdown('<span class="status-pill-ok">Detected ✓</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill-wait">Not Found ℹ</span>', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns([3, 2])
    with col_p1:
        st.write("PDF Loaded:")
    with col_p2:
        if pipeline.is_ready:
            st.markdown('<span class="status-pill-ok">Yes ✓</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill-wait">No ✗</span>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([3, 2])
    with col_f1:
        st.write("FAISS Index:")
    with col_f2:
        if pipeline.vector_store.is_initialized:
            st.markdown(f'<span class="status-pill-ok">{pipeline.vector_store.total_vectors} vectors</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill-wait">Not Built</span>', unsafe_allow_html=True)

    if pipeline.is_ready and st.session_state.pdf_metrics:
        st.markdown("---")
        st.markdown("#### 📊 Active Document")
        m = st.session_state.pdf_metrics
        st.caption(f"**File:** {m.get('filename')}")
        st.caption(f"**Pages:** {m.get('pages_count')} | **Chunks:** {m.get('chunks_count')}")
        st.caption(f"**Embeddings:** {m.get('dimension')}D (all-MiniLM-L6-v2)")
        if st.button("🗑️ Reset Document & Index", use_container_width=True):
            pipeline.reset()
            st.session_state.processed_pdf_name = None
            st.session_state.pdf_metrics = None
            st.session_state.history = []
            st.session_state.extracted_question = ""
            st.session_state.ocr_preview_img = None
            st.rerun()


# ==========================================
# HEADER
# ==========================================
st.markdown('<div class="main-title">📚 Smart PDF RAG Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload academic or study PDF documents, extract questions from images using OCR, '
    'and retrieve precise answers with verifiable page references via Groq LLM.</div>',
    unsafe_allow_html=True
)

if not has_api_key:
    st.error(
        "⚠️ **Groq API Key is not configured.** Please create a `.env` file in the project root with "
        "`GROQ_API_KEY=your_key_here` or configure your environment variables. Then restart the application."
    )


# ==========================================
# SECTION 1: UPLOAD KNOWLEDGE PDF
# ==========================================
st.markdown("### 📄 Section 1: Upload Knowledge PDF")

uploaded_pdf = st.file_uploader(
    "Choose a PDF document containing study or research material:",
    type=["pdf"],
    key="pdf_uploader",
    help="Upload an academic textbook chapter, research paper, or lecture notes."
)

if uploaded_pdf is not None:
    # Check if this PDF has already been processed in current session
    is_new_file = (st.session_state.processed_pdf_name != uploaded_pdf.name)

    if is_new_file:
        is_valid, validation_msg = validate_pdf_file(uploaded_pdf)
        if not is_valid:
            st.error(f"❌ {validation_msg}")
        else:
            status_container = st.status("🔄 Processing PDF Document through RAG Pipeline...", expanded=True)
            progress_bar = st.progress(0.0)

            try:
                def on_progress(msg: str, progress_float: float):
                    status_container.write(f"• {msg}")
                    progress_bar.progress(progress_float)

                metrics = pipeline.process_pdf(
                    file_input=uploaded_pdf,
                    filename=uploaded_pdf.name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    progress_callback=on_progress
                )

                st.session_state.processed_pdf_name = uploaded_pdf.name
                st.session_state.pdf_metrics = metrics

                status_container.update(
                    label=f"✅ Successfully indexed '{uploaded_pdf.name}' in {metrics['elapsed_time']}s!",
                    state="complete",
                    expanded=False
                )
                progress_bar.empty()
                st.rerun()

            except EmptyPDFError as e:
                status_container.update(label="Failed", state="error")
                progress_bar.empty()
                st.error(f"❌ Empty PDF Error: {str(e)}")
            except ScannedPDFError as e:
                status_container.update(label="Failed", state="error")
                progress_bar.empty()
                st.error(f"⚠️ Scanned Document Warning: {str(e)}")
            except CorruptedPDFError as e:
                status_container.update(label="Failed", state="error")
                progress_bar.empty()
                st.error(f"❌ Corrupted File: {str(e)}")
            except PDFProcessingError as e:
                status_container.update(label="Failed", state="error")
                progress_bar.empty()
                st.error(f"❌ PDF Processing Error: {str(e)}")
            except Exception as e:
                status_container.update(label="Failed", state="error")
                progress_bar.empty()
                st.error(f"❌ Unexpected Error: {str(e)}")

# Display active document confirmation banner
if pipeline.is_ready and st.session_state.pdf_metrics:
    m = st.session_state.pdf_metrics
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        st.metric(label="Document", value=m.get("filename", "")[:18] + ("..." if len(m.get("filename", "")) > 18 else ""))
    with col_b2:
        st.metric(label="Pages Processed", value=f"{m.get('pages_count')} pages")
    with col_b3:
        st.metric(label="Chunks Created", value=f"{m.get('chunks_count')} chunks")
    with col_b4:
        st.metric(label="Indexing Time", value=f"{m.get('elapsed_time')}s")

    st.success(f"✓ Document **{m.get('filename')}** is indexed in FAISS and ready for queries!")


# ==========================================
# SECTION 2: UPLOAD QUESTION IMAGE
# ==========================================
st.markdown("---")
st.markdown("### 🖼️ Section 2: Upload Question Image")

if not pipeline.is_ready:
    st.info("👆 Please upload and index a PDF in Section 1 before proceeding to question search.")

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    uploaded_image = st.file_uploader(
        "Upload an image containing your question (printed or handwritten):",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        key="image_uploader",
        help="Upload a screenshot, photo, or scan of an exam question or exercise."
    )

    if uploaded_image is not None:
        is_valid_img, img_err = validate_image_file(uploaded_image)
        if not is_valid_img:
            st.error(f"❌ {img_err}")
        else:
            # Check if image was changed or needs OCR extraction
            if "last_uploaded_img_name" not in st.session_state or st.session_state.last_uploaded_img_name != uploaded_image.name:
                with st.spinner("🔍 Running OCR with OpenCV preprocessing..."):
                    extracted_text, ocr_error, preview_pil = extract_question_from_image(uploaded_image)
                    st.session_state.last_uploaded_img_name = uploaded_image.name
                    st.session_state.ocr_preview_img = preview_pil

                    if ocr_error:
                        st.warning(f"ℹ️ {ocr_error}")
                    if extracted_text:
                        st.session_state.extracted_question = extracted_text

with col_preview:
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Question Image", use_container_width=True)

# Question Text Area (Editable)
st.markdown("#### ✏️ Extracted Question (Review or Edit Before Searching)")
st.caption("You can freely edit, refine, or manually type the question text below:")

current_question = st.text_area(
    label="Question Text:",
    value=st.session_state.extracted_question,
    height=90,
    key="question_input_area",
    placeholder="e.g., What is the difference between supervised and unsupervised learning?"
)

# Search & Answer button
col_btn, col_info = st.columns([1, 3])
with col_btn:
    search_clicked = st.button(
        "🔍 Search & Answer",
        type="primary",
        disabled=not pipeline.is_ready or not bool(current_question.strip()),
        use_container_width=True
    )
with col_info:
    if not pipeline.is_ready:
        st.caption("🔒 Disabled until a PDF is uploaded and indexed.")
    elif not current_question.strip():
        st.caption("✍️ Please provide or extract a question first.")


# ==========================================
# RAG RETRIEVAL & ANSWER GENERATION
# ==========================================
if search_clicked and current_question.strip():
    with st.spinner("🤖 Performing FAISS similarity search & generating answer with Groq LLM..."):
        try:
            start_q_time = time.time()
            answer, retrieved_sources, raw_context = pipeline.answer_question(
                question=current_question.strip(),
                top_k=top_k,
                temperature=temperature,
                model_override=selected_model
            )
            elapsed_q_time = round(time.time() - start_q_time, 2)

            # Record in session history
            history_entry = {
                "question": current_question.strip(),
                "answer": answer,
                "sources": retrieved_sources,
                "raw_context": raw_context,
                "time": elapsed_q_time,
                "model": selected_model
            }
            # Prepend to history so newest is on top
            st.session_state.history.insert(0, history_entry)

        except Exception as e:
            st.error(f"❌ Answer generation failed: {str(e)}")


# ==========================================
# DISPLAY LATEST RESULT
# ==========================================
if st.session_state.history:
    latest = st.session_state.history[0]

    st.markdown("---")
    st.markdown("### 🤖 Answer")

    st.markdown(f"""
    <div class="answer-box">
        <div style="font-size: 0.85rem; color: #15803D; font-weight: 600; margin-bottom: 8px;">
            Model: {latest['model']} • Generated in {latest['time']}s
        </div>
        <div style="font-size: 1.05rem; line-height: 1.6; color: #1E293B;">
            {latest['answer']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sources section
    st.markdown("### 📚 Sources & References")
    sources = latest.get("sources", [])

    if not sources:
        st.info("No matching source chunks found in the document.")
    else:
        for idx, src in enumerate(sources, start=1):
            page_no = src.get("page", "Unknown")
            source_file = src.get("source", "Document")
            dist = src.get("distance", 0.0)
            sim_str = format_similarity_score(dist)
            snippet = src.get("text", "")

            with st.expander(f"📄 Source #{idx}: Page {page_no} (Similarity: {sim_str}) — {source_file}"):
                st.markdown(f"**Document Page:** `{page_no}` | **L2 Distance:** `{dist:.4f}` | **Relevance:** `{sim_str}`")
                st.markdown(f"**Retrieved Content:**")
                st.code(snippet, language=None)

    # Retrieved Context Expandable Section (for viva demo)
    with st.expander("🔍 Viva Inspector: Exact Context Fed to LLM Prompt"):
        st.caption("During a lab viva or project review, expand this section to demonstrate the exact formatted context delivered to the Groq LLM:")
        st.text(latest.get("raw_context", "None"))


# ==========================================
# SECTION 3: QUESTION HISTORY
# ==========================================
if len(st.session_state.history) > 1:
    st.markdown("---")
    st.markdown("### 🕒 Previous Questions in this Session")
    st.caption("You can ask multiple questions on the same indexed PDF without re-indexing.")

    for h_idx, item in enumerate(st.session_state.history[1:], start=1):
        with st.expander(f"Q{h_idx}: {item['question']}"):
            st.markdown(f"**Answer:**\n{item['answer']}")
            st.caption(f"Retrieved {len(item.get('sources', []))} chunks using {item.get('model')} in {item.get('time')}s")


# ==========================================
# SECTION 4: EDUCATIONAL / VIVA EXPLANATION
# ==========================================
st.markdown("---")
with st.expander("⚙️ How RAG Works (College Lab Viva Guide)", expanded=False):
    st.markdown(r"""
    #### 🎓 Architecture & Pipeline Explanation

    This application implements an end-to-end **Retrieval-Augmented Generation (RAG)** architecture with Multimodal OCR Input:

    ```
    ┌──────────────────────────┐         ┌──────────────────────────┐
    │     1. User PDF          │         │  6. User Question Image  │
    └─────────────┬────────────┘         └─────────────┬────────────┘
                  │                                    │
                  ▼                                    ▼
    ┌──────────────────────────┐         ┌──────────────────────────┐
    │ 2. Text Extraction       │         │ 7. Preprocess & OCR      │
    │    (pypdf, per-page)     │         │    (OpenCV + pytesseract)│
    └─────────────┬────────────┘         └─────────────┬────────────┘
                  │                                    │
                  ▼                                    ▼
    ┌──────────────────────────┐         ┌──────────────────────────┐
    │ 3. Overlapping Chunking  │         │ 8. Question Text Review  │
    │    (800 chars / 150 ovlp)│         │    (User can edit OCR)   │
    └─────────────┬────────────┘         └─────────────┬────────────┘
                  │                                    │
                  ▼                                    ▼
    ┌──────────────────────────┐         ┌──────────────────────────┐
    │ 4. Embeddings Generation │         │ 9. Question Embedding    │
    │ (all-MiniLM-L6-v2, 384D) │         │ (all-MiniLM-L6-v2, 384D) │
    └─────────────┬────────────┘         └─────────────┬────────────┘
                  │                                    │
                  ▼                                    │
    ┌──────────────────────────┐                       │
    │ 5. FAISS Vector Store    │◄──────────────────────┘
    │    (IndexFlatL2 search)  │ (Similarity Search)
    └─────────────┬────────────┘
                  │
                  ▼ Top-K Chunks
    ┌──────────────────────────────────────────────────────────────┐
    │ 10. Context Assembly with Page Numbers                       │
    └─────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ 11. Groq LLM Generation (Strict Non-Hallucinatory Prompt)    │
    └─────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ 12. Final Answer + Verifiable Page References                │
    └──────────────────────────────────────────────────────────────┘
    ```

    #### 🔑 Key Viva Concepts:
    1. **Why not feed the whole PDF to LLM?**
       - Standard LLM context windows have limits and higher latency/cost. RAG retrieves only the most relevant passages.
    2. **Why preserve Page Numbers?**
       - Academic and research integrity requires provenance (traceability). We map every chunk back to its source page.
    3. **Why normalize embeddings before FAISS IndexFlatL2?**
       - L2 distance between normalized vectors has a direct monotonic relationship to Cosine Similarity ($L_2^2 = 2 - 2 \cdot \cos$).
    4. **Why Groq?**
       - Groq's LPU (Language Processing Unit) inference provides ultra-low latency response times.
    5. **Why editable OCR text?**
       - Real-world OCR from phone cameras can suffer from noise, uneven lighting, or handwriting quirks. Providing a review interface guarantees 100% retrieval accuracy.
    """)
