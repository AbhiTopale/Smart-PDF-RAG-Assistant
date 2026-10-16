# 📚 Smart PDF RAG Assistant

A complete, clean, and production-ready **Retrieval-Augmented Generation (RAG)** web application designed for academic research, study materials, and college lab demonstrations.

The application allows users to upload academic/study PDF documents, upload an image containing a question (printed or handwritten), extract the question via **OpenCV + Tesseract OCR**, retrieve relevant context through **Sentence Transformers + FAISS Vector Search**, and synthesize factual, non-hallucinatory answers with **verifiable page references** using the high-speed **Groq LLM API**.

---

## 🌟 Key Features

- **Real Page-by-Page PDF Text Extraction:** Preserves exact physical page numbers (`Page 1, 2, ...`) for academic citation and provenance.
- **Configurable Overlapping Chunking:** Tunable chunk size (default: 800 characters) and chunk overlap (default: 150 characters) to prevent cutting context at chunk boundaries.
- **Real Embedding Generation:** Uses Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense semantic vectors.
- **In-Memory FAISS Vector Database:** Employs `faiss-cpu` (`IndexFlatL2`) for fast nearest-neighbor similarity search.
- **Multimodal OCR Pipeline:** Preprocesses question images (Grayscale, CLAHE contrast enhancement, bilateral filtering) with OpenCV before running Tesseract OCR.
- **Editable Question Interface:** Allows the user to inspect and edit the OCR-extracted question text before triggering RAG retrieval.
- **Strict Non-Hallucinatory LLM Answering:** Custom system prompt ensures Groq answers strictly from retrieved context and reports *"The answer was not found in the uploaded PDF"* for out-of-domain queries.
- **Multi-Turn Document Session:** Maintains vector index in Streamlit session state so multiple questions can be asked without re-processing the PDF.
- **College Lab Viva Inspector:** Built-in interactive explanation sections showing the RAG pipeline flowchart and the exact raw context delivered to the LLM prompt.

---

## 🏗️ System Architecture

```
                       ┌──────────────────────────┐
                       │   Uploaded Study PDF     │
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │  pypdf Text Extraction   │ (Preserves page numbers)
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │   Overlapping Chunker    │ (800 chars / 150 overlap)
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │  Sentence Transformers   │ (all-MiniLM-L6-v2, 384D)
                       └────────────┬─────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │   FAISS Vector Store     │ (IndexFlatL2)
                       └────────────▲─────────────┘
                                    │ Similarity Search (Top-K)
┌────────────────────────┐          │
│ Uploaded Question Img  │          │
└───────────┬────────────┘          │
            │                       │
            ▼                       │
┌────────────────────────┐          │
│ OpenCV Preprocessing   │          │
│ (Grayscale + CLAHE)    │          │
└───────────┬────────────┘          │
            │                       │
            ▼                       │
┌────────────────────────┐          │
│     Tesseract OCR      │          │
└───────────┬────────────┘          │
            │                       │
            ▼                       │
┌────────────────────────┐          │
│ User Review / Edit     │          │
└───────────┬────────────┘          │
            │                       │
            ▼                       │
┌────────────────────────┐          │
│ Query Vector Embedding ├──────────┘
└────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────┐
│ Assemble Retrieved Context with Source Page Numbers    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Groq LLM (e.g. qwen/qwen3.8-27b / gpt-oss-120b)       │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Final Answer + Verifiable Source Page References       │
└────────────────────────────────────────────────────────┘
```

---

## 💻 Technology Stack

| Component | Library / Service | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | `streamlit` | Clean, modern interactive web interface |
| **PDF Extraction**| `pypdf` | Page-by-page text parsing with metadata |
| **Embeddings** | `sentence-transformers` | Semantic dense vector generation (`all-MiniLM-L6-v2`) |
| **Vector DB** | `faiss-cpu` | High-performance vector similarity search |
| **Computer Vision**| `opencv-python`, `Pillow` | Image normalization, contrast enhancement (CLAHE) |
| **OCR** | `pytesseract` | Optical character recognition for question images |
| **LLM Inference** | `groq` | Ultra-fast inference via Groq Cloud API |
| **Configuration** | `python-dotenv` | Secure API key and environment management |

---

## 📁 Project Structure

```text
RAG/
├── app.py                      # Main Streamlit web application
├── requirements.txt            # Project Python dependencies
├── .env                        # Local environment variables (DO NOT commit)
├── .env.example                # Template for environment configuration
├── .gitignore                  # Git ignore rules (.env, venv, caches)
├── README.md                   # Complete documentation & viva guide
├── test_pipeline.py            # Automated end-to-end integration test
├── create_sample_assets.py     # Script generating academic test assets
│
├── rag/                        # RAG core modules
│   ├── __init__.py
│   ├── pdf_processor.py        # PDF extraction & error handling
│   ├── chunker.py              # Overlapping chunking preserving page numbers
│   ├── embeddings.py           # Sentence-transformers embedding wrapper
│   ├── vector_store.py         # FAISS IndexFlatL2 vector database
│   ├── retriever.py            # Nearest-neighbor search & context builder
│   ├── llm.py                  # Groq API client with strict system prompt
│   └── pipeline.py             # End-to-end coordinator
│
├── ocr/                        # OCR & Image Processing modules
│   ├── __init__.py
│   └── image_ocr.py            # OpenCV preprocessing & pytesseract extraction
│
├── utils/                      # Helper utilities
│   ├── __init__.py
│   └── helpers.py              # Text cleaning, validation & formatting
│
└── sample_documents/           # Sample test files for demonstration
    ├── machine_learning_notes.pdf
    ├── question_pooling_layers.png
    └── question_supervised_vs_unsupervised.png
```

---

## 🚀 Setup & Installation (Windows)

### 1. Clone or Open the Project
Open PowerShell or Command Prompt in the project directory:
```powershell
cd c:\Users\abhij\OneDrive\Desktop\RAG
```

### 2. Create and Activate a Python Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

---

## 👁️ Tesseract OCR Installation (Windows)

For OCR image extraction to read questions directly from uploaded images:

1. Download the official Windows installer from UB-Mannheim:
   - **Download Link:** [tesseract-ocr-w64-setup-5.4.0.20240606.exe](https://github.com/UB-Mannheim/tesseract/releases/download/v5.4.0.20240606/tesseract-ocr-w64-setup-5.4.0.20240606.exe)
2. Run the installer and install to the default directory:
   `C:\Program Files\Tesseract-OCR`
3. The application will **automatically detect** Tesseract at:
   - System PATH
   - `C:\Program Files\Tesseract-OCR\tesseract.exe`
   - `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
   - Or custom path configured via `TESSERACT_PATH` in `.env`.

> **Note on OCR Fallback:** If Tesseract is not installed on the system, the application handles it gracefully by displaying a helpful reminder and still allows the user to review, type, or edit the question in the text box!

---

## 🔑 How to Configure GROQ_API_KEY

1. Copy `.env.example` to create your `.env` file (if not already created):
   ```powershell
   copy .env.example .env
   ```
2. Open `.env` and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=qwen/qwen3.8-27b
   ```
3. **Security Guarantee:**
   - The `.env` file is excluded from Git via `.gitignore`.
   - The API key is **never displayed** in the Streamlit UI, logs, or error messages.

---

## 🖥️ Running the Application

Ensure your virtual environment is active, then launch the Streamlit app:
```powershell
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 🧪 Testing with Sample Assets

Sample demonstration files are provided in the `sample_documents/` folder:

1. **Step 1: Upload PDF**
   - Click "Browse files" in **Section 1** and select `sample_documents/machine_learning_notes.pdf`.
   - Watch the progress bar index the 4 pages into 8 vector chunks in FAISS.
2. **Step 2: Upload Question Image**
   - In **Section 2**, upload `sample_documents/question_pooling_layers.png`.
   - The question *"What is the function of pooling layers in Convolutional Neural Networks?"* appears in the editable text box.
3. **Step 3: Click Search & Answer**
   - Click **"🔍 Search & Answer"**.
   - Groq synthesizes a concise response citing **Page 3**.
   - Expand the **📚 Sources** section to verify the exact matched chunk and similarity score.
4. **Step 4: Out-of-Domain Verification**
   - Type in the question box: *"What is the capital of Japan?"*
   - Click **Search & Answer**.
   - The model strictly replies: *"The answer was not found in the uploaded PDF."*

---

## 🎓 College Viva & Demonstration Guide

### Why Use RAG Instead of Fine-Tuning?
- **No Retraining Needed:** Simply upload new documents; the vector database indexes them immediately.
- **Zero Hallucination:** Answers are anchored strictly in retrieved passages.
- **Source Verification:** Every answer is backed by exact document page numbers.

### Why FAISS IndexFlatL2?
- `IndexFlatL2` computes the exact Euclidean distance between the question vector and document chunk vectors.
- Since all embeddings are L2-normalized upon creation, smaller L2 distance directly equates to higher cosine similarity ($L_2^2 = 2 - 2 \cdot \cos(\theta)$).

### Why Preprocess Images Before OCR?
- Photos taken with mobile phones often have uneven shadows or low contrast.
- Converting to grayscale and applying **CLAHE (Contrast Limited Adaptive Histogram Equalization)** sharpens the text edges and maximizes OCR recognition accuracy.

---

## 🛠️ Common Errors and Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `Groq API Key is missing` | `.env` file not created or key missing | Create `.env` in root with `GROQ_API_KEY=...` |
| `Tesseract OCR is not installed` | Tesseract binary not found on Windows | Install Tesseract from UB-Mannheim or set `TESSERACT_PATH` in `.env` |
| `Scanned document warning` | PDF contains only scanned image pages without text | Upload a digital/searchable text PDF |
| `Rate limit exceeded` | Free-tier Groq API calls exceeded | Wait 30 seconds or choose another model in the sidebar dropdown |
| `ModuleNotFoundError` | Virtual environment not activated | Run `venv\Scripts\activate` and `pip install -r requirements.txt` |
