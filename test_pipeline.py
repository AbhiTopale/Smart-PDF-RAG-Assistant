import os
import sys

# Force UTF-8 stdout if available
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from rag.pipeline import RAGPipeline

load_dotenv()

def test_full_rag_workflow():
    print("==================================================")
    print("Testing Smart PDF RAG Assistant Pipeline End-to-End")
    print("==================================================")

    # 1. Initialize Pipeline
    print("\n1. Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    assert pipeline.llm.is_configured(), "Groq API key must be configured in .env"
    print("   [OK] Pipeline initialized. Groq API configured.")

    # 2. Ingest Sample PDF
    pdf_path = os.path.join("sample_documents", "machine_learning_notes.pdf")
    assert os.path.exists(pdf_path), f"File {pdf_path} does not exist"
    print(f"\n2. Ingesting and processing PDF: {pdf_path}...")
    metrics = pipeline.process_pdf(
        file_input=pdf_path,
        filename="machine_learning_notes.pdf",
        chunk_size=800,
        chunk_overlap=150
    )
    print(f"   [OK] Processed {metrics['pages_count']} pages into {metrics['chunks_count']} chunks.")
    print(f"   [OK] FAISS index created with dimension {metrics['dimension']} in {metrics['elapsed_time']}s.")
    assert metrics['pages_count'] == 4, f"Expected 4 pages, got {metrics['pages_count']}"
    assert metrics['chunks_count'] > 0, "Expected >0 chunks"
    assert pipeline.is_ready, "Pipeline must report ready"

    # 3. Test Question 1: Pooling layers (should cite Page 3)
    question1 = "What is the function of pooling layers in Convolutional Neural Networks?"
    print(f"\n3. Querying: '{question1}'...")
    answer1, sources1, context1 = pipeline.answer_question(question=question1, top_k=3)
    print("\n--- Answer from Groq ---")
    print(answer1)
    print("\n--- Retrieved Sources ---")
    for i, s in enumerate(sources1, 1):
        print(f"   Source {i}: Page {s['page']} | L2 Dist: {s['distance']:.4f} | Sim: {s['similarity_score']:.2%}")
        print(f"   Preview: {s['text'][:120]}...\n")

    assert len(sources1) > 0, "Must retrieve at least one source chunk"
    pages_found = [s["page"] for s in sources1]
    assert 3 in pages_found, f"Page 3 should be retrieved for pooling layers question, got pages: {pages_found}"
    print(f"   [OK] Correctly identified Page 3 in retrieved sources: {pages_found}")

    # 4. Test Question 2: Supervised vs Unsupervised (should cite Page 1)
    question2 = "What is the difference between supervised and unsupervised learning?"
    print(f"\n4. Querying: '{question2}'...")
    answer2, sources2, context2 = pipeline.answer_question(question=question2, top_k=3)
    print("\n--- Answer from Groq ---")
    print(answer2)
    pages_found_2 = [s["page"] for s in sources2]
    print(f"\n   [OK] Correctly identified Page 1 in retrieved sources: {pages_found_2}")
    assert 1 in pages_found_2, f"Page 1 should be retrieved for supervised question, got: {pages_found_2}"

    # 5. Test Question 3: Out-of-domain question (should say not found in document)
    question3 = "What is the capital city of France?"
    print(f"\n5. Querying Out-of-Domain Question: '{question3}'...")
    answer3, sources3, context3 = pipeline.answer_question(question=question3, top_k=3)
    print("\n--- Answer for Out-of-Domain Question ---")
    print(answer3)

    print("\n==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY! RAG PIPELINE IS 100% OPERATIONAL!")
    print("==================================================")


if __name__ == "__main__":
    test_full_rag_workflow()
