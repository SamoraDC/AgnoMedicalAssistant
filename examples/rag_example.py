"""RAG pipeline usage example."""
from src.rag import RAGPipeline


def main():
    """Run RAG example."""
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline()

    # Sample medical knowledge
    medical_texts = [
        """
        Myocardial Infarction (Heart Attack):
        Symptoms include severe chest pain, pain radiating to left arm or jaw,
        shortness of breath, sweating, nausea, and lightheadedness.
        Immediate emergency care is required. Call 911 immediately.
        Treatment involves restoring blood flow through medications or procedures.
        """,
        """
        Common Cold:
        Symptoms include runny nose, congestion, sore throat, cough, and mild fatigue.
        Usually resolves within 7-10 days with rest and fluids.
        Over-the-counter medications can help manage symptoms.
        Seek medical attention if symptoms persist beyond 10 days or worsen.
        """,
        """
        Type 2 Diabetes:
        Chronic condition affecting blood sugar regulation.
        Symptoms include increased thirst, frequent urination, fatigue, and blurred vision.
        Managed through diet, exercise, and medication.
        Regular monitoring and medical checkups are essential.
        """
    ]

    # Ingest documents
    print("\nIngesting medical documents...")
    for i, text in enumerate(medical_texts):
        result = rag.ingest_document(
            text=text,
            metadata={"source": "medical_textbook", "topic": f"topic_{i}"},
            doc_id=f"med_doc_{i}"
        )
        print(f"Document {i+1}: {result['chunks_created']} chunks created")

    # Query examples
    queries = [
        "What are the symptoms of a heart attack?",
        "How long does a common cold last?",
        "What are the symptoms of diabetes?"
    ]

    print("\n" + "=" * 80)
    print("Running Queries")
    print("=" * 80)

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)

        result = rag.query(query, k=2)

        for i, doc in enumerate(result['results'], 1):
            print(f"\nResult {i} (similarity: {doc['similarity_score']:.3f}):")
            print(f"Content: {doc['content'][:200]}...")

    # Cleanup
    rag.close()
    print("\n✅ RAG example completed!")


if __name__ == "__main__":
    main()
