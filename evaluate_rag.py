# evaluate_rag.py
# RAG Evaluation using RAGAS

import os
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ── Step 1: Define test questions ─────────────────────────────────────────────
# These are questions + ground truth answers from your PDFs

test_data = [
    {
        "question": "What PPE is required for high voltage work?",
        "ground_truth": "Workers must wear Class E safety helmets, high voltage insulated gloves Class 2, arc flash suit rated for the voltage level, safety shoes with steel toe caps, and flame resistant clothing when working near energized equipment.",
    },
    {
        "question": "What is the procedure for outage restoration?",
        "ground_truth": "During outage restoration, the crew must first ensure all safety clearances are obtained, verify de-energization using approved voltage detectors, apply earthing and short-circuiting, display warning signs, and obtain work permits before beginning restoration work.",
    },
    {
        "question": "How often should transformer oil be tested?",
        "ground_truth": "Transformer oil dielectric strength must be tested every 6 months. Oil level should be checked weekly and oil temperature checked daily.",
    },
]

# ── Step 2: Get answers from your RAG ─────────────────────────────────────────

def get_rag_response(question: str):
    """Get answer and context from your RAG system"""
    from rag import search_documents, retrieve_candidates, embedder
    
    # Get retrieved chunks (context)
    candidates = retrieve_candidates(question, top_k=5)
    contexts = [c["text"] for c in candidates]
    
    # Get answer from RAG
    answer = search_documents(question)
    
    return answer, contexts

# ── Step 3: Build evaluation dataset ──────────────────────────────────────────

def build_eval_dataset():
    print("Building evaluation dataset...")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in test_data:
        print(f"Testing: {item['question'][:50]}...")
        
        answer, retrieved_contexts = get_rag_response(item["question"])
        
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(retrieved_contexts)
        ground_truths.append(item["ground_truth"])
        
        print(f"Answer: {answer[:100]}...")
        print(f"Contexts retrieved: {len(retrieved_contexts)}")
        print()
    
    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

# ── Step 4: Run RAGAS evaluation ──────────────────────────────────────────────

def run_evaluation():
    print("=" * 60)
    print("OpsPilot RAG Evaluation")
    print("=" * 60)
    
    # Build dataset
    dataset = build_eval_dataset()
    
    print("\nRunning RAGAS evaluation...")
    print("This may take 2-3 minutes...\n")
    
    # Run evaluation
    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Faithfulness:      {results['faithfulness']:.3f}")
    print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}")
    print(f"Context Precision: {results['context_precision']:.3f}")
    print(f"Context Recall:    {results['context_recall']:.3f}")
    
    overall = sum([
        results['faithfulness'],
        results['answer_relevancy'],
        results['context_precision'],
        results['context_recall'],
    ]) / 4
    
    print(f"\nOverall RAG Score: {overall:.3f}")
    
    if overall > 0.7:
        print("✅ Good RAG quality!")
    elif overall > 0.5:
        print("⚠️  Average RAG quality - needs improvement")
    else:
        print("❌ Poor RAG quality - needs significant improvement")
    
    return results


if __name__ == "__main__":
    run_evaluation()