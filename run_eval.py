# run_eval.py
# RAGAS evaluation for OpsPilot
# Calls search_documents() directly (no chain needed)

import os
import sys
import asyncio
import numpy as np

sys.path.append('/home/ubuntu/opspilot/opspilot')

from dotenv import load_dotenv
load_dotenv()

# ── Imports ───────────────────────────────────────────────────
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas_test_dataset import TEST_DATA

# ── Setup Groq as evaluator LLM ───────────────────────────────
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper

groq_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
))

# Set Groq as LLM for all metrics
faithfulness.llm = groq_llm
answer_relevancy.llm = groq_llm
context_precision.llm = groq_llm
context_recall.llm = groq_llm

# ── Setup embeddings for answer_relevancy ─────────────────────
from langchain_huggingface import HuggingFaceEmbeddings   # updated import
from ragas.embeddings import LangchainEmbeddingsWrapper

embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)
answer_relevancy.embeddings = embeddings

# ── Helper: safely extract a scalar from RAGAS result ─────────
def scalar(val) -> float:
    """RAGAS may return a list of per-row scores; return the mean."""
    if isinstance(val, list):
        clean = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return float(np.mean(clean)) if clean else float("nan")
    return float(val) if val is not None else float("nan")

# ── Get answers from RAG directly ─────────────────────────────
def get_rag_answer(question: str) -> tuple:
    """Call search_documents directly and get answer + contexts"""
    from rag import search_documents, retrieve_candidates

    # Get retrieved chunks (contexts)
    try:
        candidates = retrieve_candidates(question, top_k=5)
        contexts = [c["text"] for c in candidates]
    except Exception:
        contexts = []

    # Get final answer
    answer = search_documents(question)

    return answer, contexts

# ── Build evaluation dataset ──────────────────────────────────
def build_dataset():
    print("Building evaluation dataset...")
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_DATA:
        print(f"  Testing: {item['question'][:60]}...")
        answer, context = get_rag_answer(item["question"])
        print(f"  Answer:  {answer[:80]}...")
        print(f"  Chunks:  {len(context)} retrieved")
        print()

        questions.append(item["question"])
        answers.append(answer)
        contexts.append(context if context else [answer])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

# ── Run evaluation ─────────────────────────────────────────────
async def _run_ragas(dataset):
    """Run RAGAS inside a proper async context to avoid Timeout errors."""
    return evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        run_config=RunConfig(
            timeout=120,        # seconds per metric call
            max_retries=3,
            max_wait=60,
        ),
    )

def run_evaluation():
    print("=" * 60)
    print("OpsPilot RAGAS Evaluation")
    print(f"Total questions: {len(TEST_DATA)}")
    print("=" * 60)

    dataset = build_dataset()

    print("Running RAGAS metrics...")

    # ✅ Fix: run inside asyncio event loop so Timeout works correctly
    results = asyncio.run(_run_ragas(dataset))

    # ✅ Fix: extract scalar means from per-row lists
    faith  = scalar(results['faithfulness'])
    relevancy = scalar(results['answer_relevancy'])
    precision = scalar(results['context_precision'])
    recall    = scalar(results['context_recall'])

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Faithfulness:      {faith:.3f}")
    print(f"Answer Relevancy:  {relevancy:.3f}")
    print(f"Context Precision: {precision:.3f}")
    print(f"Context Recall:    {recall:.3f}")

    scores = [s for s in [faith, relevancy, precision, recall] if not np.isnan(s)]
    overall = float(np.mean(scores)) if scores else float("nan")

    print(f"\nOverall Score:     {overall:.3f}")

    if overall > 0.7:
        print("✅ Good RAG quality!")
    elif overall > 0.5:
        print("⚠️  Average - needs improvement")
    else:
        print("❌ Poor - needs significant improvement")

    # Save results to file
    with open("ragas_results.txt", "w") as f:
        f.write(f"Faithfulness:      {faith:.3f}\n")
        f.write(f"Answer Relevancy:  {relevancy:.3f}\n")
        f.write(f"Context Precision: {precision:.3f}\n")
        f.write(f"Context Recall:    {recall:.3f}\n")
        f.write(f"Overall Score:     {overall:.3f}\n")

    print("\nResults saved to ragas_results.txt!")
    return results

if __name__ == "__main__":
    run_evaluation()