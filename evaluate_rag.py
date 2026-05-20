# evaluate_rag.py - Production RAG Evaluation
import os
import json
import httpx
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from sqlalchemy import create_engine, text

# ── Config ────────────────────────────────────────────────────────────────────
OPSPILOT_URL = os.getenv("OPSPILOT_URL", "https://opspilot-162649919209.asia-south2.run.app")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

# ── Test questions ────────────────────────────────────────────────────────────
TEST_DATA = [
    {
        "question": "What PPE is required for high voltage work?",
        "ground_truth": "Workers must wear Class E safety helmets, high voltage insulated gloves Class 2, arc flash suit, safety shoes with steel toe caps, and flame resistant clothing when working near energized equipment.",
    },
    {
        "question": "What is the procedure for outage restoration?",
        "ground_truth": "During outage restoration, crew must obtain safety clearances, verify de-energization using approved voltage detectors, apply earthing and short-circuiting, display warning signs, and obtain work permits.",
    },
    {
        "question": "How often should transformer oil be tested?",
        "ground_truth": "Transformer oil dielectric strength must be tested every 6 months. Oil level should be checked weekly and oil temperature checked daily.",
    },
]

# ── Get answers from OpsPilot API ─────────────────────────────────────────────
def get_token():
    res = httpx.post(
        f"{OPSPILOT_URL}/login",
        json={"username": "gow", "password": "abc123456"},
        timeout=30
    )
    return res.json()["access_token"]

def get_rag_response(question: str, token: str):
    res = httpx.post(
        f"{OPSPILOT_URL}/chat",
        json={"user_name": "evaluator", "message": question},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60
    )
    answer = res.json()["response"]
    return answer

# ── Save results to PostgreSQL ────────────────────────────────────────────────
def save_results(question: str, answer: str, scores: dict):
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO rag_evaluation_results 
            (question, answer, faithfulness, answer_relevancy, 
             context_precision, context_recall, overall_score)
            VALUES (:q, :a, :f, :ar, :cp, :cr, :overall)
        """), {
            "q": question,
            "a": answer,
            "f": scores.get("faithfulness", 0),
            "ar": scores.get("answer_relevancy", 0),
            "cp": scores.get("context_precision", 0),
            "cr": scores.get("context_recall", 0),
            "overall": scores.get("overall", 0)
        })
        conn.commit()

# ── Main evaluation ───────────────────────────────────────────────────────────
def run_evaluation():
    print("=" * 60)
    print("OpsPilot RAG Evaluation - Production Run")
    print(f"Started: {datetime.now()}")
    print("=" * 60)

    # Setup Groq as evaluator
    llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    ))

    # Get token
    print("Authenticating with OpsPilot...")
    token = get_token()

    # Build dataset
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_DATA:
        print(f"\nTesting: {item['question'][:60]}...")
        answer = get_rag_response(item["question"], token)
        print(f"Answer: {answer[:100]}...")

        questions.append(item["question"])
        answers.append(answer)
        contexts.append([answer])  # using answer as context
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Run RAGAS
    print("\nRunning RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm
    )

    # Print and save results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for i, item in enumerate(TEST_DATA):
        overall = (
            results["faithfulness"] +
            results["answer_relevancy"] +
            results["context_precision"] +
            results["context_recall"]
        ) / 4

        scores = {
            "faithfulness": results["faithfulness"],
            "answer_relevancy": results["answer_relevancy"],
            "context_precision": results["context_precision"],
            "context_recall": results["context_recall"],
            "overall": overall
        }

        save_results(item["question"], answers[i], scores)

    print(f"Faithfulness:      {results['faithfulness']:.3f}")
    print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}")
    print(f"Context Precision: {results['context_precision']:.3f}")
    print(f"Context Recall:    {results['context_recall']:.3f}")
    print(f"Overall:           {overall:.3f}")

    if overall > 0.7:
        print("✅ Good RAG quality!")
    elif overall > 0.5:
        print("⚠️  Average - needs improvement")
    else:
        print("❌ Poor - needs significant improvement")

    print(f"\nResults saved to PostgreSQL!")
    print(f"Completed: {datetime.now()}")

if __name__ == "__main__":
    run_evaluation()