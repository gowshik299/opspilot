
import os
import httpx
import mlflow
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Setup
OPSPILOT_URL = "http://16.112.61.255:8080"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Test data with ground truth
TEST_DATA = [
    {
        "question": "What PPE is required for high voltage work?",
        "ground_truth": "Workers must wear Class E safety helmets, high voltage insulated gloves Class 2, arc flash suit, safety shoes with steel toe caps, and flame resistant clothing when working near energized equipment.",
    },
    {
        "question": "How often should transformer oil be tested?",
        "ground_truth": "Transformer oil dielectric strength must be tested every 6 months. Oil level should be checked weekly and oil temperature checked daily.",
    },
    {
        "question": "What safety measures are required before working on electrical equipment?",
        "ground_truth": "Before working on electrical equipment, workers must obtain work permits, verify de-energization, apply earthing and short-circuiting, display warning signs, and ensure proper PPE is worn.",
    },
]

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
        json={"user_name": "ragas_eval", "message": question},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60
    )
    answer = res.json()["response"]
    return answer, [answer]  # answer used as context too

def run_ragas_evaluation():
    print("=" * 60)
    print("OpsPilot RAGAS Evaluation")
    print("=" * 60)

    # Setup Groq as evaluator LLM
    llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY
    ))

    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Get token
    token = get_token()

    # Build dataset
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_DATA:
        print(f"Testing: {item['question'][:50]}...")
        answer, context = get_rag_response(item["question"], token)
        print(f"Answer: {answer[:80]}...")
        print()

        questions.append(item["question"])
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # Run RAGAS
    print("Running RAGAS evaluation...")
    results = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=llm,
        embeddings=embeddings
    )

    # Log to MLflow
    mlflow.set_experiment("OpsPilot RAG Quality")
    with mlflow.start_run(run_name="ragas_evaluation"):
        mlflow.log_param("evaluator", "ragas_0.4.3")
        mlflow.log_param("llm_judge", "llama-3.3-70b")
        mlflow.log_param("num_questions", len(TEST_DATA))

        mlflow.log_metric("faithfulness", results["faithfulness"])
        mlflow.log_metric("answer_relevancy", results["answer_relevancy"])
        mlflow.log_metric("context_precision", results["context_precision"])
        mlflow.log_metric("context_recall", results["context_recall"])

        overall = sum([
            results["faithfulness"],
            results["answer_relevancy"],
            results["context_precision"],
            results["context_recall"],
        ]) / 4

        mlflow.log_metric("overall_ragas_score", overall)

    # Print results
    print("\n" + "=" * 60)
    print("RAGAS RESULTS")
    print("=" * 60)
    print(f"Faithfulness:      {results['faithfulness']:.3f}")
    print(f"Answer Relevancy:  {results['answer_relevancy']:.3f}")
    print(f"Context Precision: {results['context_precision']:.3f}")
    print(f"Context Recall:    {results['context_recall']:.3f}")
    print(f"Overall Score:     {overall:.3f}")

    if overall > 0.7:
        print("✅ Good RAG quality!")
    elif overall > 0.5:
        print("⚠️  Average - needs improvement")
    else:
        print("❌ Poor - needs significant improvement")

    print("\nResults logged to MLflow!")
    print("Check: http://localhost:5000")

if __name__ == "__main__":
    run_ragas_evaluation()