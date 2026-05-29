
import os
import time
import mlflow
import httpx
from dotenv import load_dotenv
load_dotenv()

# OpsPilot URL
OPSPILOT_URL = "http://16.112.61.255:8080"

# Test questions with expected answers
TEST_QUESTIONS = [
    {
        "question": "What PPE is required for high voltage work?",
        "expected_keywords": ["helmet", "gloves", "arc flash", "safety shoes"]
    },
    {
        "question": "What is the procedure for outage restoration?",
        "expected_keywords": ["clearance", "earthing", "permit", "de-energization"]
    },
    {
        "question": "How often should transformer oil be tested?",
        "expected_keywords": ["6 months", "dielectric", "weekly", "daily"]
    },
    {
        "question": "Who are our top suppliers?",
        "expected_keywords": ["bharat", "power", "supplier"]
    },
    {
        "question": "Show pending high priority requirements",
        "expected_keywords": ["high", "open", "REQ"]
    }
]

def get_token():
    res = httpx.post(
        f"{OPSPILOT_URL}/login",
        json={"username": "gow", "password": "abc123456"},
        timeout=30
    )
    return res.json()["access_token"]

def ask_question(question: str, token: str) -> tuple[str, float]:
    start = time.time()
    res = httpx.post(
        f"{OPSPILOT_URL}/chat",
        json={"user_name": "mlflow_test", "message": question},
        headers={"Authorization": f"Bearer {token}"},
        timeout=60
    )
    latency = time.time() - start
    answer = res.json()["response"]
    return answer, latency

def score_answer(answer: str, expected_keywords: list) -> float:
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)

def run_experiment(experiment_name: str, run_name: str, params: dict):
    mlflow.set_experiment(experiment_name)
    
    token = get_token()
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)
        
        scores = []
        latencies = []
        
        for item in TEST_QUESTIONS:
            print(f"Testing: {item['question'][:50]}...")
            answer, latency = ask_question(item["question"], token)
            score = score_answer(answer, item["expected_keywords"])
            
            scores.append(score)
            latencies.append(latency)
            
            print(f"Score: {score:.2f} | Latency: {latency:.2f}s")
            print(f"Answer: {answer[:100]}...")
            print()
        
        # Log metrics
        avg_score = sum(scores) / len(scores)
        avg_latency = sum(latencies) / len(latencies)
        
        mlflow.log_metric("avg_relevancy_score", avg_score)
        mlflow.log_metric("avg_latency_seconds", avg_latency)
        mlflow.log_metric("min_score", min(scores))
        mlflow.log_metric("max_score", max(scores))
        
        print("=" * 50)
        print(f"Run: {run_name}")
        print(f"Avg Score: {avg_score:.3f}")
        print(f"Avg Latency: {avg_latency:.2f}s")
        print("=" * 50)
        
        return avg_score

if __name__ == "__main__":
    # Run 1 — current settings
    run_experiment(
        experiment_name="OpsPilot RAG Quality",
        run_name="baseline_llama70b",
        params={
            "model": "llama-3.3-70b-versatile",
            "chunk_size": 500,
            "top_k": 5,
            "search_type": "hybrid_bm25_pgvector"
        }
    )
    
    print("\nStarting MLflow UI...")
    print("Run: mlflow ui")
    print("Open: http://localhost:5000")

# Run 2 — mixtral
run_experiment(
    experiment_name="OpsPilot RAG Quality",
    run_name="mixtral_8x7b",
    params={
        "model": "mixtral-8x7b-32768",
        "chunk_size": 500,
        "top_k": 5,
        "search_type": "hybrid_bm25_pgvector"
    }
)