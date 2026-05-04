import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag import retrieve_candidates, rerank_chunks, mmr_select

q = "What PPE is required for high voltage work?"
candidates = retrieve_candidates(q, top_k=10)
selected = mmr_select(q, candidates, k=6)
reranked = rerank_chunks(q, selected)

print("FINAL CONTEXT SENT TO LLM:")
print("─" * 40)
for i, c in enumerate(reranked):
    print(f"[{i+1}] {c['source']} p{c['page']}")
    print(c['text'])
    print()