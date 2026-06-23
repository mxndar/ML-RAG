import asyncio
from app.config import settings
from app.tools import bm25_tool
from app.retrieval.reranker import rerank

bm25_tool.load_bm25()

query = "Professional Qualifications in Sujal's resume"

async def run():
    print(f"Query: {query}")

    # 1. BM25 Search
    bm25_results = await bm25_tool.bm25_search(query, top_k=10)
    print("\n--- BM25 Top 10 ---")
    for r in bm25_results:
        print(f"[{r.metadata.document_name}] Score: {r.score:.3f} | {r.text[:60]}...")

    # 2. Reranking
    print("\n--- Cross Encoder Top 2 ---")
    reranked = await rerank(query, bm25_results, top_k=2)
    for r in reranked:
        print(f"[{r.metadata.document_name}] Score: {r.score:.3f} | {r.text[:60]}...")

asyncio.run(run())
