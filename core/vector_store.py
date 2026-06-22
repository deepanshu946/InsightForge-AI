import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank


COLLECTION_NAME = "meeting_transcript"


@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


@st.cache_resource
def get_reranker():
    return CohereRerank(
        model="rerank-v3.5",
        top_n=5,
    )


def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)
    st.session_state.chunk_count = len(chunks)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    # Build BM25 with no hardcoded k — set dynamically in invoke()
    bm25_retriever = BM25Retriever.from_documents(docs)

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
    )

    # Single source of truth — session_state only
    st.session_state.vector_store = vector_store
    st.session_state.bm25_retriever = bm25_retriever

    return vector_store


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def reciprocal_rank_fusion(
    results_list: list[list[Document]],
    weights: list[float] = None,
    k: int = 60,
) -> list[dict]:
    if weights is None:
        weights = [1.0] * len(results_list)

    source_names = ["bm25", "vector"]
    fused_scores = {}
    doc_map      = {}
    source_map   = {}

    for list_idx, results in enumerate(results_list):
        w    = weights[list_idx]
        name = source_names[list_idx] if list_idx < len(source_names) else f"source_{list_idx}"

        for rank, doc in enumerate(results):
            key = doc.page_content

            doc_map[key]    = doc
            fused_scores[key] = fused_scores.get(key, 0) + w * (1 / (k + rank + 1))

            if key not in source_map:
                source_map[key] = set()
            source_map[key].add(name)

    reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            "doc":       doc_map[key],
            "rrf_score": round(score, 4),
            "found_by":  "both" if len(source_map[key]) > 1 else next(iter(source_map[key])),
        }
        for key, score in reranked
    ]


class HybridRetriever:

    def __init__(
        self,
        vector_store: Chroma,
        bm25_retriever,
        k: int = 4,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        search_mode: str = "hybrid",   # "hybrid" | "semantic" | "keyword"
    ):
        self.vector_store   = vector_store
        self.bm25_retriever = bm25_retriever
        self.k              = k
        self.bm25_weight    = bm25_weight
        self.vector_weight  = vector_weight
        self.search_mode    = search_mode
        self.reranker = get_reranker()

    def invoke(self, query: str) -> list[dict]:
        candidate_k = max(self.k * 5, 20)

        if self.search_mode == "semantic":
            docs = self.vector_store.similarity_search(query, k=self.k)
            return [{"doc": d, "rrf_score": None, "found_by": "vector"} for d in docs]

        if self.search_mode == "keyword":
            self.bm25_retriever.k = self.k
            docs = self.bm25_retriever.invoke(query)
            return [{"doc": d, "rrf_score": None, "found_by": "bm25"} for d in docs]

        # Hybrid — fetch more candidates from both, then fuse
        self.bm25_retriever.k = candidate_k
        semantic_docs = self.vector_store.similarity_search(query, k=candidate_k)
        keyword_docs  = self.bm25_retriever.invoke(query)

        fused = reciprocal_rank_fusion(
            [keyword_docs, semantic_docs],
            weights=[self.bm25_weight, self.vector_weight],
        )

        candidate_docs = [item["doc"] for item in fused[:candidate_k]]

        reranked_docs = self.reranker.compress_documents(
            documents=candidate_docs,
            query=query,
        )

        metadata_lookup = {
            item["doc"].page_content: item
            for item in fused
        }

        results = []

        for doc in reranked_docs[: self.k]:
            original = metadata_lookup.get(doc.page_content)

            rerank_score = None
            if hasattr(doc, "metadata"):
                rerank_score = doc.metadata.get("relevance_score")

            results.append(
                {
                    "doc": doc,
                    "rrf_score": original["rrf_score"] if original else None,
                    "rerank_score": round(rerank_score, 4) if isinstance(rerank_score, (int, float)) else None,
                    "found_by": original["found_by"] if original else "reranker",
                }
            )

        return results


def get_hybrid_retriever(
    vector_store: Chroma,
    k: int = 4,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    search_mode: str = "hybrid",
) -> HybridRetriever:

    bm25_retriever = st.session_state.get("bm25_retriever")

    if bm25_retriever is None:
        raise ValueError(
            "BM25 retriever not found in session state. "
            "Call build_vector_store() first."
        )

    return HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        k=k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        search_mode=search_mode,
    )