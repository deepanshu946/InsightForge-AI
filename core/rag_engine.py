import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from core.vector_store import build_vector_store, get_hybrid_retriever

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def get_web_search_tool():
    return DuckDuckGoSearchResults(output_format="list")


def web_search_answer(question: str, llm):

    search_tool = get_web_search_tool()
    results = search_tool.invoke(question)

    sources_text = []
    source_links = []

    for result in results[:5]:

        title = result.get("title", "")
        snippet = result.get("snippet", "")
        link = result.get("link", "")

        sources_text.append(
            f"Title: {title}\nSnippet: {snippet}\nURL: {link}"
        )

        source_links.append({
            "title": title,
            "url": link,
        })

    web_context = "\n\n".join(sources_text)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Answer the user's question using the provided web search results.

If the search results are insufficient, say so clearly.

Web Results:
{context}""",
        ),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": web_context,
            "question": question,
        }
    )

    return answer, source_links

def format_context(retrieved: list[dict]) -> tuple[str, list[dict]]:
    """Splits retrieved items into a context string and a sources list."""
    context_parts = []
    sources = []
    for item in retrieved:
        doc = item["doc"]
        context_parts.append(doc.page_content)
        sources.append({
            "text":         doc.page_content[:150] + "…",
            "chunk_index":  doc.metadata.get("chunk_index"),
            "found_by":     item["found_by"],
            "rrf_score":    item.get("rrf_score"),
            "rerank_score": item.get("rerank_score"),
        })
    return "\n\n".join(context_parts), sources

def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever    = get_hybrid_retriever(vector_store, k=4)
    llm          = get_llm()

    chunk_count = None
    try:
        if hasattr(vector_store, "docstore") and hasattr(vector_store.docstore, "_dict"):
            chunk_count = len(vector_store.docstore._dict)
    except Exception:
        chunk_count = None

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert YouTube video assistant.

Answer the user's question using ONLY the information from the provided YouTube transcript context.

If the answer is not present in the transcript, respond with:
"I could not find this information in the video transcript."

Be clear, concise, and accurate. When summarizing, focus on the key points discussed in the video. If quoting or referencing a speaker, make that clear.

Video transcript context:
{context}""",
        ),
        ("human", "{question}"),
    ])

    # Return a plain dict — NOT a RunnableSequence
    return {
        "retriever": retriever,
        "prompt": prompt,
        "llm": llm,
        "chunk_count": chunk_count,
    }


def ask_question(rag_chain: dict, question: str) -> dict:
    retriever = rag_chain["retriever"]
    prompt    = rag_chain["prompt"]
    llm       = rag_chain["llm"]

    # Retrieve with full metadata
    retrieved = retriever.invoke(question)
    context, sources = format_context(retrieved)

    # Build and run the chain with pre-fetched context
    chain = prompt | llm | StrOutputParser()
    rag_answer = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    no_answer_phrases = [
        "i could not find this information in the video transcript",
        "not present in the transcript",
        "not mentioned in the transcript",
        "cannot find this information",
    ]

    rag_failed = any(
        phrase in rag_answer.lower()
        for phrase in no_answer_phrases
    )

    web_answer, web_sources = web_search_answer(question, llm)

    if rag_failed:
        return {
            "mode": "web_only",
            "rag_answer": None,
            "web_answer": web_answer,
            "transcript_sources": [],
            "web_sources": web_sources,
        }

    return {
        "mode": "rag_plus_web",
        "rag_answer": rag_answer,
        "web_answer": web_answer,
        "transcript_sources": sources,
        "web_sources": web_sources,
    }