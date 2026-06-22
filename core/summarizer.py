from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os 

def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.3)


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )

    return splitter.split_text(transcript)

def summarize(transcript : str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [ 
        ("system",
        """
        You are an expert educational content analyst.

        Analyze this portion of a YouTube video transcript and create a concise summary.

        Focus on:
        - Main topics discussed
        - Important concepts introduced
        - Key explanations and examples
        - Important insights or conclusions

        Do not include filler conversation, greetings, sponsor messages, or repeated statements.

        Return 3-7 concise bullet points."""
        ),
        ("human", "{text}"),
    ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
            """
            You are an expert YouTube content analyst.

            Combine these partial summaries into a comprehensive video summary.

            Requirements:
            - Create a structured summary of the entire video.
            - Preserve the logical flow of topics.
            - Highlight the most important concepts and explanations.
            - Remove duplicate information.
            - Focus on educational value and practical insights.
            - Write in a way that someone could understand the video's content without watching it.

            Output format:

            ## Video Overview
            (2-4 paragraphs)

            ## Main Topics Covered
            - Topic 1
            - Topic 2
            - Topic 3

            ## Key Insights
            - Insight 1
            - Insight 2
            - Insight 3

            ## Conclusion
            (Brief conclusion summarizing the video's main message)
            """,),
        ("human", "{text}"),
    ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)

def generate_title(transcipt : str) -> str:
    llm = get_llm()
    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the meeting transcript, generate a short professional title for the video"
                "Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ]
    )
    

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | title_prompt | llm | StrOutputParser()
    )

    return title_chain.invoke(transcipt)



