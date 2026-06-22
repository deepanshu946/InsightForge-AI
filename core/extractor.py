#Actionableitems , decision , questions 

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os 


def get_llm():
    return ChatMistralAI(model = "mistral-small-latest", mistral_api_key = os.getenv("MISTRAL_API_KEY"),temperature=0.2)



def build_chain(system_prompt : str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript:str)->str:
    chain = build_chain(
        """You are an expert video content analyst.

        Analyze the transcript and generate YouTube-style chapters.

        Instructions:
        - Divide the video into logical sections.
        - Use timestamps from the transcript whenever available.
        - Create a short and descriptive chapter title.
        - Chapters should cover the entire video flow.
        - Do not create excessively small chapters.
        - If timestamps are unavailable, estimate chapter boundaries based on topic transitions.

        Output format:

        1. [00:00] Introduction
        Brief description of what is discussed.

        2. [02:15] What is Retrieval-Augmented Generation (RAG)?
        Explanation of RAG and its components.

        3. [07:40] Building the Vector Database
        Discussion of embeddings and storage.

        Continue until the end of the video."""
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        """You are an expert educational content analyst.

        Analyze the video transcript and extract the most important concepts, insights, and learnings.

        Instructions:
        - Focus on what a viewer should remember after watching.
        - Combine similar points.
        - Explain each takeaway in 2-4 concise sentences.
        - Prioritize practical insights and core concepts.

        Output format:

        1. Key Takeaway Title
        Explanation

        2. Key Takeaway Title
        Explanation

        If no significant learnings are present, state:
        'No major takeaways identified.'"""
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        """You are an expert tutor.

        Based on the video transcript, generate the most likely questions a viewer may ask after watching the video.

        Instructions:
        - Focus on conceptual understanding.
        - Include beginner and intermediate level questions.
        - Provide concise and accurate answers.
        - Cover the most important topics discussed.
        - Generate between 5 and 15 questions depending on content depth.

        Output format:

        Q1. What is RAG?
        A1. Retrieval-Augmented Generation is a technique that combines information retrieval with large language models.

        Q2. Why are embeddings used?
        A2. Embeddings convert text into numerical vectors that allow semantic similarity search.

        Continue for all important topics."""
    )
    return chain.invoke(transcript)