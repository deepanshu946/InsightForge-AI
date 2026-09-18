import os


def validate_environment(language: str) -> list:
    """Return a list of human-readable error strings; empty list means OK."""
    errors = []

    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is not set — required for transcription, LLM calls, and embeddings.")

    if not os.getenv("COHERE_API_KEY"):
        errors.append("COHERE_API_KEY is not set — required for result reranking.")

    if language.lower() == "hinglish" and not os.getenv("SARVAM_API_KEY"):
        errors.append("SARVAM_API_KEY is not set — required for Hinglish transcription.")

    return errors
