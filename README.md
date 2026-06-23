# InsightForge AI 

> Transform videos into searchable knowledge bases using AI-powered transcription, hybrid retrieval, reranking, and conversational question answering. working link - https://insightforge--ai.streamlit.app/

---

# Overview

InsightForge AI is a Video Intelligence Platform that converts videos into structured, searchable knowledge. The platform accepts YouTube URLs or local video files, transcribes their content using OpenAI's GPT-4o Mini Transcribe model, extracts actionable insights, builds a hybrid Retrieval-Augmented Generation (RAG) pipeline, and enables users to interact with video content through natural language conversations.

Unlike traditional transcription tools, InsightForge AI combines semantic search, keyword retrieval, Reciprocal Rank Fusion (RRF), Cohere reranking, and web-search augmentation to provide highly accurate, grounded answers.

---

# Key Features

## Video Processing

- YouTube URL ingestion
- Local video file support
- Automatic audio extraction
- Audio compression before transcription
- Chunk-based processing for large files

## AI-Powered Transcription

- OpenAI GPT-4o Mini Transcribe
- Parallel chunk transcription
- Multilingual support
- English support
- Hinglish support

## Content Intelligence

Automatically generates:

- AI-generated title
- Executive summary
- Action items
- Key decisions
- Open questions

## Hybrid Retrieval-Augmented Generation (RAG)

- Semantic search
- BM25 keyword retrieval
- Reciprocal Rank Fusion (RRF)
- Cohere reranking
- Context-aware answer generation

## Web Search Augmentation

- DuckDuckGo Search integration
- External knowledge retrieval
- Source attribution
- Fallback answering when transcript knowledge is insufficient

## Explainable Retrieval

Displays:

- Retrieved chunks
- Retrieval source
- RRF score
- Cohere rerank score
- Web sources

## Analytics Dashboard

Tracks:

- Videos processed
- Questions asked
- Transcription latency
- Pipeline latency
- Question answering latency
- Average rerank score
- Web-search fallback rate

---

# System Architecture

```text
                        ┌────────────────────┐
                        │  User Input        │
                        │                    │
                        │ • YouTube URL      │
                        │ • Video File       │
                        └─────────┬──────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Audio Processing Layer   │
                    │                          │
                    │ • Audio Extraction       │
                    │ • Audio Compression      │
                    │ • Audio Chunking         │
                    └─────────┬────────────────┘
                              │
                              ▼
                 ┌──────────────────────────────┐
                 │ GPT-4o Mini Transcribe       │
                 │                              │
                 │ Parallel Chunk Processing    │
                 └─────────┬────────────────────┘
                           │
                           ▼
                ┌─────────────────────────────┐
                │ Complete Transcript         │
                └─────────┬───────────────────┘
                          │
       ┌──────────────────┼───────────────────┐
       ▼                  ▼                   ▼
 ┌──────────┐      ┌────────────┐      ┌─────────────┐
 │ Summary  │      │ Extraction │      │ Title Gen   │
 └──────────┘      └────────────┘      └─────────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Transcript Chunking│
                └─────────┬──────────┘
                          │
                          ▼
         ┌───────────────────────────────────┐
         │ OpenAI text-embedding-3-small     │
         └──────────────┬────────────────────┘
                        │
                        ▼
                ┌──────────────┐
                │ ChromaDB     │
                │ Vector Store │
                └──────┬───────┘
                       │
       ┌───────────────┼────────────────┐
       ▼                                ▼
┌───────────────┐              ┌────────────────┐
│ Vector Search │              │ BM25 Retrieval │
└───────┬───────┘              └───────┬────────┘
        │                              │
        └──────────────┬───────────────┘
                       ▼
          ┌────────────────────────┐
          │ Reciprocal Rank Fusion │
          │         (RRF)          │
          └────────────┬───────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Cohere Reranker   │
             └─────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Top-K Chunks     │
              └─────────┬────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ LLM Answer Gen   │
               └─────────┬────────┘
                         │
      ┌──────────────────┼──────────────────┐
      ▼                                     ▼
┌──────────────┐                  ┌─────────────────┐
│ RAG Answer   │                  │ DuckDuckGo Web  │
│              │                  │ Search Fallback │
└───────┬──────┘                  └────────┬────────┘
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              ┌───────────────────┐
              │ Final Response    │
              └───────────────────┘
```

---

# Detailed Workflow

## 1. Video Ingestion

The system accepts:

### Supported Inputs

- YouTube URLs
- MP4 files
- MOV files
- Local video files

Examples:

```text
https://youtube.com/watch?v=xxxx

meeting.mp4

lecture.mov
```

---

## 2. Audio Processing

The video is converted into audio.

### Processing Steps

1. Audio extraction
2. Audio compression
3. Audio chunk generation
4. Chunk preparation for transcription

Benefits:

- Faster uploads
- Lower API cost
- Improved scalability
- Better handling of long videos

---

## 3. Transcription Layer

### Model

```text
OpenAI GPT-4o Mini Transcribe
```

The audio chunks are transcribed independently and merged into a single transcript.

Benefits:

- High accuracy
- Better punctuation
- Better multilingual support
- Faster processing through chunk parallelization

---

## 4. Content Intelligence Layer

The transcript is processed using LLMs to generate:

### Title Generation

Example:

```text
Weekly Product Planning Meeting
```

### Summary Generation

Produces concise summaries of long transcripts.

### Action Item Extraction

Example:

```text
- Finalize API design
- Deploy application
- Prepare roadmap
```

### Key Decision Extraction

Example:

```text
- Adopt OpenAI embeddings
- Use ChromaDB
```

### Open Question Extraction

Example:

```text
- Should analytics be persisted?
- Which deployment platform should be used?
```

---

# Knowledge Base Creation

## Transcript Chunking

Large transcripts are split into smaller chunks.

```text
Transcript
   ↓
Chunk 1
Chunk 2
Chunk 3
...
Chunk N
```

This improves retrieval precision.

---

# Embedding Generation

## Model

```text
OpenAI text-embedding-3-small
```

Each chunk is converted into a dense vector representation.

Benefits:

- Semantic understanding
- Meaning-based retrieval
- Improved search quality

---

# Vector Database

## ChromaDB

Used for:

- Vector storage
- Similarity search
- Fast retrieval

Current implementation uses session-scoped storage rather than long-term persistence.

---

# Hybrid Retrieval System

Traditional RAG systems rely solely on semantic search.

InsightForge AI combines multiple retrieval strategies.

## Semantic Search

Uses vector similarity through ChromaDB.

Example:

```text
Question:
How was deployment discussed?

Transcript:
Hosting strategy and infrastructure planning...
```

Semantic retrieval can match these concepts.

---

## BM25 Retrieval

Keyword-based retrieval.

Benefits:

- Exact term matching
- Acronym matching
- Technical terminology matching

---

## Reciprocal Rank Fusion (RRF)

Combines:

```text
Semantic Search
+
BM25 Retrieval
```

into a unified ranking system.

Benefits:

- Better recall
- Better retrieval robustness
- Improved relevance

---

# Reranking Layer

## Cohere Rerank

After retrieval:

```text
Top Candidate Chunks
```

are reranked by:

```text
Cohere Rerank API
```

The reranker assigns relevance scores.

Example:

```text
Chunk A : 0.98
Chunk B : 0.94
Chunk C : 0.52
```

Only the most relevant chunks are used for answer generation.

---

# Question Answering Pipeline

```text
User Question
      ↓
Hybrid Retrieval
      ↓
RRF Fusion
      ↓
Cohere Rerank
      ↓
Top Context Chunks
      ↓
LLM Generation
      ↓
Grounded Answer
```

The answer is generated exclusively from retrieved transcript context whenever possible.

---

# Web Search Augmentation

If transcript retrieval confidence is low:

```text
Question
     ↓
DuckDuckGo Search
     ↓
External Sources
     ↓
Web Answer
```

The system returns:

- Transcript-based answer
- Web-based answer
- Source references

This improves answer coverage beyond the uploaded video.

---

# Retrieval Explainability

For every response, InsightForge AI exposes:

## Retrieval Source

- Semantic Search
- BM25
- Both

## RRF Score

Displays fusion ranking score.

## Cohere Rerank Score

Displays reranker relevance score.

## Chunk Inspection

Allows users to inspect retrieved transcript chunks.

This improves transparency and trust.

---

# Analytics & Monitoring

The platform includes a real-time analytics dashboard.

## Metrics Collected

### Processing Metrics

- Videos processed
- Pipeline execution time
- Transcription latency

### Retrieval Metrics

- Average rerank score
- Retrieval effectiveness

### User Metrics

- Questions asked
- Response latency

### Fallback Metrics

- Web-search usage percentage

---

# Technology Stack

## Frontend

- Streamlit

## AI Models

### Transcription

- GPT-4o Mini Transcribe

### Embeddings

- OpenAI text-embedding-3-small

### Answer Generation

- Mistral AI

### Reranking

- Cohere Rerank

---

## Retrieval

- Semantic Search
- BM25
- Reciprocal Rank Fusion (RRF)

---

## Vector Database

- ChromaDB

---

## Search

- DuckDuckGo Search

---

## Backend

- Python
- LangChain

---

# Engineering Concepts Demonstrated

## AI Engineering

- Transcription Pipelines
- Retrieval-Augmented Generation
- Embeddings
- Hybrid Search
- Reranking
- Prompt Engineering

## Information Retrieval

- BM25
- Semantic Search
- Reciprocal Rank Fusion
- Dense Retrieval

## Software Engineering

- Modular Architecture
- Pipeline Design
- Error Handling
- Analytics
- Monitoring
- API Integration

## Product Engineering

- Interactive UI
- Explainable AI
- Search Transparency
- Performance Tracking

---

# Future Enhancements

- Persistent analytics storage (SQLite/Supabase)
- Multi-video knowledge base
- Speaker diarization
- PDF report export
- DOCX report export
- Team workspaces
- Authentication
- Usage tracking
- Retrieval evaluation framework
- Cloud deployment

---
