# RAG System Architecture Document

## 1. Overview

This document describes the architectural design of a Retrieval-Augmented Generation (RAG) system built from first principles, progressing from a naive baseline to a production-ready implementation with advanced enhancements.

## 2. Dataset Specification

**Source**: RAG Mini Wikipedia dataset from Hugging Face  
**Structure**:
- Training corpus: 3,200 text passages from Wikipedia articles
- Test set: 918 question-answer pairs
- Average passage length: 390 characters
- Passage length range: 1-2,515 characters

**Data Quality Assessment**:
- No missing values detected
- No empty passages after cleaning
- Data distribution shows varied passage lengths suitable for retrieval testing
- Questions span factual, temporal, and contextual query types

## 3. System Architecture

### 3.1 Embedding Layer

**Primary Model**: `all-MiniLM-L6-v2` (Sentence Transformers)
- Embedding dimension: 384
- Rationale: Balance between performance and computational efficiency
- Processes: Encodes both passages and queries into dense vector representations

**Alternative Configuration**: `all-mpnet-base-v2`
- Embedding dimension: 768
- Used for comparison experiments to evaluate impact of embedding richness

### 3.2 Vector Storage

**Initial Implementation**: Milvus Lite
- Lightweight, local vector database
- Schema: ID (INT64), passage (VARCHAR), embedding (FLOAT_VECTOR)
- Index type: FLAT with L2 distance metric

**Production Switch**: FAISS (Facebook AI Similarity Search)
- Reason for migration: Milvus process instability in long-running Colab sessions
- Implementation: IndexFlatL2 for exact nearest neighbor search
- Trade-off: Sacrificed some database features for reliability and in-memory stability

### 3.3 Language Model

**Selected Model**: Google Flan-T5-base
- Parameters: 250M
- Max sequence length: 512 tokens
- Rationale: Free, accessible model suitable for text-to-text generation
- Limitation: Smaller capacity constrains complex reasoning abilities

### 3.4 Retrieval Pipeline

**Baseline Configuration**:
1. Query embedding: Convert user question to 384-dim vector
2. Similarity search: Retrieve top-K most similar passages using L2 distance
3. Context assembly: Concatenate retrieved passages with newline separators
4. Prompt construction: Format context and question for LLM
5. Answer generation: Generate response using Flan-T5

**Initial Parameters**:
- K=3 retrieved passages
- Simple prompting strategy
- No query preprocessing
- No result reranking

## 4. Design Decisions and Trade-offs

### 4.1 Model Selection

**Embedding Model**: Chose sentence-transformers over OpenAI embeddings for:
- Zero API costs
- Reproducibility without external dependencies
- Local execution control

**LLM**: Selected Flan-T5-base over larger models because:
- Free tier accessibility in Google Colab
- Fast inference for experimentation
- Trade-off: Reduced answer quality vs. larger models like GPT-4

### 4.2 Vector Database

**FAISS over Milvus**:
- Advantage: Pure in-memory operation, no server process to crash
- Limitation: Less feature-rich than full vector databases
- Production consideration: For scale, would migrate to managed vector DB (Pinecone, Weaviate)

### 4.3 Retrieval Strategy

**Initial top-K=3**:
- Rationale: Balance between context richness and token budget
- Constraint: Flan-T5's 512 token limit requires conservative passage selection
- Alternative explored: top-K=1,5,10 for performance comparison

## 5. Evaluation Framework

**Metrics**:
- F1 Score: Measures token overlap between predicted and ground truth answers
- Exact Match (EM): Binary metric for perfect answer matches
- SQuAD evaluation library from Hugging Face

**Advanced Evaluation** (Phase 2):
- RAGAs framework for production-readiness assessment
- Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall

## 6. Infrastructure

**Development Environment**:
- Platform: Google Colab (free tier)
- Python: 3.12
- Key dependencies: sentence-transformers, transformers, faiss-cpu, pymilvus

**Reproducibility**:
- All random seeds fixed where applicable
- Complete requirements.txt with pinned versions
- Modular code structure for component swapping

## 7. Scalability Considerations

**Current Limitations**:
- Single-node, in-memory processing
- No distributed retrieval
- Limited to CPU inference

**Production Path**:
- Distributed vector storage (managed service)
- GPU inference for faster generation
- Caching layer for repeated queries
- Load balancing for concurrent requests

## 8. Next Steps

The naive baseline described here serves as the foundation for experimentation with:
1. Prompting strategies
2. Retrieval parameters (top-K, embedding dimensions)
3. Advanced enhancements (reranking, query rewriting)
4. Production optimizations

This architecture prioritizes reproducibility and systematic evaluation over premature optimization, enabling evidence-based improvements in subsequent phases.
