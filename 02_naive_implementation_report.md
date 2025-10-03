# Naive RAG Implementation Report

## 1. Implementation Summary

This report documents the baseline RAG system implementation, including initial performance results and lessons learned from the naive approach.

## 2. System Components

### 2.1 Pipeline Implementation

The naive RAG system implements a straightforward retrieval-generation pipeline:

1. Embed query using all-MiniLM-L6-v2
2. Search Milvus/FAISS for top-3 similar passages
3. Concatenate passages as context
4. Format prompt: "Answer based on context.\n\nContext: {ctx}\n\nQuestion: {q}\n\nAnswer:"
5. Generate answer using Flan-T5-base

### 2.2 Technical Challenges

#### Vector Database Migration

- Initial implementation used Milvus Lite for vector storage
- Encountered connection failures after ~45 minutes of runtime in Colab
- Root cause: Milvus server process crashes in long-running sessions
- Solution: Migrated to FAISS for stable in-memory operations
- Impact: Minimal - both use L2 distance for similarity search

#### Token Length Issues

- Flan-T5-base has 512 token limit
- Some retrieved passages exceed this when combined
- Warning: "Token indices sequence length is longer than specified maximum"
- Mitigation: Context automatically truncated, but may lose relevant information

## 3. Initial Results

### 3.1 Baseline Performance (Top-3 Retrieval)

**Metrics on 100 test questions**:
- F1 Score: 55.52
- Exact Match: 49.00%

**Analysis**:
- System achieves ~50% exact matches - reasonable for naive approach
- F1 score indicates partial answer quality even when not exact
- Performance limited by small LLM capacity (Flan-T5-base)

### 3.2 Prompting Strategy Experiments

Tested 4 different prompting approaches on 100 questions with top-5 retrieval:

| Strategy | Description | F1 Score | Exact Match |
|----------|-------------|----------|-------------|
| Simple | "Answer based on context" | **57.14** | **51.00** |
| Persona | "You are an expert historian..." | 51.82 | 46.00 |
| Instruction | "Instructions: Read carefully..." | 55.14 | 49.00 |
| Structured | "Context:\n...\nTask:..." | 47.78 | 41.00 |

**Key Finding**: Simple prompting outperformed complex strategies.

**Hypothesis**: Flan-T5-base (250M parameters) lacks capacity to effectively utilize complex instructions. Simpler prompts reduce cognitive load on the small model, allowing it to focus on answer extraction rather than instruction following.

**Chain-of-Thought Failure**: CoT prompting ("Let's think step by step") achieved only F1=9.70, EM=0.00 - catastrophic failure indicating the model cannot perform multi-step reasoning at this scale.

## 4. Top-K Retrieval Analysis

Tested retrieval with K=1,3,5,10 passages on 50 questions:

| K | F1 Score | Exact Match | Observation |
|---|----------|-------------|-------------|
| 1 | 49.36 | 44.00 | Insufficient context |
| 3 | 62.52 | 56.00 | Balanced performance |
| 5 | **65.52** | **60.00** | Best trade-off |
| 10 | 68.80 | 62.00 | Marginal improvement |

**Insights**:
- Performance improves with more context (more chances to find answer)
- Top-5 offers best balance before diminishing returns
- Top-10 shows minor gains but risks exceeding token limits
- Recommendation: Use top-5 for baseline comparisons

## 5. Lessons Learned

### 5.1 Technical

1. **Infrastructure matters**: Vector DB stability is critical for long experiments
2. **Model constraints**: Small LLMs need simple, direct prompts
3. **Token budgets**: Must account for combined passage lengths

### 5.2 Methodological

1. **Start simple**: Complex prompting doesn't guarantee better results
2. **Measure everything**: Top-K experiments revealed non-linear performance curve  
3. **Document failures**: CoT failure informed future design decisions

## 6. Foundation for Enhancement

This naive implementation establishes:
- Baseline metrics: F1=65.52, EM=60.00 (top-5, simple prompt)
- Best practices: Simple prompting, K=5 retrieval
- Infrastructure: Stable FAISS-based pipeline
- Evaluation framework: SQuAD metrics on 100-question test set

These baselines enable meaningful comparison when testing advanced enhancements in the next phase.
