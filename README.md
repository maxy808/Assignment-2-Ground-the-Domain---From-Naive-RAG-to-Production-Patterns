# RAG System: From Naive Implementation to Production-Ready Enhancements

**Course**: NLX and LLM - Assignment 2  
**Author**: Christine Ma 
**Date**: October 2025

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers factual questions using Wikipedia passages. Starting from a naive baseline, we systematically add production-ready enhancements and evaluate performance improvements.

## Quick Results

| System | F1 Score | Exact Match |
|--------|----------|-------------|
| Naive Baseline | 65.52 | 60.00 |
| **Enhanced (Best)** | **69.54** | **64.00** |

**Key Finding**: Combined query rewriting + cross-encoder reranking provides +4.02 F1 improvement.

## Repository Structure
```
.
├── README.md                              # This file
├── requirements.txt                       # Dependencies
├── 01_architecture_document.md           # System design (750 words)
├── 02_naive_implementation_report.md     # Baseline results (500 words)
├── 03_enhanced_system_analysis.md        # Enhancements (750 words)
├── 04_evaluation_technical_report.md     # Full evaluation (2450 words)
├── notebooks/
│   └── Assignment_2.ipynb                # Complete implementation
└── results/
├── *.csv                             # Experimental results
├── *.png                             # Visualizations
└── *.pkl                             # RAGAs outputs
```
## Quick Start

### Installation
```bash
# Clone repository
git clone [your-repo-url]
cd rag-system-assignment2

# Install dependencies
pip install -r requirements.txt
```
## Run Experiments
### Google Colab (Recommended):

Upload notebooks/Assignment_2.ipynb to Colab
Run all cells sequentially
Expected runtime: ~90 minutes

### Local Jupyter:
bashjupyter notebook notebooks/Assignment_2.ipynb

## System Architecture
```
Embeddings: all-MiniLM-L6-v2 (384-dim)
Vector DB: FAISS (migrated from Milvus for stability)
LLM: Flan-T5-base
Enhancements: Query rewriting + Cross-encoder reranking
```
## Key Experiments
### 1. Prompting Strategies
Simple prompting beat complex strategies by 9.36 F1 points with small LLM.
### 2. Top-K Retrieval
Optimal at K=5 (F1=65.52) - diminishing returns after.
### 3. Embedding Dimensions
384-dim outperformed 768-dim by 7.22 F1 points (unexpected).
### 4. Enhancements

Reranking: +2.02 F1
Query Rewriting: +2.00 F1
Combined: +4.02 F1

## Documentation
```
Complete analysis in individual reports:

Architecture Document - System design
Naive Implementation - Baseline results
Enhanced System - Improvements
Evaluation Report - Full analysis
```
## Key Results
### RAGAs Evaluation (30 questions)
| Metric | Naive | Enhanced | Change |
|--------|-------|----------|--------|
| Context Precision | 0.7327 | **0.8583** | **+0.1256** |
| Faithfulness | 0.7667 | 0.7667 | 0.0000 |
| Answer Relevancy | 0.7648 | 0.7759 | +0.0111 |
Most significant improvement: Context Precision (+17% relative gain)
## Production Deployment
### Recommended Configuration:
```
pythonembedding_model = "all-MiniLM-L6-v2"  # 384-dim
vector_db = "FAISS"
retrieval_k = 10  # Initial retrieval
reranking_k = 3   # After cross-encoder
query_rewriting = True
```
### Performance Targets:

F1 Score: 69.54 → 70+ (with larger LLM)
Latency: 600ms → <500ms (with optimization)
Context Precision: 0.8583 ✓

### Technical Challenges Solved
```
Milvus Instability: Migrated to FAISS after crashes
Token Limits: Optimized top-K selection for 512-token budget
Small LLM: Simple prompting strategy proved most effective
```
## Files Generated
```
master_experiment_results.csv - All experiments
experiment_comparison.png - Performance charts
ragas_comparison.png - RAGAs visualization
Full list in /results folder
```
## Requirements
```
txtsentence-transformers==3.0.1
faiss-cpu==1.8.0
datasets==2.19.0
transformers==4.41.0
evaluate==0.4.1
ragas==0.1.9
pandas==2.2.2
numpy==2.0.2
See requirements.txt for complete list.
```
## Dataset
Source: RAG Mini Wikipedia

3,200 passages
918 test questions

## AI Usage
All AI assistance documented in ai_usage_log.md. Primary tool: Claude.ai (Sonnet 4.5) for debugging, code structure, and documentation assistance. All code independently tested and verified.
## References

Sentence Transformers
RAGAs Framework
FAISS

## License
Educational project for Advanced NLP course.
## Contact

GitHub: maxy808
Email: xm3@andrew.cmu.edu


Best Performance: F1=69.54, EM=64.00 with combined enhancements

**Now it's properly in markdown format - copy everything inside the code block!**
