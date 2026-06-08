# DocTabularize — Document Intelligence Refinery

**Advanced PDF Table & Structured Data Extraction Pipeline**  
*Target-aware, self-improving, and verifiable extraction using vision LLMs + clustering*

---

## Overview

DocTabularize is a sophisticated document intelligence system that automatically extracts structured tabular data from complex PDFs (invoices, financial reports, research papers, etc.). 

It combines **computer vision**, **semantic clustering**, **LLM-powered schema discovery**, and **multi-stage verification** to achieve high accuracy even on challenging, unstructured documents.

---

## How It Works (Step by Step)

1. **Ingestion**  
   PDF pages are extracted. Low-content pages (<15 characters) are automatically skipped.

2. **Semantic Understanding**  
   Every page is converted into embeddings using Sentence Transformers. Similar pages are grouped in vector space.

3. **Intelligent Clustering**  
   HDBSCAN clusters similar pages together. Noise pages are filtered out.

4. **Schema Discovery**  
   For each cluster, a language model analyzes representative samples and proposes an optimal JSON schema. The system remembers high-performing schemas across runs (genetic memory).

5. **Vision-Based Extraction**  
   Pages are converted to high-resolution images and sent to a vision-language model (Qwen2.5-VL or compatible). Extraction is **target-score aware** — the model retries until it meets quality thresholds.

6. **Multi-Layer Verification**  
   - **Semantic fidelity** via Cross-Encoder  
   - **Structural completeness**  
   - **SQL compliance** (DuckDB)  
   Each row receives confidence metadata.

7. **Optional Fact Verification**  
   Cross-checks extracted facts against original document text.

8. **Cross-Cluster Reconciliation**  
   Attempts to link related data across different clusters.

9. **Output**  
   Rich JSON with per-page scores, row-level confidence, verification flags, and pipeline statistics.

---

## Features

- **Modular & Production-Ready** architecture (`src/doctabularize`)
- **Target-score aware extraction** with automatic retries
- **Self-improving schema memory** (genetic evolution)
- **Per-row confidence metadata**
- **Support for local (Ollama) and OpenAI-compatible APIs**
- **Highly configurable** via TOML
- **Robust JSON parsing & recovery**

---

## Requirements

- Python 3.11+
- Ollama (recommended for local use) with `qwen2.5vl:7b`
- See `requirements.txt` for full dependencies

---

## Setup

```bash
git clone https://github.com/DomArruda/DocTabularize.git
cd DocTabularize

# Install in editable mode
pip install -e .

# Pull the vision model
ollama pull qwen2.5-vl:7b
