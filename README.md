# Lord of the Rings RAG System

A Retrieval-Augmented Generation (RAG) system built on Lord of the Rings character data, featuring comprehensive evaluation frameworks for both retrieval quality and generation accuracy.

## Overview

This project demonstrates a complete RAG pipeline, from data collection to evaluation. It scrapes Lord of the Rings character information, stores it in a vector database, and provides a question-answering system with robust evaluation metrics.

## Features

- **Data Collection**: Automated scraping and cleaning of LOTR character data
- **Vector Search**: Qdrant-powered semantic search using Jina embeddings
- **RAG Implementation**: GPT-4o-mini for answer generation
- **Retrieval Evaluation**: Performance metrics for the search engine
- **RAG Evaluation**: LLM-as-Judge evaluation using GPT-4o-mini and Claude
- **Containerization**: Docker support for easy deployment

## Tech Stack

- **LLM**: GPT-4o-mini
- **Embeddings**: jina-embeddings-v3 (512 dimensions)
- **Vector Database**: Qdrant (local or cloud)
- **Evaluation**: LLM-as-Judge method with multiple judge models

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Desktop (for containerized setup)
- API Keys:
  - OpenAI API key
  - Anthropic API key (optional, for Claude evaluation)
  - Jina AI API key ([Get it here](https://jina.ai/))

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/bobbykim89/lotr-characters.git
cd lotr-characters
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv
source ./.venv/bin/activate  # On Windows: .\.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=<REPLACE WITH YOUR OPENAI API KEY>
ANTHROPIC_API_KEY=<REPLACE WITH YOUR ANTHROPIC API KEY>
JINA_API_KEY=<REPLACE WITH YOUR JINA AI API KEY>
QDRANT_API_KEY=local_qdrant_key
QDRANT_URL=http://localhost:6333
```

## Usage

### Data Preparation

Scrape and clean Lord of the Rings character data:

```bash
python ./src/scrape_data.py
```

### Setting Up Qdrant

Initialize the vector database with embeddings:

```bash
python ./src/setup_qdrant.py
```

### Retrieval Evaluation

Evaluate the search engine's performance:

```bash
# Run retrieval evaluation (first 100 entries)
python ./src/retrieval_evaluation_run.py

# View results from existing evaluation files
python ./src/retrieval_evaluation_json_only.py
```

Base functions are located at `./src/retrieval_evaluation.py`.

### RAG Evaluation (LLM-as-Judge)

Evaluate the complete RAG system using LLM judges:

```bash
# Evaluate with GPT-4o-mini as judge
python ./src/rag_eval_gpt.py

# Evaluate with Claude as judge (requires Anthropic API key)
python ./src/rag_eval_anthropic.py

# View GPT-4o-mini evaluation results
python ./src/rag_eval_result_only.py

# View Claude evaluation results
python ./src/rag_eval_result_only_anthropic.py
```

Base functions are located at `./src/rag_evaluation_fn.py`.

## Docker Setup

For a containerized deployment with automatic Qdrant setup:

### Prerequisites

- Docker installed and running
- Docker Desktop (recommended)

### Setup Steps

1. **Prepare environment configuration:**
   ```bash
   cp ./qdrant-worker/docker.env.txt ./docker.env
   ```

2. **Make setup script executable:**
   ```bash
   chmod +x ./setup_qdrant.sh
   ```

3. **Run setup:**
   ```bash
   ./setup_qdrant.sh
   ```

4. **Wait for completion** - The script will set up Qdrant and complete the embedding upsert process.

## Important Notes

### Jina AI API Limitations

The Jina AI API has rate limits and may occasionally return internal server errors under heavy load. If you encounter repeated errors:

- Wait a few minutes before retrying
- The service typically recovers quickly
- Consider implementing exponential backoff in production use

## Project Structure

```
.
├── composables/
│   ├── data_processing.py                  # Composable functions for data processing
│   ├── files.py                            # Composable functions for read/save Json files
│   ├── search.py                           # Composable functions for LLM and Search features
├── notebooks/                              # Jupyter notebook files
├── assets/                                 # Jupyter notebook files
├── src/
│   ├── assets/                             # asset files folder (json, csv)
│   ├── scrape_data.py                      # Data collection
│   ├── setup_qdrant.py                     # Vector DB initialization
│   ├── retrieval_evaluation.py             # Retrieval metrics functions
│   ├── retrieval_evaluation_run.py         # Run retrieval tests
│   ├── retrieval_evaluation_json_only.py   # View retrieval results
│   ├── rag_evaluation_fn.py                # RAG evaluation functions
│   ├── rag_eval_gpt.py                     # GPT-4o-mini evaluation
│   ├── rag_eval_anthropic.py               # Claude evaluation
│   ├── rag_eval_result_only.py             # View GPT results
│   └── rag_eval_result_only_anthropic.py   # View Claude results
├── qdrant-worker/
│   ├── runner.sh                           # Docker qdrant-worker script
│   └── docker.env.txt                      # Docker environment template
├── setup_qdrant.sh                         # Docker setup script
├── requirements.txt
├── .env
└── README.md
```

## Evaluation Methodology

### Retrieval Evaluation

Measures the search engine's ability to retrieve relevant character information using standard information retrieval metrics.

### RAG Evaluation (LLM-as-Judge)

Uses advanced language models to evaluate:

1. Relevance — Does the answer directly address the question?
2. Groundedness — Are all facts supported by the provided context (no hallucinations)?
3. Completeness — Does the answer include all key details from the context?
4. Faithfulness — Does it follow the system rules (concise, factual, no invention, admits missing info)?

Multiple judge models (GPT-4o-mini and Claude) provide cross-validation of results.

License
[MIT License](https://github.com/bobbykim89/lotr-characters/blob/master/LICENSE.md)

## Acknowledgments

- Data sourced from multiple Lord of the Rings character databases
- Powered by Jina AI embeddings and Qdrant vector search
- Evaluation framework inspired by LLM-as-Judge methodologies
