# LOTR Character Data

This project provides a pipeline for preparing Lord of the Rings character data, setting up a Qdrant Cloud database, and running a Retrieval-Augmented Generation (RAG) chatbot.

## Data Preparation (`/notebooks/scraping.ipynb`)

A Jupyter notebook for collecting and formatting LOTR character information from multiple sources.

- Reads raw CSV files and converts them into JSON objects.
- Fetches additional biography and history data from source URLs.
- Merges all information (general details, biography, and history) into a single JSON file.
- Exports both JSON and CSV versions of the processed dataset.

## Qdrant Setup (`/notebooks/setup-qdrant-complete.ipynb`)

A notebook to configure Qdrant Cloud, embed character data, and store it for retrieval.

- Reads the formatted JSON dataset.
- Converts character information into descriptive text blocks, truncated to fit within token limits.
- Sorts text by token size and batches entries to maximize efficiency.
- Generates embeddings with the Jina API in batches.
- Upserts embeddings, metadata, and truncated text into Qdrant.

## RAG Setup (`/notebooks/rag-setup.ipynb`)

A notebook for building a RAG pipeline with Qdrant and OpenAI.

- Connects to Qdrant Cloud and OpenAI.
- Executes search queries against Qdrant to retrieve relevant character information.
- Formats results into a prompt with both system and user instructions.
- Sends the prompt to the `gpt-4o-mini` model to generate answers.
