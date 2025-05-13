# GraphRAG from a brief literature review

Implements a GraphRAG pipeline for knowledge extraction and document annotation. This repository converts a literature review into a structured knowledge graph, facilitating semantic enrichment of incoming documents.

## Description

### Automated Knowledge Graph Construction from 5G Scientific Literature
#### Overview

This project implements a workflow for the automated extraction of structured knowledge (entities and their relationships) from unstructured scientific literature, specifically focusing on 5G communication systems literature reviews. It leverages a powerful Large Language Model (LLM) to parse textual content and generate a structured representation of information, forming the foundation for a domain-specific knowledge graph.

The core idea is to transform raw text chunks from PDF documents into actionable, interconnected data points, enabling researchers and analysts to quickly grasp key concepts, identify relationships, and gain insights from large volumes of academic papers that would otherwise be time-consuming to process manually.

#### Features

    Text Chunk Processing: Reads pre-processed text chunks extracted from PDF documents.
    LlamaIndex Integration: Utilizes LlamaIndex to structure text chunks into manageable "nodes" suitable for LLM processing.
    LLM-Powered Entity & Relationship Extraction: Employs a robust LLM (Groq's deepseek-r1-distill-llama-70b by default, with an option for local Ollama models like deepseek-r1:32b) to identify and extract explicit binary relationships between relevant entities from each text node.
    Domain-Specific Extraction: The LLM is guided by a meticulously crafted system prompt, specializing it in the domain of electrical engineering and 5G communication systems, ensuring high-quality and relevant extractions.
    Strict Output Formatting: Enforces strict rules for entity and relationship naming (singular nouns, infinitive verbs, lemmatization) to ensure consistency and facilitate subsequent knowledge graph construction.
    Efficient Parallel Processing: Incorporates concurrent execution with a custom rate limiter to efficiently process a large number of text nodes while respecting API usage limits (e.g., for Groq API).
    Structured Data Output: Saves the enriched nodes, containing the extracted relationships as metadata, into a pickle file for persistence and further analysis or knowledge graph visualization using tools like NetworkX.

#### How it Works

    Input Data Loading: The script begins by loading a dictionary of pre-extracted text chunks from PDFs, typically organized by document title.
    Document and Node Creation: These text chunks are converted into llama_index.core.Document objects and then further processed into smaller nodes using LlamaIndex's SimpleNodeParser. This prepares the text for granular LLM interaction.
    LLM Setup: An LLM instance (currently configured for Groq API) is initialized with specific parameters for optimal performance and response generation within the defined context window.
    Knowledge Extraction Loop:
        For each text node, a meticulously designed system prompt is provided to the LLM. This prompt instructs the LLM to act as an expert in 5G literature reviews and extract only explicit binary relationships in a specified Entity A -> Entity B (relationship type) format.
        The extraction process is parallelized using ThreadPoolExecutor to speed up execution.
        A custom RateLimiter is implemented to manage API call frequency, preventing rate limit errors when interacting with external LLM services.
    Result Storage: The extracted relationships are added as metadata to their respective nodes. All processed nodes, now enriched with structured relationship data, are serialized and saved as a pickle file, timestamped for versioning.

#### Technologies Used

    Python: The primary programming language.
    LlamaIndex: For data ingestion, document processing, and LLM orchestration.
    Groq API: Provides fast and efficient inference for large language models (deepseek-r1-distill-llama-70b).
    Pandas: (Imported, though not directly used for DataFrame manipulation in this snippet, typically used for data handling).
    NetworkX: (Imported, indicating future use for building and analyzing the actual knowledge graph from the extracted relationships).
    Tqdm: For displaying progress bars during long-running operations.
    Pickle: For serializing and deserializing Python objects.
    concurrent.futures: For parallelizing tasks.

## Installation
Instructions on how to get a copy of your project up and running on a local machine.

Clone the repository:
git clone https://github.com/puli83/GraphRag_literature_review.git



Install dependencies (if any):
### Example for Python
pip install -r requirements.txt


## Usage
Use the scripts in your own working environment (i.e. using pycharm, spyder, etc.)

### Structure of files

 1. 00_extract_chunck_from_pdf.py : This file will extract chuncks from pdf to be passed then as nodes to the LLM
 2. 01_extract_entities_from_lit_review.py : This file contains code to load chuncks, converts to nodes, and pass to LLM
 3. 02_normalize_entities.py : This file normalize entites using embedding and clustering amoung nodes. BE SURE TO DOWNLOAD the sentence transformer model and insert into the folder "embedding_sentence_transformer_models"



# License

MIT License

# Contact

Davide Pulizzotto, Polytechnique Montréal - [mail](davide.pulizzotto@polymtl.ca)

Project Link: [link](https://github.com/puli83/GraphRag_literature_review)
