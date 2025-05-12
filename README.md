#GraphRAG from a brief literature review

A brief, one-sentence description of your project.

## Description
Implements a GraphRAG pipeline for knowledge extraction and document annotation. This repository converts a literature review into a structured knowledge graph, facilitating semantic enrichment of incoming documents.

## Installation
Instructions on how to get a copy of your project up and running on a local machine.

Clone the repository:
git clone https://github.com/puli83/GraphRag_literature_review.git
cd [GraphRag_literature_review]


Install dependencies (if any):
### Example for Python
pip install -r requirements.txt


## Usage
Use the scripts in your own working environment (i.e. using pycharm, spyder, etc.)

### Structure of files

 1. 00_extract_chunck_from_pdf.py : This file will extract chuncks from pdf to be passed then as nodes to the LLM
 2. 01_extract_entities_from_lit_review.py : This file contains code to load chuncks, converts to nodes, and pass to LLM



#License

MIT License

#Contact

Davide Pulizzotto, Polytechnique Montréal - [mail](davide.pulizzotto@polymtl.ca)

Project Link: [link](https://github.com/puli83/GraphRag_literature_review)
