import re
import pandas as pd
from tqdm import tqdm
import pickle
import os
import datetime
import networkx as nx
from customTools import clear_gpu_memory, TimeExecution
# os.environ['HF_HOME']
# =============================================================================
#
# %% PATH
pth_import = 'data_Rag_litt_rev'
pth_nodes = 'data_nodes'

# %% import

with open(os.path.join(pth_import, 'dict_extracted_chunks_pdfs.pkl'), 'rb') as file:
    chunks_dict = pickle.load(file)

chunks_dict.keys()
# %% Transform to nopdces
from llama_index.core import Document
#
documents_df = [
    Document(
        text=chunk,
        id_= '_'.join((str(idx), str(idx_chun))),

        metadata={
    #         'IDparversement': row['IDparversement'],

            'title': title,
            'chunk_id': idx_chun,


    #         # 'AwardAmount': row['AwardAmount'],
        }
    )
    for idx, (title, chunk_list) in enumerate(chunks_dict.items()) for idx_chun, chunk in enumerate(chunk_list)
]


len(chunks_dict.items())


from llama_index.core.node_parser import SimpleNodeParser
# Instantiate the SimpleNodeParser
node_parser = SimpleNodeParser()

# Get nodes from the documents
nodes = node_parser.get_nodes_from_documents(documents_df)

len(nodes)
# =============================================================================
# %% build llm
# from groq import Groq

# Get the token from an environment variable
api_token = os.environ.get("GROQ_API_TOKEN")



##### test llamaindex class
# https://docs.llamaindex.ai/en/stable/examples/llm/groq/
from llama_index.llms.groq import Groq
llm = Groq(
    # model='llama3:instruct',  # or another model you've pulled in Ollama
    model='deepseek-r1-distill-llama-70b',
    api_key=api_token,
    # context_window=4096,
    context_window=8192,
    num_output=1024,  # note: this is equivalent to max_new_tokens
    # temperature=0.0,
    temperature=0.5,
    # top_p=0.00,
    top_p=0.95,
    request_timeout=120,
    repeat_penalty=1.1,  # note: this is equivalent to repetition_penalty
)

#####################
#####################
# =============================================================================
# ############################ deepseek with ollama ######################
# =============================================================================
# from GRAPH_RAG.customTools import clear_gpu_memory, TimeExecution


# from llama_index.core import Settings
# Settings.num_output = 1024
# from llama_index.llms.ollama import Ollama
# from llama_index.core import PromptTemplate
# llm = Ollama(
#     # model='llama3:instruct',  # or another model you've pulled in Ollama
#     model='deepseek-r1:32b',  # or another model you've pulled in Ollama
#     # context_window=4096,
#     context_window=8192,
#     num_output=1024,  # note: this is equivalent to max_new_tokens
#     # temperature=0.0,
#     temperature=0.7,
#     # top_p=0.00,
#     top_p=0.95,
#     request_timeout=120,
#     repeat_penalty=1.1,  # note: this is equivalent to repetition_penalty
# )
# query_str =
# prompt = "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
test_response = llm.complete("Which is the main city in Sicily?")

dir(test_response)
test_response.text



#usage:
# with TimeExecution():
#     test = get_completion(nodes[1], llm)
# test.text




# %% execute extraction  with llm
from KG_Extractor import EntityExtractor

SYSTEM_PROMPT = """You are an expert system specialized in extracting entities and relationships from scientific literature in the domain of electrical engineering and 5G communication systems, with a specific focus on literature reviews. Your goal is to analyze paragraphs from 5G literature reviews and extract structured information to contribute to building a comprehensive domain-specific knowledge graph.

Given a paragraph from a scientific paper's literature review about 5G technology:

Task:
Analyze the provided paragraph and extract all explicit binary relationships between relevant entities within the domain.

Output Format:
Present the extracted relationships under the heading "RELATIONSHIPS:", with each relationship on a new line in the following format:
Entity A -> Entity B (relationship type)

Rules for Extraction and Formatting:
1.  **Entity Identification:** Identify all relevant entities such as technologies, concepts, applications, standards, hardware components, stakeholders, etc., that are mentioned in the paragraph and pertain to the 5G domain. (Note: You identify these internally to find relationships; they are not listed separately in the output).
2.  **Relationship Identification:** Identify direct relationships that are explicitly stated in the text linking two identified entities. Focus only on relationships entirely contained within the paragraph.
3.  **Binary Relationships:** Extract only relationships involving exactly two entities.
4.  **Explicitness:** Do NOT infer relationships that are not directly stated in the text.
5.  **Exhaustiveness:** Aim to capture as many explicit binary relationships as possible from the provided paragraph.
6.  **Consistency:** Maintain consistent naming for the same entity across different relationships extracted *from this specific paragraph*.
7.  **Naming Convention:**
    * All entity names and relationship labels must be in English.
    * Entity names (nouns) should be in singular form and lemmatized (e.g., "technology" not "technologies", "application" not "applications", "network" not "networks").
    * Relationship labels (verbs) should be in the infinitive form and lemmatized where appropriate (e.g., "use" not "uses" or "using", "enable" not "enables" or "enabled", "provide" not "provides").
    * Use clear, concise, and descriptive labels for relationship types (e.g., "enable", "include", "use", "provide", "require", "is a type of", "improve").
8.  **Output Content:** Provide ONLY the "RELATIONSHIPS:" section followed immediately by the list of relationships. Do NOT include a separate list of entities, introductory/concluding remarks, or any other explanatory text.
9.  **Formatting:** Adhere strictly to the specified `Entity A -> Entity B (relationship type)` format for each relationship line.
"""

extractor = EntityExtractor(llm_instance=llm, system_prompt=SYSTEM_PROMPT, max_workers=2)

#test = extractor.extract_elements_from_chunks([nodes[0]], max_requests_per_minute=28)
#

with TimeExecution():
    elements = extractor.extract_elements_from_chunks(nodes, max_requests_per_minute=28)


len(elements)
elements[20]
nodes[20].metadata["deepseek_llama_370_answer"]


# %%
# Save nodes

with open(os.opath.join(pth_nodes, f'nodes_lit_rev_{str(datetime.datetime.now().strftime("%Y-%m-%d"))}.pkl'), 'wb') as f:
    pickle.dump(nodes, f)


# %% NOTES:


# from llama_index.core.llms import ChatMessage

# messages = [
#     ChatMessage(
#         role="system", content="You are a professor of history"
#     ),
#     ChatMessage(role="user", content="Which is the capital of Italy?"),
# ]
# resp = llm.chat(messages)
# dir(resp)
# dir(resp.message)
# resp.message.blocks

# clear_gpu_memory()


#########################################################################
#######################
# =============================================================================
# priompt never used:SYSTEM_PROMPT = """You are an expert system for entity and relationship extraction in the domain of electrical engineering and 5G communication systems. Given a pargarph of a scientific paper in 5G, paticulary, a paragraphn of a literature review about 5G tehcnology, identify:
#
# 1. All distinct entities (technologies, concepts, objects, applications, etc.)
# 2. Direct relationships between entities that are explicitly stated in the text
# 3. A score, from 0 to 3, that evaluates the intensity of the relationship
#
# Format your response as:
#
# RELATIONSHIPS:
#   Entity A -> Entity B (relationship type)[score]
#   Entity C -> Entity D (relationship type)[score]
# [...]
#
# Rules:
# - Only include relationships explicitly stated in the text
# - Use clear, concise relationship labels
# - Maintain consistent entity names
# - Entity names and relationship labels should be in English only, lemmatized, with nouns in singular form and verbs in infinitive form
# - Order entities in each relationship to follow logical causation
# - Output format:
#     -- Provide only RELATIONSHIPS, no ENTITIES
#     -- Be as exhaustive as possible in converting the whole text  in entities and relationships
# """
# =============================================================================



