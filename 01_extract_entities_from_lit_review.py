import re
import pandas as pd
from tqdm import tqdm
import pickle
import os
import networkx as nx
from customTools import clear_gpu_memory, TimeExecution
# os.environ['HF_HOME']
# =============================================================================
#
pth_import = 'data_Rag_litt_rev'

with open(os.path.join(pth_import, 'dict_extracted_chunks_pdfs.pkl'), 'rb') as file:
    chunks_dict = pickle.load(file)

chunks_dict.keys()


import datetime

from llama_index.core import Document
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

import datetime
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


# Define a function to handle a single completion
def get_completion(node, llm):
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

    # query_wrapper_prompt = PromptTemplate(
    #     "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
    # )

    # llm.query_wrapper_prompt = query_wrapper_prompt
    # llm.query_wrapper_prompt = "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
    return llm.complete(f""""<<system prompt>>\n + {SYSTEM_PROMPT} + <</system prompt>>\n\n <<user query>>Extract entities and relationships from the following text: {node.text} <</user query>>""")

#usage:
with TimeExecution():
    test = get_completion(nodes[1], llm)
test.text



# clear_gpu_memory()

####################### this does not work really
# import time
# import concurrent.futures
# from threading import Semaphore

# # Define a semaphore to limit the number of concurrent requests
# max_requests_per_minute = 28
# semaphore = Semaphore(max_requests_per_minute)

# # Function to process a single node with rate limiting
# def process_single_node(node, llm):
#     clear_gpu_memory()

#     # Acquire the semaphore to enforce rate limiting
#     semaphore.acquire()

#     try:
#         # Call the get_completion function
#         answer = get_completion(node, llm)

#         # Add the cleaned text as metadata to the node
#         node.metadata["deepseek_llama_370_answer"] = answer
#     finally:
#         # Release the semaphore after a delay to enforce the rate limit
#         time.sleep(60 / max_requests_per_minute)
#         semaphore.release()

#     return answer

# # Function to process all nodes in parallel
# def process_nodes_in_parallel(nodes, llm):
#     results = []

#     # Use ThreadPoolExecutor to process nodes in parallel
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # Submit tasks to the executor
#         futures = [executor.submit(process_single_node, node, llm) for node in nodes]

#         # Collect results as they complete
#         for future in concurrent.futures.as_completed(futures):
#             try:
#                 result = future.result()
#                 results.append(result)
#             except Exception as e:
#                 print(f"An error occurred: {e}")

#     return results

# results = process_nodes_in_parallel(nodes, llm)
# len(results)

# results[1]
####################

##################### tried 2025-05-05
import concurrent.futures
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Define a rate-limiter class to control the processing rate
class RateLimiter:
    def __init__(self, max_per_minute):
        self.max_per_minute = max_per_minute
        self.interval = 60 / max_per_minute  # Time interval between tasks
        self.last_time = time.time()

    def wait(self):
        current_time = time.time()
        elapsed_time = current_time - self.last_time
        if elapsed_time < self.interval:
            time.sleep(self.interval - elapsed_time)
        self.last_time = time.time()

# Modify the process_single_node function to include rate limiting
def process_single_node(node, llm, rate_limiter=None):
    if rate_limiter:
        rate_limiter.wait()  # Wait to ensure rate limit is respected
    # Process a single node
    answer = get_completion(node, llm)
    # Add the cleaned text as metadata to the node
    node.metadata["deepseek_llama_370_answer"] = answer
    return answer

# Function to extract elements from chunks with rate limiting
def extract_elements_from_chunks(nodes, llm, max_per_minute=30):
    elements = []
    # Create a rate limiter for 28 nodes per minute
    rate_limiter = RateLimiter(max_per_minute=max_per_minute)

    # Create a partial function that includes llm and rate_limiter
    from functools import partial
    process_node = partial(process_single_node, llm=llm, rate_limiter=rate_limiter)

    # Use ThreadPoolExecutor to process nodes concurrently
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all nodes to the executor
        futures = [executor.submit(process_node, node) for node in nodes]

        # Use tqdm to monitor progress
        with tqdm(total=len(futures), desc="Processing nodes") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    elements.append(result)
                except Exception as e:
                    print(f"An error occurred: {e}")
                pbar.update(1)

    return elements

# Example usage
with TimeExecution():
    elements = extract_elements_from_chunks(nodes, llm, max_per_minute=28)


len(elements)
elements[20]
nodes[20].text
# nodes[[134, 224, 272, 466, 674, 699]]

# len(nodes)
# len([x for idx,x in enumerate(nodes) if idx in [134, 224, 272, 466, 674, 699]] )

# with TimeExecution():
#     elements = extract_elements_from_chunks( [x for idx,x in enumerate(nodes) if idx in [134, 224, 272, 466, 674, 699]] , llm) #


# # llm.generate_kwargs = {"do_sample": False,
# #                        # "temperature": 0.6,
# #                        "temperature": 0.0,
# #                         "top_p": 0.0,
# #                        }

# # import random


# for idx, nod_ in  enumerate([x for idx,x in enumerate(nodes) if idx in [134, 224, 272, 466, 674, 699]]):
#     print(idx)
#     print(nod_.metadata.get('entities_relationship'))

correct_answer = get_completion(nodes[351], llm)
# nodes[351].metadata['entities_relationship'] = correct_answer

# os.listdir()
# # Save nodes
# with open('RAG_EXTRACT_TEST/nodes_HF-2024-12-09-bis.pkl', 'wb') as f:
#     pickle.dump(nodes, f)


# Save nodes
with open('RAG_EXTRACT_TEST/nodes_HF-2024-12-09.pkl', 'wb') as f:
    pickle.dump(nodes, f)

# Save nodes
with open('RAG_EXTRACT_TEST/elements_HF.pkl', 'wb') as f:
    pickle.dump(elements, f)




# Load nodes from pickle file
with open('RAG_EXTRACT_TEST/nodes_HF-2024-12-09-bis.pkl', 'rb') as f:  # Note 'rb' for reading binary
    nodes = pickle.load(f)

len(nodes)
nodes[0]
nodes[0].text
documents_df[0]

# #usage:
with TimeExecution():
    test = get_completion(nodes[0], llm)
    test = get_completion(nodes[200], llm)
test.text

nodes[0]


