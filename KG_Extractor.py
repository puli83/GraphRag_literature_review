# %% Imports
import time
import concurrent.futures
from functools import partial
from tqdm import tqdm

# It's assumed that 'llm' and 'node' objects are defined elsewhere
# and have 'complete' and 'text' / 'metadata' attributes respectively.
# For example, a placeholder Node class:
# class Node:
#     def __init__(self, text):
#         self.text = text
#         self.metadata = {}

class EntityExtractor:
    """
    A class to extract entities and relationships from text chunks using an LLM,
    with rate limiting and concurrent processing.
    """

    class _RateLimiter:
        """
        A helper class to control the processing rate of tasks.
        It ensures that tasks are not executed more frequently than a specified maximum per minute.
        """
        def __init__(self, max_per_minute):
            """
            Initializes the RateLimiter.

            Args:
                max_per_minute (int): The maximum number of tasks allowed per minute.
            """
            if max_per_minute <= 0:
                raise ValueError("max_per_minute must be positive.")
            self.max_per_minute = max_per_minute
            self.interval = 60.0 / max_per_minute  # Time interval in seconds between tasks
            self.last_time = time.monotonic() # Use monotonic clock for measuring intervals

        def wait(self):
            """
            Waits if necessary to ensure the rate limit is not exceeded.
            This method should be called before executing a rate-limited task.
            """
            current_time = time.monotonic()
            elapsed_time = current_time - self.last_time
            wait_time = self.interval - elapsed_time

            if wait_time > 0:
                time.sleep(wait_time)

            self.last_time = time.monotonic() # Update last_time after waiting or if no wait was needed


    def __init__(self, llm_instance, system_prompt, max_workers=2):
        """
        Initializes the EntityExtractor.

        Args:
            llm_instance: An instance of the language model to be used for completions.
                          This instance must have a `complete(prompt_string)` method.
            max_workers (int, optional): The maximum number of threads to use for
                                         concurrent processing. Defaults to 2.
        """
        self.llm = llm_instance
        self.max_workers = max_workers
        self.system_prompt = system_prompt # Store the provided system prompt

    def _get_completion(self, node):
        """
        Sends a request to the language model to extract entities and relationships
        from the text of a given node.

        Args:
            node: A node object with a 'text' attribute containing the text to process.

        Returns:
            str: The processed text (answer) from the language model.
        """
        # Constructing the prompt for the LLM.
        # It's important that the LLM is configured to understand this prompt structure.
        # The original code had a specific way of formatting this, which is replicated here.
        prompt = f"<<system prompt>>\n{self.system_prompt}\n<</system prompt>>\n\n<<user query>>Extract entities and relationships from the following text: {node.text}<</user query>>"

        # Assuming the llm object has a 'complete' method that takes the prompt string.
        return self.llm.complete(prompt)

    def _process_single_node(self, node, rate_limiter=None):
        """
        Processes a single node: gets completion from LLM and updates node metadata.
        Includes rate limiting.

        Args:
            node: The node object to process. It should have 'text' and 'metadata' attributes.
            rate_limiter (EntityExtractor._RateLimiter, optional): An instance of the rate limiter.

        Returns:
            str: The result (answer) from the language model.
        """
        if rate_limiter:
            rate_limiter.wait()  # Wait to ensure rate limit is respected

        answer = self._get_completion(node)

        # Add the cleaned text as metadata to the node
        # Ensure node.metadata is a dictionary
        if not hasattr(node, 'metadata') or node.metadata is None:
            node.metadata = {}
        node.metadata["llm_extracted_answer"] = str(answer) # Store the answer
        return answer

    def extract_elements_from_chunks(self, nodes, max_requests_per_minute=30):
        """
        Extracts elements from a list of text nodes using concurrent processing
        and rate limiting.

        Args:
            nodes (list): A list of node objects. Each node should have a 'text' attribute
                          and will have its 'metadata' attribute updated with the LLM's answer.
            max_requests_per_minute (int, optional): The maximum number of requests to send
                                                     to the LLM per minute. Defaults to 30.

        Returns:
            list: A list of results (answers) from the language model for each node.
        """
        if not nodes:
            return []

        elements = []
        # Create a rate limiter
        rate_limiter = self._RateLimiter(max_per_minute=max_requests_per_minute)

        # Create a partial function that includes the rate_limiter.
        # The 'llm' instance is already part of 'self'.
        process_node_with_limiter = partial(self._process_single_node, rate_limiter=rate_limiter)

        # Use ThreadPoolExecutor to process nodes concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all nodes to the executor
            # The process_node_with_limiter expects only 'node' as its first argument now.
            futures = [executor.submit(process_node_with_limiter, node) for node in nodes]

            # Use tqdm to monitor progress
            print(f"Starting processing of {len(nodes)} nodes with max {self.max_workers} workers and {max_requests_per_minute} RPM limit.")
            with tqdm(total=len(futures), desc="Processing nodes") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        elements.append(str(result)) # Ensure result is string
                    except Exception as e:
                        print(f"An error occurred while processing a node: {e}")
                        elements.append(None) # Append None or some error indicator
                    pbar.update(1)

        print(f"Finished processing. Extracted {len(elements)} elements.")
        return elements

# Example Usage (assuming you have an LLM and Node class defined):
if __name__ == '__main__':
    # --- Mock LLM and Node classes for demonstration ---
    class MockLLM:
        def __init__(self, name="mock_llm"):
            self.name = name
            self._call_count = 0

        def complete(self, prompt_str):
            self._call_count += 1
            # print(f"\n--- LLM '{self.name}' received prompt ---\n{prompt_str[:200]}...\n--- End of LLM Prompt ---")
            time.sleep(0.5) # Simulate network latency or processing time
            return f"Mocked LLM response for prompt. Call count: {self_call_count}"

    class MockNode:
        def __init__(self, id, text):
            self.id = id
            self.text = text
            self.metadata = {} # Initialize metadata

        def __repr__(self):
            return f"Node(id={self.id}, text='{self.text[:30]}...', metadata={self.metadata})"
    # --- End of Mock classes ---

    # 1. Create an instance of your LLM
    my_llm = MockLLM(name="deepseek_llama_370_mock")

    # 2. Create an instance of the EntityExtractor
    # You can pass the llm instance during initialization
    extractor = EntityExtractor(llm_instance=my_llm, max_workers=3)

    # 3. Prepare your nodes (list of text chunks)
    sample_nodes = [
        MockNode(id=1, text="5G technology enables enhanced mobile broadband and ultra-reliable low-latency communications."),
        MockNode(id=2, text="Network slicing is a key feature of 5G, allowing virtual networks to be created on a shared physical infrastructure."),
        MockNode(id=3, text="Massive MIMO improves spectral efficiency in 5G systems."),
        MockNode(id=4, text="Edge computing reduces latency by processing data closer to the user, which is crucial for many 5G applications."),
        MockNode(id=5, text="The 3GPP defines standards for 5G New Radio (NR)."),
        # Add more nodes as needed
    ]

    # 4. Call the extraction method
    # You can specify the rate limit per minute here
    print("Starting extraction...")
    extracted_data = extractor.extract_elements_from_chunks(
        nodes=sample_nodes,
        max_requests_per_minute=10 # Example: 10 requests per minute
    )

    print("\n--- Extracted Data ---")
    for i, data in enumerate(extracted_data):
        print(f"Node {i+1} Result: {data}")

    print("\n--- Nodes after processing (metadata check) ---")
    for node in sample_nodes:
        print(node)

    # --- Example with a higher rate (closer to original 30 RPM) ---
    # extractor_high_rate = EntityExtractor(llm_instance=my_llm, max_workers=2)
    # sample_nodes_2 = [MockNode(id=i, text=f"This is sample text for node {i}.") for i in range(6, 11)]
    # print("\nStarting extraction with higher rate...")
    # extracted_data_high_rate = extractor_high_rate.extract_elements_from_chunks(
    #     nodes=sample_nodes_2,
    #     max_requests_per_minute=60 # Simulate 60 RPM, so 1 per second
    # )
    # print("\n--- Extracted Data (High Rate) ---")
    # for i, data in enumerate(extracted_data_high_rate):
    #     print(f"Node {i+1} Result: {data}")
    # print("\n--- Nodes after processing (High Rate - metadata check) ---")
    # for node in sample_nodes_2:
    #     print(node)




# %%old code: extractor# %% function to operate in parrallel
# =============================================================================
#
# ##################### tried 2025-05-05
# from tqdm import tqdm
# import concurrent.futures
# import time
# # from functools import partial
# from concurrent.futures import ThreadPoolExecutor
#
#
#
# # Define a function to handle a single completion
# def get_completion(node, llm):
#     SYSTEM_PROMPT = """You are an expert system specialized in extracting entities and relationships from scientific literature in the domain of electrical engineering and 5G communication systems, with a specific focus on literature reviews. Your goal is to analyze paragraphs from 5G literature reviews and extract structured information to contribute to building a comprehensive domain-specific knowledge graph.
#
#     Given a paragraph from a scientific paper's literature review about 5G technology:
#
#     Task:
#     Analyze the provided paragraph and extract all explicit binary relationships between relevant entities within the domain.
#
#     Output Format:
#     Present the extracted relationships under the heading "RELATIONSHIPS:", with each relationship on a new line in the following format:
#     Entity A -> Entity B (relationship type)
#
#     Rules for Extraction and Formatting:
#     1.  **Entity Identification:** Identify all relevant entities such as technologies, concepts, applications, standards, hardware components, stakeholders, etc., that are mentioned in the paragraph and pertain to the 5G domain. (Note: You identify these internally to find relationships; they are not listed separately in the output).
#     2.  **Relationship Identification:** Identify direct relationships that are explicitly stated in the text linking two identified entities. Focus only on relationships entirely contained within the paragraph.
#     3.  **Binary Relationships:** Extract only relationships involving exactly two entities.
#     4.  **Explicitness:** Do NOT infer relationships that are not directly stated in the text.
#     5.  **Exhaustiveness:** Aim to capture as many explicit binary relationships as possible from the provided paragraph.
#     6.  **Consistency:** Maintain consistent naming for the same entity across different relationships extracted *from this specific paragraph*.
#     7.  **Naming Convention:**
#         * All entity names and relationship labels must be in English.
#         * Entity names (nouns) should be in singular form and lemmatized (e.g., "technology" not "technologies", "application" not "applications", "network" not "networks").
#         * Relationship labels (verbs) should be in the infinitive form and lemmatized where appropriate (e.g., "use" not "uses" or "using", "enable" not "enables" or "enabled", "provide" not "provides").
#         * Use clear, concise, and descriptive labels for relationship types (e.g., "enable", "include", "use", "provide", "require", "is a type of", "improve").
#     8.  **Output Content:** Provide ONLY the "RELATIONSHIPS:" section followed immediately by the list of relationships. Do NOT include a separate list of entities, introductory/concluding remarks, or any other explanatory text.
#     9.  **Formatting:** Adhere strictly to the specified `Entity A -> Entity B (relationship type)` format for each relationship line.
#
#     """
#
#     # query_wrapper_prompt = PromptTemplate(
#     #     "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
#     # )
#
#     # llm.query_wrapper_prompt = query_wrapper_prompt
#     # llm.query_wrapper_prompt = "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
#     return llm.complete(f""""<<system prompt>>\n + {SYSTEM_PROMPT} + <</system prompt>>\n\n <<user query>>Extract entities and relationships from the following text: {node.text} <</user query>>""")
#
#
# # Define a rate-limiter class to control the processing rate
# class RateLimiter:
#     def __init__(self, max_per_minute):
#         self.max_per_minute = max_per_minute
#         self.interval = 60 / max_per_minute  # Time interval between tasks
#         self.last_time = time.time()
#
#     def wait(self):
#         current_time = time.time()
#         elapsed_time = current_time - self.last_time
#         if elapsed_time < self.interval:
#             time.sleep(self.interval - elapsed_time)
#         self.last_time = time.time()
#
# # Modify the process_single_node function to include rate limiting
# def process_single_node(node, llm, rate_limiter=None):
#     if rate_limiter:
#         rate_limiter.wait()  # Wait to ensure rate limit is respected
#     # Process a single node
#     answer = get_completion(node, llm)
#     # Add the cleaned text as metadata to the node
#     node.metadata["deepseek_llama_370_answer"] = answer
#     return answer
#
# # Function to extract elements from chunks with rate limiting
# def extract_elements_from_chunks(nodes, llm, max_per_minute=30):
#     elements = []
#     # Create a rate limiter for 28 nodes per minute
#     rate_limiter = RateLimiter(max_per_minute=max_per_minute)
#
#     # Create a partial function that includes llm and rate_limiter
#     from functools import partial
#     process_node = partial(process_single_node, llm=llm, rate_limiter=rate_limiter)
#
#     # Use ThreadPoolExecutor to process nodes concurrently
#     with ThreadPoolExecutor(max_workers=2) as executor:
#         # Submit all nodes to the executor
#         futures = [executor.submit(process_node, node) for node in nodes]
#
#         # Use tqdm to monitor progress
#         with tqdm(total=len(futures), desc="Processing nodes") as pbar:
#             for future in concurrent.futures.as_completed(futures):
#                 try:
#                     result = future.result()
#                     elements.append(result)
#                 except Exception as e:
#                     print(f"An error occurred: {e}")
#                 pbar.update(1)
#
#     return elements
#
#
#
# # Define a function to handle a single completion
# def get_completion(node, llm):
#     SYSTEM_PROMPT = """You are an expert system specialized in extracting entities and relationships from scientific literature in the domain of electrical engineering and 5G communication systems, with a specific focus on literature reviews. Your goal is to analyze paragraphs from 5G literature reviews and extract structured information to contribute to building a comprehensive domain-specific knowledge graph.
#
#     Given a paragraph from a scientific paper's literature review about 5G technology:
#
#     Task:
#     Analyze the provided paragraph and extract all explicit binary relationships between relevant entities within the domain.
#
#     Output Format:
#     Present the extracted relationships under the heading "RELATIONSHIPS:", with each relationship on a new line in the following format:
#     Entity A -> Entity B (relationship type)
#
#     Rules for Extraction and Formatting:
#     1.  **Entity Identification:** Identify all relevant entities such as technologies, concepts, applications, standards, hardware components, stakeholders, etc., that are mentioned in the paragraph and pertain to the 5G domain. (Note: You identify these internally to find relationships; they are not listed separately in the output).
#     2.  **Relationship Identification:** Identify direct relationships that are explicitly stated in the text linking two identified entities. Focus only on relationships entirely contained within the paragraph.
#     3.  **Binary Relationships:** Extract only relationships involving exactly two entities.
#     4.  **Explicitness:** Do NOT infer relationships that are not directly stated in the text.
#     5.  **Exhaustiveness:** Aim to capture as many explicit binary relationships as possible from the provided paragraph.
#     6.  **Consistency:** Maintain consistent naming for the same entity across different relationships extracted *from this specific paragraph*.
#     7.  **Naming Convention:**
#         * All entity names and relationship labels must be in English.
#         * Entity names (nouns) should be in singular form and lemmatized (e.g., "technology" not "technologies", "application" not "applications", "network" not "networks").
#         * Relationship labels (verbs) should be in the infinitive form and lemmatized where appropriate (e.g., "use" not "uses" or "using", "enable" not "enables" or "enabled", "provide" not "provides").
#         * Use clear, concise, and descriptive labels for relationship types (e.g., "enable", "include", "use", "provide", "require", "is a type of", "improve").
#     8.  **Output Content:** Provide ONLY the "RELATIONSHIPS:" section followed immediately by the list of relationships. Do NOT include a separate list of entities, introductory/concluding remarks, or any other explanatory text.
#     9.  **Formatting:** Adhere strictly to the specified `Entity A -> Entity B (relationship type)` format for each relationship line.
#
#     """
#
#     # query_wrapper_prompt = PromptTemplate(
#     #     "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
#     # )
#
#     # llm.query_wrapper_prompt = query_wrapper_prompt
#     # llm.query_wrapper_prompt = "[INST]<<SYS>>\n" + SYSTEM_PROMPT + "<</SYS>>\n\n{query_str}[/INST] "
#     return llm.complete(f""""<<system prompt>>\n + {SYSTEM_PROMPT} + <</system prompt>>\n\n <<user query>>Extract entities and relationships from the following text: {node.text} <</user query>>""")
#
# =============================================================================
