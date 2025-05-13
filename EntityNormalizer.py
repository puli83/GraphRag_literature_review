# Ensure you have the necessary libraries installed:
# pip install spacy sentence-transformers scikit-learn scipy numpy
# python -m spacy download en_core_web_sm (or other models you use)

import spacy
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import re
import os # For cache folder path joining

class EntityNormalizer:
    """
    A class to perform advanced entity normalization using NLP preprocessing,
    vector embeddings, and hierarchical clustering.
    """

    def __init__(self,
                 spacy_model_name='en_core_web_sm',
                 embedding_model_name='all-MiniLM-L6-v2',
                 cache_folder=None,
                 linkage_method='average',
                 linkage_metric='cosine'):
        """
        Initializes the EntityNormalizer.

        Args:
            spacy_model_name (str): Name of the spaCy model to load (e.g., 'en_core_web_sm').
            embedding_model_name (str): Name of the SentenceTransformer model.
            cache_folder (str, optional): Path to the cache folder for SentenceTransformer models.
                                          Defaults to None (uses default Hugging Face cache).
            linkage_method (str): Method for hierarchical clustering (e.g., 'average', 'complete').
            linkage_metric (str): Metric for hierarchical clustering (e.g., 'cosine', 'euclidean').
        """
        self.nlp = self._load_spacy_model(spacy_model_name)
        self.embedding_model = self._load_embedding_model(embedding_model_name, cache_folder)
        self.linkage_method = linkage_method
        self.linkage_metric = linkage_metric
        print(f"EntityNormalizer initialized with spaCy model: '{spacy_model_name}', "
              f"Embedding model: '{embedding_model_name}', "
              f"Linkage: '{linkage_method}'/'{linkage_metric}'.")

    def _load_spacy_model(self, model_name):
        """Loads or downloads the specified spaCy model."""
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"Downloading spaCy '{model_name}' model...")
            spacy.cli.download(model_name)
            return spacy.load(model_name)

    def _load_embedding_model(self, model_name, cache_folder):
        """Loads the SentenceTransformer model, optionally using a specific cache folder."""
        if cache_folder:
            # Ensure cache_folder exists or SentenceTransformer might handle it
            # For clarity, one might add: os.makedirs(cache_folder, exist_ok=True)
            # However, SentenceTransformer typically uses ~/.cache/torch/sentence_transformers
            # The `cache_folder` argument in SentenceTransformer directly specifies where to download/load.
            print(f"Loading SentenceTransformer model '{model_name}' using cache folder: '{cache_folder}'")
            return SentenceTransformer(model_name, cache_folder=cache_folder)
        else:
            print(f"Loading SentenceTransformer model '{model_name}' using default cache.")
            return SentenceTransformer(model_name)

    def preprocess_entity_name(self, name):
        """
        Applies lowercasing, punctuation removal (basic), and lemmatization to an entity name using spaCy.
        """
        if not isinstance(name, str): # Handle non-string inputs gracefully
            return ""

        # processed_name_str = name.lower() # spaCy's lemmatizer can be case-sensitive
        # processed_name_str = re.sub(r'[^\w\s-]', '', name)
        # processed_name_str = re.sub(r'\s+', ' ', processed_name_str).strip()

        processed_name_str = name

        doc = self.nlp(processed_name_str)
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

        processed_name_final = " ".join(filter(None, lemmatized_tokens))

        # Return original cleaned name (after regex) if lemmatization results in empty
        return processed_name_final if processed_name_final else processed_name_str

    def get_embeddings(self, entity_names_list):
        """
        Generates embeddings for a list of preprocessed entity names.
        """
        if not entity_names_list:
            return np.array([])

        valid_names_list = [name for name in entity_names_list if isinstance(name, str) and name.strip()]
        if not valid_names_list:
            return np.array([])

        embeddings = self.embedding_model.encode(valid_names_list, convert_to_tensor=False)
        return embeddings

    def cluster_and_normalize_entities(self, original_entity_names, distance_threshold=0.2):
        """
        Performs NLP preprocessing, embedding, clustering, and creates a normalization map.

        Args:
            original_entity_names (list): A list of unique raw entity names.
            distance_threshold (float): Cosine distance threshold for merging clusters.

        Returns:
            dict: A mapping from original entity names to their canonical (normalized) forms.
        """
        if not original_entity_names:
            return {}

        original_to_preprocessed = {}
        for name_val in original_entity_names:
            preprocessed = self.preprocess_entity_name(name_val)
            original_to_preprocessed[name_val] = preprocessed if preprocessed else name_val.lower().strip()

        unique_preprocessed_names = sorted(list(set(val for val in original_to_preprocessed.values() if val)))

        if not unique_preprocessed_names:
            return {orig: orig.lower().strip() for orig in original_entity_names}

        embeddings = self.get_embeddings(unique_preprocessed_names)

        if embeddings.shape[0] < 2:
            final_map = {}
            for original, preprocessed_val in original_to_preprocessed.items():
                final_map[original] = preprocessed_val if preprocessed_val else original.lower().strip()
            return final_map

        linkage_matrix = linkage(embeddings, method=self.linkage_method, metric=self.linkage_metric)
        cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')

        cluster_to_canonical = {}
        cluster_members = {}

        for i, label in enumerate(cluster_labels):
            preprocessed_name = unique_preprocessed_names[i]
            if label not in cluster_members:
                cluster_members[label] = []
            cluster_members[label].append(preprocessed_name)

        for label, members in cluster_members.items():
            members = [m for m in members if m]
            if not members: continue
            canonical_name = sorted(members, key=lambda x: (len(x), x))[0]
            cluster_to_canonical[label] = canonical_name

        preprocessed_to_canonical = {}
        for i, preprocessed_name in enumerate(unique_preprocessed_names):
            if not preprocessed_name: continue
            cluster_label = cluster_labels[i] # Get the cluster label for the i-th unique preprocessed name
            if cluster_label in cluster_to_canonical:
                 preprocessed_to_canonical[preprocessed_name] = cluster_to_canonical[cluster_label]
            else:
                preprocessed_to_canonical[preprocessed_name] = preprocessed_name

        final_normalization_map = {}
        for original_name, preprocessed_name_val in original_to_preprocessed.items():
            if preprocessed_name_val in preprocessed_to_canonical:
                 final_normalization_map[original_name] = preprocessed_to_canonical[preprocessed_name_val]
            else:
                 final_normalization_map[original_name] = preprocessed_name_val if preprocessed_name_val else original_name.lower().strip()

        return final_normalization_map

# --- Example Usage ---
if __name__ == "__main__":
    # --- Instantiate the normalizer ---
    # Example with a custom cache folder (optional)
    # script_dir = os.path.dirname(__file__)
    # models_cache_folder = os.path.join(script_dir, 'sentence_transformer_cache')
    models_cache_folder = 'embedding_sentence_transformer_models'
    normalizer = EntityNormalizer(
        spacy_model_name='en_core_web_lg',
        embedding_model_name='all-MiniLM-L6-v2',
        cache_folder=models_cache_folder, # Set to None to use default cache
        linkage_method='average',# ward, centroid, average
        linkage_metric='cosine'
    )

    # normalizer = EntityNormalizer(linkage_method='complete', linkage_metric='cosine')


    sample_entities = [
        "5G Technologies",
        "5g technology",
        "Fifth Generation",
        "Wireless Communication",
        "Wireless Comm.",
        "Mobile Networks",
        "Mobile Network System",
        "Data Speed",
        "Speed of Data",
        "Latency",
        "Low Latency",
        "Network Coverage",
        "Coverage Area",
        "Apple Inc.",
        "apple incorporated",
        "Running shoes",
        "Shoes for running",
        "AI Models",
        "Artificial Intelligence Model"
    ]

    print("\nOriginal Entities:")
    for entity in sample_entities:
        print(f"- {entity}")

    # Perform normalization using the normalizer instance
    normalization_map = normalizer.cluster_and_normalize_entities(sample_entities, distance_threshold=0.40)

    print("\nNormalization Map (Original -> Canonical):")
    if normalization_map:
        for original, canonical in normalization_map.items():
            print(f"- '{original}'  =>  '{canonical}'")
    else:
        print("Normalization map is empty.")

    print("\n--- Example of how to use this map in graph building ---")
    raw_entity_from_chunk = "5G Technologies"
    # Use the normalizer's preprocess method for fallback if needed
    canonical_node_name = normalization_map.get(raw_entity_from_chunk,
                                                normalizer.preprocess_entity_name(raw_entity_from_chunk))
    print(f"Raw: '{raw_entity_from_chunk}', Canonical for graph: '{canonical_node_name}'")

    raw_entity_from_chunk_2 = "Shoes for run"
    canonical_node_name_2 = normalization_map.get(raw_entity_from_chunk_2,
                                                  normalizer.preprocess_entity_name(raw_entity_from_chunk_2))
    print(f"Raw: '{raw_entity_from_chunk_2}', Canonical for graph: '{canonical_node_name_2}'")

    print("\n--- Test preprocessing directly ---")
    test_name = "AI Models and Applications!"
    preprocessed_test = normalizer.preprocess_entity_name(test_name)
    print(f"Original: '{test_name}', Preprocessed: '{preprocessed_test}'")

    test_name_2 = "  Multiple   Spaces  "
    preprocessed_test_2 = normalizer.preprocess_entity_name(test_name_2)
    print(f"Original: '{test_name_2}', Preprocessed: '{preprocessed_test_2}'")

