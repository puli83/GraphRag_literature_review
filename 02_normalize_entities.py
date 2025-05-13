import pickle
import os
import datetime
from llama_index.core import Document
from Modules.EntityNormalizer import EntityNormalizer
from Modules.Parsers import ParserLLM_KG_extracted_output
# %% PATH
pth_nodes = 'data_nodes'

models_cache_folder = 'embedding_sentence_transformer_models'

# %% import


with open(os.path.join(pth_nodes, f'nodes_lit_rev_with_KG_2025-05-12.pkl'), 'rb') as f:
    nodes = pickle.load(f)


# %% prepare entities

nodes[0].metadata.keys()
nodes[0].metadata['deepseek_llama_370_answer'].text

# %% initialize tools
parser = ParserLLM_KG_extracted_output()

normalizer = EntityNormalizer(
    spacy_model_name='en_core_web_lg',
    embedding_model_name='all-MiniLM-L6-v2',
    cache_folder=models_cache_folder, # Set to None to use default cache
    linkage_method='average',# ward, centroid, average
    linkage_metric='cosine'
)


# %% parse llm output
entities = set()
for node in nodes:
    try:
        _, data_json = parser.parse_text(node.metadata['deepseek_llama_370_answer'].text)
    except:
        continue
    structured_rels = parser.extract_structured_relationships(data_json)
    node.metadata['structured_rels'] = structured_rels

    for item in structured_rels:
        entities.add(item["source"])
        entities.add(item["target"])

# Convert set to a list (optional: sorted for readability)
unique_entities = sorted(list(entities))



# %% Perform normalization using the normalizer instance
distance_threshold = 0.20
normalization_map = normalizer.cluster_and_normalize_entities(unique_entities, distance_threshold=distance_threshold)

print(f"Compression rate for distance_threshold = {distance_threshold}\t ComRate = {len(set(normalization_map.values())) / len(set(normalization_map.keys()))} ")


# %% update nodes


for node in nodes:
    struct_rel_normalized = []
    if 'structured_rels' in node.metadata.keys():
        for item in node.metadata['structured_rels']:
            struct_rel_normalized.append({'source' : normalization_map.get(item["source"]), 'target' : normalization_map.get(item["target"]), 'relation' : item["relation"]})

        node.metadata['structured_rels_normalized_embed'] = struct_rel_normalized

# %%
# Save nodes

with open(os.path.join(pth_nodes, f'nodes_lit_rev_with_KG_normalzed_embed{str(datetime.datetime.now().strftime("%Y-%m-%d"))}.pkl'), 'wb') as f:
    pickle.dump(nodes, f)
