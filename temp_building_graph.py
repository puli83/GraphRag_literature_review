import pickle
import os
import random
import re

# %% path
pth_nodes = 'data_nodes'
os.listdir(pth_nodes)
# %% function

def extract_chain_of_thoughts(input_string):
    # Extract the <think> part
    start_tag = "<think>"
    end_tag = "</think>"

    # Find the start and end positions
    start_index = input_string.find(start_tag) + len(start_tag)
    end_index = input_string.find(end_tag)

    # Extract the content between <think> and </think>
    think_part = input_string[start_index:end_index].strip()
    return(think_part)

def add_newline_before_tag(text):
    return re.sub(r'(.*?)(<<</SYS>>|\[INST\]|<<SYS>>|\[/INST\])', r'\1\n\2', text, count=1)
def clean_response_llm(text):
    # text = re.sub(r'\[\/?INST\]', '', text)
    # text = re.sub(r'<<SYS>>', '', text)
    think_part = extract_chain_of_thoughts(text)
    result = ''
    try:
        result = re.search(r'(RELATIONSHIPS|RELATIONSHMENTS|RELATIONSHIONS|RELATIONSHONES|RELATIONSHINES|RELATIONSHATIONS):.*(<<</SYS>>|\[INST\]|<<SYS>>|\[/INST\])', text, re.DOTALL)
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return

    if result:
            # If tags were found, use that result
            result = result.group()
    else:
        try:
            # If no tags found, get all text after RELATIONSHIPS...
            result = re.search(r'(RELATIONSHIPS|RELATIONSHMENTS|RELATIONSHIONS|RELATIONSHONES|RELATIONSHINES|RELATIONSHATIONS):.*', text, re.DOTALL).group()
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return
    # try:
    #     result = re.search(r'(RELATIONSHIPS|RELATIONSHMENTS|RELATIONSHIONS|RELATIONSHONES|RELATIONSHINES|RELATIONSHATIONS):.*\[.*?[0-3]\]', text, re.DOTALL).group()
    # except Exception as e:
    #     print(f"Error processing chunk: {e}")
    result = add_newline_before_tag(result)
    return think_part, result



def parse_text(text):
    thinking_text, text = clean_response_llm(text)
    # Initialize dictionary - we only need Relationships now
    result = {"Relationships": []}

    # Split to get relationships section
    try:
        # Split after "RELATIONSHIPS:" and take everything that follows
        relationships_section = text.split("RELATIONSHIPS:")[-1]
        # Split by newlines and clean up each relationship
        relationships = [r.strip() for r in relationships_section.strip().split('\n') if r.strip()]
        result["Relationships"] = list(set(relationships))
        result["Relationships"] = [x for x in result["Relationships"] if re.search('->|→', x)]

    except Exception as e:
        print(f"Error processing chunk: {e}")
        result["Relationships"] = []

    return thinking_text, result





# %%import


# Load nodes from pickle file
with open(os.path.join(pth_nodes,'nodes_lit_rev_with_KG_2025-05-12.pkl'), 'rb') as f:  # Note 'rb' for reading binary
    nodes = pickle.load(f)

nodes[0].metadata.keys()
nodes[0].metadata['deepseek_llama_370_answer'].text
dir(nodes[0].metadata['deepseek_llama_370_answer'])
nodes[0].metadata['deepseek_llama_370_answer'].text
counter = 0
for node in nodes:
    if 'deepseek_llama_370_answer' in node.metadata.keys() :
        counter +=1




# test = random.sample(nodes[0:100], 10)



# for idx,x in enumerate(test):
#     print(f'{idx}, {idx},{idx}, {idx},{idx}, {idx},{idx}, {idx}\\n\n')
#     print(x.metadata['entities_relationship'])
# persed_t = parse_text(test_sin_text)

# clean_response_llm
# %% parse


result_two = parse_text(nodes[0].metadata['deepseek_llama_370_answer'].text)


# node = nodes[200]
#############################################
import networkx as nx
import re
from tqdm import tqdm

def build_graph_structured_data_simple_mode(nodes, name_graph_metadata = 'deepseek_llama_370_answer'):
    G = nx.Graph()
    error_idx_list = []

    for idx, node in tqdm(enumerate(nodes), total=len(nodes), desc="Processing nodes"):
        # Get data - now only relationships
        _, data = parse_text(node.metadata.get(name_graph_metadata, '').text)
        chunk_id_temp = node.metadata.get('chunk_id')
        if not data["Relationships"] :
            print(f'error at idx : {idx}')
            error_idx_list.append(idx)
            continue

        for line in data['Relationships']:
            # Initialize default values
            # weight = None
            relation = ""
            # print(line)

            # Extract weight using regex
            # weight_match = re.search(r'\[.*?([0-3])\]', line)
            # weight_match = re.search(r'\[(\d+)\]', line)
            # if weight_match:
            #     weight = int(weight_match.group(1))
            #     main_part = line.split('[')[0]
            # else:
            #     main_part = line+

            main_part = line

            # Extract relation
            relation_match = re.search(r'\((.*?)\)', line)
            if relation_match:
                relation = relation_match.group(1)
                main_part = re.split(r'\(.*?\)', main_part)[0]

            # Split on arrow (handling both -> and →)
            parts = re.split(r'->|→', main_part)
            if len(parts) >= 2:
                source = parts[0].strip()
                target = parts[-1].strip()

                # Add nodes implicitly from the relationship
                G.add_node(source, chunk_id = chunk_id_temp)
                G.add_node(target, chunk_id = chunk_id_temp)

                # Add the edge with attributes
                G.add_edge(source, target,
                          label=relation,
                          # weight=weight,
                          direction="forward")
    return G, error_idx_list

####################


def build_graph_structured_data_simple_mode(nodes):
    G = nx.Graph()  # Create a directed graph
    error_idx_list = []

    for idx, node in tqdm(enumerate(nodes), total=len(nodes), desc="Processing nodes"):
        # Get data - now only relationships
        data = parse_text(node.metadata.get('entities_relationship', '').text)
        author_temp = node.metadata.get('Authors')
        author_temp_dict = {}
        for aut in author_temp:
            author_temp_dict[aut] = 1

        if not data["Relationships"] :
            print(f'error at idx : {idx}')
            error_idx_list.append(idx)
            continue

        for line in data['Relationships']:
            # Initialize default values
            weight = 0
            relation = ""
            # print(line)

            # Extract weight using regex
            weight_match = re.search(r'\[.*?([0-3])\]', line)
            # weight_match = re.search(r'\[(\d+)\]', line)
            if weight_match:
                weight = int(weight_match.group(1))
                main_part = line.split('[')[0]
            else:
                main_part = line

            # Extract relation
            relation_match = re.search(r'\((.*?)\)', line)
            if relation_match:
                relation = relation_match.group(1)
                main_part = re.split(r'\(.*?\)', main_part)[0]

            # Split on arrow (handling both -> and →)
            parts = re.split(r'->|→', main_part)
            if len(parts) >= 2:

                source = parts[0].strip()
                target = parts[-1].strip()

                # Handle node attributes
                if source not in G:
                    G.add_node(source, authors=author_temp_dict)
                else:
                    # Append new author to existing authors list
                    # if 'authors' in G.nodes[source]:
                    current_authors = G.nodes[source].get('authors', {})
                    for aut in author_temp_dict.keys():
                        if aut in current_authors.keys():
                            # Update count for author (add 1 if exists, or set to 1 if new)
                            current_authors[aut] = current_authors.get(aut, 0) + 1
                        else:
                            current_authors[aut] = 1

                    # Update node attribute
                    G.nodes[source]['authors'] = current_authors


                # Do the same for target node
                if target not in G:
                    G.add_node(target, authors=author_temp_dict)
                else:
                    # Append new author to existing authors list
                    # if 'authors' in G.nodes[target]:
                    current_authors = G.nodes[target].get('authors', {})
                    for aut in author_temp_dict.keys():
                        if aut in current_authors.keys():
                            # Update count for author (add 1 if exists, or set to 1 if new)
                            current_authors[aut] = current_authors.get(aut, 0) + 1
                        else:
                            current_authors[aut] = 1

                    # Update node attribute
                    G.nodes[target]['authors'] = current_authors



                # # Add nodes implicitly from the relationship
                # G.add_node(source, authors=author_temp_dict)
                # G.add_node(target, authors=author_temp_dict)

                # Add the edge with attributes
                G.add_edge(source, target,
                          label=relation,
                          weight=weight,
                          direction="forward")
    return G, error_idx_list

# node = nodes[254]
# node.metadata
def build_graph_structured_data_simple_mode_update(nodes):
    G = nx.Graph()  # Create a directed graph
    error_idx_list = []

    for idx, node in tqdm(enumerate(nodes), total=len(nodes), desc="Processing nodes"):
        data = parse_text(node.metadata.get('entities_relationship', '').text)
        author_temp = node.metadata.get('Authors')
        author_temp_dict = {}
        for aut in author_temp:
            author_temp_dict[aut] = 1

        if not data["Relationships"]:
            print(f'error at idx : {idx}')
            continue
        # line = data['Relationships'][0]
        for line in data['Relationships']:
            # Initialize default values
            weight = 0
            relation = ""
            # print(line)

            # Extract weight using regex
            weight_match = re.search(r'\[.*?([0-3])\]', line)
            # weight_match = re.search(r'\[(\d+)\]', line)
            if weight_match:
                weight = int(weight_match.group(1))
                main_part = line.split('[')[0]
            else:
                main_part = line

            # Extract relation
            relation_match = re.search(r'\((.*?)\)', line)
            if relation_match:
                relation = relation_match.group(1)
                main_part = re.split(r'\(.*?\)', main_part)[0]

            # Split on arrow (handling both -> and →)
            parts = re.split(r'->|→', main_part)
            if len(parts) >= 2:
                source = parts[0].strip()
                target = parts[-1].strip()
                # Handle node attributes
                if source not in G:
                    G.add_node(source, authors=author_temp_dict)
                else:
                    # Append new author to existing authors list
                    # if 'authors' in G.nodes[source]:
                    current_authors = G.nodes[source].get('authors', {})
                    for aut in author_temp_dict.keys():
                        if aut in current_authors.keys():
                            # Update count for author (add 1 if exists, or set to 1 if new)
                            current_authors[aut] = current_authors.get(aut, 0) + 1
                        else:
                            current_authors[aut] = 1

                    # Update node attribute
                    G.nodes[source]['authors'] = current_authors


                # Do the same for target node
                if target not in G:
                    G.add_node(target, authors=author_temp_dict)
                else:
                    # Append new author to existing authors list
                    # if 'authors' in G.nodes[target]:
                    current_authors = G.nodes[target].get('authors', {})
                    for aut in author_temp_dict.keys():
                        if aut in current_authors.keys():
                            # Update count for author (add 1 if exists, or set to 1 if new)
                            current_authors[aut] = current_authors.get(aut, 0) + 1
                        else:
                            current_authors[aut] = 1

                    # Update node attribute
                    G.nodes[target]['authors'] = current_authors



                # Handle edge attributes
                if G.has_edge(source, target):
                    # Append new relationship info or update existing
                    edge_data = G.edges[source, target]
                    if isinstance(edge_data['label'], list):
                        if relation not in edge_data['label']:
                            edge_data['label'].append(relation)
                    else:
                        edge_data['label'] = [edge_data['label'], relation]

                    # Add weights
                    if weight:
                        current_weight = edge_data.get('weight', 0)  # Get current weight, default to 0 if none
                        edge_data['weight'] = current_weight + weight
                else:
                    # Add new edge
                    G.add_edge(source, target,
                              label=[relation],
                              weight=weight,
                              direction="forward")

    return G, error_idx_list


# Check if all edges have a 'weight' attribute
all_have_weight = all('weight' in data for _, _, data in G.edges(data=True))
all_have_authors = all('authors' in data for  _, data in G.nodes(data=True))
from collections import Counter

weights = [data.get('weight') for _, _, data in G.edges(data=True) if 'weight' in data]
weight_counts = Counter(weights)

for weight, count in weight_counts.items():
    print(f"Weight {weight}: {count} edges")

node_name= 'social distancing'
'innovation' in G

edges = list(G.edges(node_name))

list(G.in_edges(node_name))


# Gather outgoing edges
outgoing_edges = list(G.out_edges(node_name, data=True))

# Gather incoming edges
incoming_edges = list(G.in_edges(node_name, data=True))

# Combine them
all_edges = outgoing_edges + incoming_edges
print(all_edges)




G.nodes['innovation']
type(G.nodes)
G.nodes()[0]

from cdlib import algorithms, evaluation

def detect_communities(graph, algorithm='louvain'):
   """
   Detect communities using various algorithms

   Args:
       graph (networkx.Graph): Input graph
       algorithm (str): Algorithm to use - options: 'louvain', 'leiden', 'lp', 'walktrap'

   Returns:
       list: List of communities (each community is a set of nodes)
   """
   try:
       # Dictionary mapping algorithm names to functions
       algorithm_map = {
           'louvain': algorithms.louvain,
           'leiden': algorithms.leiden,
           'lp': algorithms.label_propagation,
           'walktrap': algorithms.walktrap,
           'hierarchical_link_community' : algorithms.hierarchical_link_community
       }

       if algorithm.lower() not in algorithm_map:
           print(f"Algorithm {algorithm} not found. Using Louvain as default.")
           algorithm = 'louvain'

       # Get and apply the selected algorithm
       community_func = algorithm_map[algorithm.lower()]
       communities = community_func(graph)

       print(f"\nUsing {algorithm} algorithm")
       print(f"Total communities detected: {len(communities.communities)}")

       # Basic statistics about communities
       community_sizes = [len(c) for c in communities.communities]
       if community_sizes:
           print(f"Average community size: {sum(community_sizes)/len(community_sizes):.2f}")
           print(f"Largest community size: {max(community_sizes)}")
           print(f"Smallest community size: {min(community_sizes)}")

       # Calculate modularity
       try:
           modularity = evaluation.newman_girvan_modularity(graph, communities)
           print(f"Modularity score: {modularity.score:.4f}")
       except Exception as e:
           print(f"Could not calculate modularity: {str(e)}")

       return communities

   except Exception as e:
       print(f"Error in community detection: {str(e)}")
       return []


# Create the graph with project interactions
G, error_idx = build_graph_structured_data_simple_mode(nodes)
G, error_idx = build_graph_structured_data_simple_mode_update(nodes)

# for x in error_idx:
#     print(x)
#     print(nodes[x].metadata.get('entities_relationship'))
# # list error: [134, 224, 272, 351, 432, 464, 466, 596, 674, 699, 846]
# # to redo : 134, 224, 272, 466, 674, 699
# # other corrected with regex in clean_response_llm()
# # after redo llm extract reklationship , still 3 nodes egives errors : 224, 272, 351
# # 224, 272 cannot be reolved


# By degree (total connections)
degrees = dict(G.degree())
most_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:30]
print("\nTop 10 nodes by total connections:")
for node, degree in most_connected:
    print(f"{node}: {degree} connections")

nodes_to_remove = ["Canada", "Project", "Researchers", "project", 'Research', 'Research project','researchers','CRD project','Research Project']
G.remove_nodes_from(nodes_to_remove)


# =============================================================================
#
# # Get degree of all nodes
# degrees = dict(G.degree())
#
# # Set a threshold (for example, remove nodes with less than 3 connections)
# threshold = 2
#
# # Find nodes to remove
# nodes_to_remove = [node for node, degree in degrees.items() if degree < threshold]
# len(nodes_to_remove)
# len(G.nodes)
# # Remove the nodes (this will also remove all their edges)
# G.remove_nodes_from(nodes_to_remove)
# =============================================================================


# By in-degree (incoming connections)
in_degrees = dict(G.in_degree())
most_incoming = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 nodes by incoming connections:")
for node, degree in most_incoming:
    print(f"{node}: {degree} incoming connections")

# By out-degree (outgoing connections)
out_degrees = dict(G.out_degree())
most_outgoing = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 nodes by outgoing connections:")
for node, degree in most_outgoing:
    print(f"{node}: {degree} outgoing connections")

# If you want to consider weights:
weighted_degrees = dict(G.degree(weight='weight'))
most_connected_weighted = sorted(weighted_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 nodes by weighted connections:")
for node, weight in most_connected_weighted:
    print(f"{node}: {weight} total weight")

import matplotlib.pyplot as plt
nx.draw(G, with_labels=True)
plt.show()

# For more customized visualization:
pos = nx.spring_layout(G)
nx.draw(G, pos,
        node_color='lightblue',
        node_size=100,
        with_labels=True,
        font_size=5)
plt.show()


len(G.degree)
len(G.edges)

# Detect communities
communities = detect_communities(G, algorithm='leiden')  # Leiden
# Usage examples:
# communities = detect_communities(G)  # Default louvain
# communities = detect_communities(G, algorithm='lp')  # Label propagation
# communities = detect_communities(G, algorithm='leiden')  # Leiden
# communities = detect_communities(G, algorithm='walktrap')  # Walktrap
# communities = detect_communities(G, algorithm='hierarchical_link_community')  # Walktrap



size_counts = Counter([len(comm) for comm in communities.communities])
# Get all entries with size < 3
small_communities = {size: count for size, count in size_counts.items() if size < 3}


# Basic statistics about communities
community_sizes = [len(c) for c in communities.communities]
if community_sizes:
    print(f"Average community size: {sum(community_sizes)/len(community_sizes):.2f}")
    print(f"Largest community size: {max(community_sizes)}")
    print(f"Smallest community size: {min(community_sizes)}")


def recursive_leiden(G, min_size=50):
    # Get initial communities
    communities = algorithms.leiden(G)
    initial_communities = communities.communities  # Your current communities
    final_communities = []
    ###
    community_stats = {
    'initial_communities': {
        'total_communities': len(communities.communities)
    }
}

    # Basic statistics about communities
    community_sizes = [len(c) for c in initial_communities]
    if community_sizes:
        community_stats['initial_communities'].update({
        'average_size': round(sum(community_sizes)/len(community_sizes), 2),
        'largest_size': max(community_sizes),
        'smallest_size': min(community_sizes)
    })

    # Calculate modularity
    try:
        modularity = evaluation.newman_girvan_modularity(G, communities)
        community_stats['initial_communities']['modularity_score'] = round(modularity.score, 4)
    except Exception as e:
        community_stats['initial_communities']['modularity_error'] = str(e)


    for community in initial_communities:
        # break
        if len(community) > min_size:
            # Create subgraph for this large community
            subgraph = G.subgraph(community)
            # Apply Leiden to this subgraph
            sub_communities = algorithms.leiden(subgraph)
            # Add these smaller communities to final list
            final_communities.extend(sub_communities.communities)
        else:
            # Keep small communities as they are
            final_communities.append(community)

    # Print some statistics
    print(f"Original number of communities: {len(initial_communities)}")
    print(f"Final number of communities: {len(final_communities)}")
    print("\nSize distribution of final communities:")
    sizes = [len(comm) for comm in final_communities]
    print(f"Largest community: {max(sizes)}")
    print(f"Smallest community: {min(sizes)}")
    print(f"Average size: {sum(sizes)/len(sizes):.2f}")

    return final_communities

new_communities = recursive_leiden(G,  min_size=50)

Counter([len(comm) for comm in new_communities])



# Save
def save_network_data(graph, my_list, filename):
    data = {
        'graph': nx.node_link_data(graph),  # Convert graph to serializable format
        'list': my_list
    }
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

save_network_data(G, new_communities, 'RAG_EXTRACT_TEST/Graph_communities-2024-12-10.pkl')



# # Load
# def load_network_data(filename):
#     with open(filename, 'rb') as file:
#         data = pickle.load(file)
#     return nx.node_link_graph(data['graph']), data['list']



from collections import Counter

def analyze_community_authors(G, communities):
    community_authors = {}

    # For each community
    for idx, community in enumerate(communities):
        # Collect all authors and their counts in this community
        author_counts = Counter()

        # For each node in community
        node  = community[0]
        for node in community:
            # Get authors dictionary from node
            node_authors = G.nodes[node].get('authors', {})
            # Update counts
            author_counts.update(node_authors)

        # Store results
        community_authors[idx] = author_counts

    # Print results
    for comm_idx, authors in community_authors.items():
        print(f"\nCommunity {comm_idx} (size: {len(communities[comm_idx])} nodes)")
        print("Top authors:")
        for author, count in authors.most_common(5):  # Show top 5 authors
            print(f"  {author}: {count} contributions")

    return community_authors

# Use the function
len(new_communities)
community_authors = analyze_community_authors(G, [x for x in new_communities if len(x)> 50])



# raw confing: 644 - 254  = 390
# remove onlsuy specific nodes : 1149 - (274+444) : 431
# remove also nodes one connection only  : 411 - (41+233): 137

# Find the largest community
largest_community = max(new_communities, key=len)
print(f"Size of largest community: {len(largest_community)}")

# Create a subgraph of just the largest community
largest_comm_subgraph = G.subgraph(largest_community)

# Now you can analyze this community in different ways:

# 1. See most connected nodes within this community
degrees = dict(largest_comm_subgraph.degree())
most_connected = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
print("\nMost connected nodes in largest community:")
for node, degree in most_connected:
    print(f"{node}: {degree} connections")

# 2. Look at edge weights
edge_weights = [(u, v, d['weight']) for u, v, d in largest_comm_subgraph.edges(data=True) if 'weight' in d]
heaviest_edges = sorted(edge_weights, key=lambda x: x[2], reverse=True)[:10]
print("\nStrongest relationships (highest weights):")
for source, target, weight in heaviest_edges:
    print(f"{source} -> {target}: weight {weight}")

# 3. Look at relationship types
edge_types = [(u, v, d['label']) for u, v, d in largest_comm_subgraph.edges(data=True) if 'label' in d]
print("\nTypes of relationships present:")
for source, target, label in edge_types[:10]:  # showing first 10 as example
    print(f"{source} -> {target}: {label}")

# 4. Basic statistics
print(f"\nCommunity statistics:")
print(f"Number of nodes: {len(largest_comm_subgraph.nodes())}")
print(f"Number of edges: {len(largest_comm_subgraph.edges())}")
print(f"Density: {nx.density(largest_comm_subgraph):.4f}")


weights = [data.get('weight') for _, _, data in G.edges(data=True) if 'weight' in data]
len(weights)
len(G.edges)

coms = algorithms.leiden(G, weights = weights)

    # Convert communities to list to analyze
comm_list = list(coms.communities)

# print(f"\nResults for resolution = {resolution}")
print(f"Total communities: {len(comm_list)}")
print(f"Largest community size: {max(len(c) for c in comm_list)}")
print(f"Smallest community size: {min(len(c) for c in comm_list)}")
print(f"Average community size: {sum(len(c) for c in comm_list)/len(comm_list):.2f}")


# Create subgraph with edges weight > 1
edges_above_1 = [(u, v) for u, v, data in G.edges(data=True)
                 if 'weight' in data and data['weight'] > 1]

# Create the subgraph
subgraph = G.edge_subgraph(edges_above_1)

communities = detect_communities(subgraph, algorithm='leiden')


len(communities)

evaluation.newman_girvan_modularity(G, communities)

sizes = [len(sublist) for sublist in communities.communities]
from collections import Counter
Counter(sizes)

# Dictionary to store author relevance for each community
from collections import defaultdict
community_author_relevance = defaultdict(lambda: defaultdict(float))

# Process each community
for community_id, community in enumerate(communities):
    for node in community:
        authors = G.nodes[node].get('authors', [])
        if authors:
            weight = 1 / len(authors)  # Divide relevance equally among authors
            for author in authors:
                community_author_relevance[community_id][author] += weight

# Display the results
for community_id, authors in community_author_relevance.items():
    print(f"Community {community_id}:")
    for author, relevance in authors.items():
        print(f"  Author: {author}, Relevance: {relevance:.2f}")

#########################################

# from community import community_louvain
# partition = community_louvain.best_partition(G)  # Dictionary {node: community}

# # Step 2: Aggregate author relevance for communities

# community_author_relevance = defaultdict(lambda: defaultdict(float))

# for node, community in partition.items():
#     authors = G.nodes[node].get("authors", [])
#     if authors:
#         weight = 1 / len(authors)  # Divide relevance equally among all authors
#         for author in authors:
#             community_author_relevance[community][author] += weight

# # Step 3: Display results
# for community, authors in community_author_relevance.items():
#     print(f"Community {community}:")
#     for author, relevance in authors.items():
#         print(f"  Author: {author}, Relevance: {relevance:.2f}")


# G.edges(data=True)
