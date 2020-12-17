# Instructions
#######################
# Please, install node2vec package before running this script. 
# You can install it using following command:
# pip install node2vec or pip3 install node2vec
# Then use following command to run this script:
# python embeddings-generator.py or python3 embeddings-generator.py
#######################

import json
import numpy as np
import networkx as nx
import requests
from node2vec import Node2Vec

# Downloading and writing data to file

url = 'https://snap.stanford.edu/data/soc-sign-Slashdot090221.txt.gz'
r = requests.get(url, allow_redirects=True)
data_file = open('soc-sign-Slashdot090221.txt.gz', 'wb').write(r.content)

# Reading signed network data into directed graph using networkx
signed_network = nx.read_weighted_edgelist('soc-sign-Slashdot-wu.txt', comments='#', create_using=nx.DiGraph(), nodetype = int)

# Updating weights (-1 to 1 and 1 to 2) as node2vec does not work on negative weights
for from_node,to_node, weight in signed_network.edges(data=True):
    if weight['weight'] == -1:
        weight['weight'] = 1
    elif weight['weight'] == 1:
        weight['weight'] = 2
    
EMBEDDING_FILENAME = "embedded-soc-sign-slashdot"

EMBEDDING_FILENAME = "embedded-soc-sign-slashdot"

# Precompute probabilities and generate walks
node2vec = Node2Vec(signed_network, dimensions=100)

# Embed nodes
model = node2vec.fit()# Save embeddings for later use
model.wv.save_word2vec_format(EMBEDDING_FILENAME)

# Save model for later use
EMBEDDING_MODEL_FILENAME = "embedded-soc-sign-slashdot-model"
model.save(EMBEDDING_MODEL_FILENAME)

# Embed edges using Hadamard method
from node2vec.edges import HadamardEmbedder
edges_embs = HadamardEmbedder(keyed_vectors=model.wv)

# Save edge embeddings for later use
EDGES_EMBEDDING_FILENAME = "edges-embedded-soc-sign-slashdot"
# Get all edges in a separate KeyedVectors instance - use with caution could be huge for big networks
edges_kv = edges_embs.as_keyed_vectors()
edges_kv.save_word2vec_format(EDGES_EMBEDDING_FILENAME)