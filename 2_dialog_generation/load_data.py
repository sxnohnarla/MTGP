import os
import pickle
import networkx as nx
import json

def load_kg(data_dir):
    print("loading cpnet....")
    data_path = os.path.join(data_dir, 'cpnet_laq.graph')
    kg_full = nx.read_gpickle(data_path)

    kg_simple = nx.DiGraph()
    for u, v, data in kg_full.edges(data=True):
        kg_simple.add_edge(u, v)

    return kg_full, kg_simple

def load_vocab(data_dir):
    rel_path = os.path.join(data_dir, 'rel_vocab.pkl')
    ent_path = os.path.join(data_dir, 'entity_vocab.pkl')

    with open(rel_path, 'rb') as handle:
        rel_vocab = pickle.load(handle)

    with open(ent_path, 'rb') as handle:
        ent_vocab = pickle.load(handle)

    return rel_vocab['i2r'], rel_vocab['r2i'], ent_vocab['i2e'], ent_vocab['e2i']

def load_relation2text():
    with open("utils/newrelation2text.json") as f:
        relation2text = json.load(f)
        return relation2text

data_dir = "data/my_conceptnet/"
i2r, r2i, i2e, e2i = load_vocab(data_dir)
kg_full, kg_simple = load_kg(data_dir)
relation2text = load_relation2text()
