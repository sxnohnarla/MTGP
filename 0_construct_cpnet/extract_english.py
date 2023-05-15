import json
import networkx as nx
from tqdm import tqdm
import nltk
import math
import os
import pickle

filtered_relations = [
    'atlocation', 
    'capableof',
    'causes',
    'causesdesire',
    'createdby',
    'definedas',
    'desires',
    'distinctfrom',
    'hasa',
    'hasfirstsubevent',
    'haslastsubevent',
    'hasprerequisite',
    'hasproperty',
    'hassubevent',
    'isa',
    'locatednear',
    'madeof',
    'mannerof',
    'motivatedbygoal', 
    'partof',
    'receivesaction',
    'similarto',
    'symbolof',
    'usedfor'
]

symmetric_relations = [
    'distinctfrom',
    'locatednear',
    'similarto'
]

blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "may", "wanter"]

def not_save(cpt):
    if cpt in blacklist:
        return True

    if cpt in nltk_stopwords:
        return True

    return False

def del_pos(s):
    """
        删除语音编码
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s

def extract_english_and_save(r_raw_concept_path, w_all_triples_path, w_entity_vocab_path, w_rel_vocab_path):
    """
    读取原始的conceptnet csv文件,并将所有的英文关系(头和尾都是英文实体)提取到一个新的文件中。
    一个新文件，每行的格式如下。<关系> <头> <尾> <权重>。
    """
    only_english = []
    rel_list = set()
    with open(r_raw_concept_path, encoding="utf8") as f:
        for line in f:
            ls = line.rstrip('\n').split('\t')
            if ls[2].startswith('/c/en/') and ls[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = ls[1].split("/")[-1].lower()
                head = del_pos(ls[2]).split("/")[-1].lower()
                tail = del_pos(ls[3]).split("/")[-1].lower()

                if rel not in filtered_relations:
                    continue

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue

                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue
                
                if rel not in filtered_relations:
                    continue

                data = json.loads(ls[4])

                only_english.append("\t".join([rel, head, tail, str(data["weight"])]))
   
    with open(w_all_triples_path, "w", encoding="utf8") as f:
        f.write("\n".join(only_english))

    graph = nx.MultiDiGraph()
    # graph = nx.Graph()

    for line in tqdm(only_english, desc="saving to graph"):
        ls = line.split('\t')
        rel = ls[0]
        subj = ls[1]
        obj = ls[2]
        weight = float(ls[3])
        if not_save(ls[1]) or not_save(ls[2]):
            continue
        if subj == obj: # delete loops
            continue
        weight = 1+float(math.exp(1-weight))
        graph.add_edge(subj, obj, key=rel, weight=weight)
        if rel not in symmetric_relations:
            opposite_rel = "_" + rel
            if opposite_rel not in filtered_relations:
                filtered_relations.append(opposite_rel)
        else:
            opposite_rel = rel
        graph.add_edge(obj, subj, key=opposite_rel, weight=weight)

    e2i = { w:i for i, w in enumerate(graph.nodes) }
    i2e = [w for w in graph.nodes ]

    r2i = { w:i for i, w in enumerate(filtered_relations) }
    i2r = [w for w in filtered_relations]

    # 存入pkl文件   
    with open(w_entity_vocab_path, 'wb') as f:
        pickle.dump({'e2i': e2i,'i2e': i2e,}, f)

    with open(w_rel_vocab_path, 'wb') as f:
        pickle.dump({'r2i': r2i,'i2r': i2r,}, f)

    nx.write_gpickle(graph, w_cpnet_graph_path)
    print(len(graph.nodes), len(graph.edges))

if __name__ == "__main__":
    r_raw_concept_path = "../raw_files/conceptnet-assertions-5.7.0.csv"
    w_all_triples_path = "../data/my_conceptnet/conceptnet_en.txt"
    w_entity_vocab_path = "../data/my_conceptnet/entity_vocab.pkl"
    w_rel_vocab_path = "../data/my_conceptnet/rel_vocab.pkl"
    w_cpnet_graph_path = "../data/my_conceptnet/cpnet_laq.graph"
    extract_english_and_save(r_raw_concept_path, w_all_triples_path, w_entity_vocab_path, w_rel_vocab_path)
