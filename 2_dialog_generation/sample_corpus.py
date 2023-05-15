
from itertools import groupby
from load_data import kg_full, kg_simple, relation2text
import json
import networkx as nx
# concept_list 提取出的所有对话的concept
def load_concepts_grounded_data(concepts_grounded_data_path):
    concepts_list = []
    with open(concepts_grounded_data_path, 'r') as f:
        for row in f:
            data = json.loads(row)
            concepts_list.append(data["ac"])
    return concepts_list

# 返回最大连续长度
def find_max_continuity(list):
    continuity_list = []
    fun = lambda x: x[1]-x[0]
    for _, g in groupby(enumerate(list), fun):
        l1 = [j for _, j in g]    # 连续数字的列表
    continuity_list.append(len(l1))
    max_continuity = max(continuity_list)
    return max_continuity

def construct_path_in_tgconv(concepts_list, w_path, r_dataset_path):
    with open(w_path, 'w') as g:
        with open(r_dataset_path, 'r') as f:    
            idx = 0
            for row in f:
                data = json.loads(row)
                # dialog_list = data['dialog']
                dialog_list = data['session']
                turn_count = len(dialog_list)
                # 当前这组对话所有语句的concept_list
                group_concepts  = concepts_list[idx:idx+turn_count]
                idx += turn_count

                G = nx.MultiDiGraph()
                for turn in range(turn_count - 1):
                    curr_nodes_list = group_concepts[turn]
                    next_nodes_list = group_concepts[turn+1]
                    for curr_node in curr_nodes_list:
                        edges = [n for n in kg_simple[curr_node]]
                        for next_node in next_nodes_list:
                            if next_node in edges:
                                rel_list = list(set([item for item in kg_full[curr_node][next_node]]))
                                weight_list = [kg_full[curr_node][next_node].get(rel)['weight'] for rel in rel_list]
                                for rel,weight in zip(rel_list, weight_list):
                                    G.add_edge(curr_node, next_node, key=rel, weight=weight, turn=turn)
                # key为轮次，对于
                dialog_dict = {}
                for start_nodes in G.nodes:
                    for end_nodes in G.nodes:
                        if start_nodes != end_nodes:
                            if nx.has_path(G, start_nodes, end_nodes):
                                paths = nx.all_simple_edge_paths(G, source=start_nodes, target=end_nodes)
                                d = {} # 对同一个起点和终点，保留weight最大的路径
                                for path in paths:
                                    # print(path)
                                    weight_list = [] # 每一个path更新一次，该路径的置信度是所有triple的weight之和
                                    turn_list = []
                                    simple_path = []
                                    simple_path_rel = []
                                    for triple in path:
                                        head = triple[0]
                                        tail = triple[1]
                                        rel = triple[2]
                                        weight = G[head][tail][rel]['weight']
                                        turn_a = G[head][tail][rel]['turn']
                                        weight_list.append(weight)
                                        turn_list.append(turn_a)
                                        if head not in simple_path:
                                            simple_path.append(head)
                                        if tail not in simple_path:
                                            simple_path.append(tail)
                                        simple_path_rel.append(rel)
                                    simple_path_key = ' '.join(simple_path)
                                    if simple_path_key in d.keys():
                                        if d[simple_path_key]['weight'] < sum(weight_list):
                                            d[simple_path_key] = {'weight': sum(weight_list), 'turn': turn_list, 'rel': simple_path_rel}
                                    else:
                                        d[simple_path_key] = {'weight':sum(weight_list), 'turn':turn_list, 'rel':simple_path_rel}
                                    turn_list_key = ' '.join([str(i) for i in turn_list])
                                    temp_list_key = turn_list_key.split(' ')
                                    temp_list_key = [int(item) for item in temp_list_key]
                                    # 判断temp_list_key是否是连续的
                                    max_continuity = find_max_continuity(temp_list_key)
                                    if max_continuity == len(temp_list_key):
                                        if len(temp_list_key) == len(list(set(temp_list_key))):# 有无重复的turn 2 2 3
                                            if turn_list_key in dialog_dict.keys():
                                                if dialog_dict[turn_list_key]['weight'] < sum(weight_list): # sum(weight_list)是当前路径的置信度
                                                    dialog_dict[turn_list_key] = {'path':simple_path, 'rel':simple_path_rel, 'weight':sum(weight_list)}
                                            else:
                                                dialog_dict[turn_list_key] = {'path':simple_path, 'rel':simple_path_rel, 'weight':sum(weight_list)}

                # print(dialog_dict)
                if len(dialog_dict) > 0:
                    for dialog_key in dialog_dict:
                        dialog_key_list = list(dialog_key.split(' '))
                        context = [dialog_list[int(i)] for i in dialog_key_list]
                        output = dialog_list[int(dialog_key_list[-1])+1]
                        
                        path = dialog_dict[dialog_key]['path']
                        rel = dialog_dict[dialog_key]['rel']

                        # 用rel把path连接
                        path_with_rel = []
                        for i in range(len(path)-1):
                            path_with_rel.append(path[i])
                            path_with_rel.append(relation2text[rel[i]])
                        path_with_rel.append(path[-1])
                        path_with_rel = ' '.join(path_with_rel)

                        g.write(json.dumps({'context':context, 
                                            'output':output,
                                            'full_path': path_with_rel,
                                            'path':dialog_dict[dialog_key]['path'],
                                            'rel':dialog_dict[dialog_key]['rel'],
                                            'weight':dialog_dict[dialog_key]['weight']
                                            }) + '\n')

if __name__ == '__main__':
    concepts_grounded_path = "2_dialog_generation/concepts_grounded_data/concepts_grounded_raw_test.json"
    w_path = "data/DailyDialog/test/test.json"
    r_dataset_path = "data/DailyDialog/dev/raw.json"
    concepts_list = load_concepts_grounded_data(concepts_grounded_path)
    construct_path_in_tgconv(concepts_list, w_path, r_dataset_path)
    
# nohup python -u 2_dialog_generation/construct_train_dev_data.py > 2_dialog_generation/logs/tgconv_to_path11.log 2>&1