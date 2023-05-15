import json
from load_data import kg_full, r2i, e2i
from demo_bilinear import Scorer
import json

def exctract_triples(r_path, w_path):
    with open(r_path, "r") as f:
        with open(w_path, "w") as f2:
            for row in f:
                data = json.loads(row)
                generate = data["generate"]
                source = data["source"]
                target = data["target"]
                # path = data["path"]
                one_path_triple_list = []
                one_data_triple_list = []
                for path in generate:
                    one_path_triple_list = []
                    rel_pos = []
                    # 找到关系词所在位置
                    path_word_list = str(path).split(' ')
                    for i in range(1, len(path_word_list)):
                        if path_word_list[i] in r2i:
                            rel_pos.append(i)
                    # 找到关系词的前后词
                    for i in range(len(rel_pos)):
                        rel = path_word_list[rel_pos[i]]
                        if i == 0:
                            head = path_word_list[0:rel_pos[i]]
                        else:
                            head = tail
                        pre_rel_pos = rel_pos[i]
                        if i == len(rel_pos)-1:
                            tail = path_word_list[rel_pos[i]+1:]
                        else:
                            latter_rel_pos  = rel_pos[i+1]
                            tail = path_word_list[pre_rel_pos+1:latter_rel_pos]
                        con_head = '_'.join(head)
                        con_tail = '_'.join(tail)
                        one_path_triple_list.append((con_head, rel, con_tail))
                        # print("head:",con_head, "rel:", rel ,"tail:",con_tail)
                    one_data_triple_list.append(one_path_triple_list)
                f2.write(json.dumps({'source': source, 'target': target, 'generate': generate, 'triple':one_data_triple_list}) + '\n')

def evaluate_triples(r_path, w_path):
    print("evaluating......")
    scorer = Scorer()
    valid_entity = 0
    entity_count = 0
    novelty_count = 0
    triples_count = 0 # triples 总数
    sum_count = 0
    best_count = 0
    max_count = 0 
    connection_count = 0
    path_count = 0
    is_connection  = False
    with open(r_path, "r") as f:
        with open(w_path, "w") as f2:
            for row in f:
                data = json.loads(row)
                one_data_triple_list = data["triple"]
                target = data['target']
                sum_score_list = []
                max_score_list = []
                best_score_list = []
                for one_path_triple_list in one_data_triple_list:
                    sum_triple_sum_score = 0
                    sum_triple_best_score = 0
                    sum_triple_max_score = 0
                    path_count += 1
                    is_connection  = False
                    k = 0
                    for triple in one_path_triple_list:
                        k += 1
                        head = triple[0]
                        rel = triple[1]
                        tail = triple[2]
                        if k == len(one_path_triple_list) and target in tail:
                            is_connection = True
                        if head != '':
                            entity_count += 1
                        if  tail != '':
                            entity_count += 1
                        # head or tail in cpnet
                        if head in e2i:
                            valid_entity += 1
                        if tail in e2i:
                            valid_entity += 1
                        
                        # triples in cpnet
                        if kg_full.has_node(head) and kg_full.has_node(tail) and kg_full.has_edge(head, tail, rel):
                            novelty_count += 1
                        if kg_full.has_node(head) and kg_full.has_node(tail) and not kg_full.has_edge(head, tail, rel):
                            print(head,rel,tail)

                        # three types of score
                        sum_score_result = scorer.gen_score(head, tail, 'sum')
                        best_score_results = scorer.gen_score(head, tail, 'all')
                        max_score_result = scorer.gen_score(head, tail, 'max')
                        for rel_tuple in best_score_results:
                            if rel == rel_tuple[0]:
                                best_score_results = rel_tuple[1]
                                break
                            else:
                                best_score_results = 0
                        if sum_score_result >= 3:
                            sum_count += 1
                        if best_score_results >= 0.5:
                            best_count += 1
                        if max_score_result[0][1] >= 0.5:
                            max_count += 1
                        sum_triple_sum_score += sum_score_result
                        sum_triple_best_score += best_score_results
                        sum_triple_max_score += max_score_result[0][1]
                    
                    triples_count += len(one_path_triple_list)

                    if is_connection == True:
                        connection_count += 1
                    
                    sum_score_list.append('{:.2f}'.format(sum_triple_sum_score / len(one_path_triple_list)))
                    max_score_list.append('{:.2f}'.format(sum_triple_max_score / len(one_path_triple_list)))
                    best_score_list.append('{:.2f}'.format(sum_triple_best_score / len(one_path_triple_list)))
                f2.write(json.dumps({'source': data['source'], 'target': target, 
                                    'sum_score' :sum_score_list,'max_score' :max_score_list,'best_score' :best_score_list, 
                                    'generate': data['generate'], 'triple':one_data_triple_list, }) + '\n')
        result_valid_entity = '{:.2f}'.format(valid_entity/entity_count *100)
        novelty = '{:.2f}'.format(novelty_count/triples_count *100)
        sum_score = '{:.2f}'.format(sum_count/triples_count *100)       
        best_score = '{:.2f}'.format(best_count/triples_count *100)
        max_score = '{:.2f}'.format(max_count/triples_count *100)
        connection = '{:.2f}'.format(connection_count/path_count *100)
        print(
                'connection',connection,
                'result_valid_entity', result_valid_entity, 
                'novelty', novelty, 
                'sum_score', sum_score, 
                'best_score', best_score, 
                'max_score', max_score,
                'valid_entity', valid_entity, 
                'entity_count', entity_count,
                'novelty_count',novelty_count, 
                'triples_count', triples_count
        )

if __name__ == "__main__":
    # path_data = "results_paths/predict_local_result_topk_beam_version46.json"
    # triples_data = "results_triples/predict_local_triples_topk_beam_version46.json"
    # data_with_score = "results_triples/predict_local_scores_topk_beam_version46.json"
    # exctract_triples(path_data, triples_data)
    # evaluate_triples(triples_data, data_with_score)

    path_data = "1_path_generation/results_paths/predict_dd_test.json"
    triples_data = "1_path_generation/results_triples/predict_dd_test.json"
    exctract_triples(path_data, triples_data)

# cd 1_path_generation python extract_triples_and_evaluate.py
# nohup python -u extract_triples_from_path.py > extract_triples_from_path.log 2>&1 &\
# beam_search 
# result_valid_entity 94.65 novelty 57.04 valid_entity 8410 entity_count 8885 novelty_count 2559 triples_count 4486
# result_valid_entity 94.65 novelty 57.04 sum_score 33.42 best_score 20.95 max_score 53.23 valid_entity 8410 entity_count 8885 novelty_count 2559 triples_count 4486
# connection 83.22 result_valid_entity 94.65 novelty 57.04 sum_score 33.42 best_score 20.95 max_score 53.23 valid_entity 8410 entity_count 8885 novelty_count 2559 triples_count 4486
# local connection 88.15 result_valid_entity 94.52 novelty 28.69 sum_score 40.74 best_score 21.97 max_score 61.89 valid_entity 7300 entity_count 7723 novelty_count 1114 triples_count 3883

# connection 99.33 result_valid_entity 99.69 novelty 54.67 sum_score 33.13 best_score 22.21 max_score 57.38 valid_entity 3888 entity_count 3900 novelty_count 1066 triples_count 1950



# nohup python -u extract_triples_and_evaluate.py > logs/extract_local2.log 2>&1