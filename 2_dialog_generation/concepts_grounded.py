from grounding import ground_TG
import json
from multiprocessing import cpu_count
from load_data import i2e

def save(write_vocab_path, write_patterns_path):
    # 处理节点数据，存入cpnet/laq/concept.txt
    with open(write_vocab_path, 'w') as f:
        for item in i2e:
            f.write(item)
            f.write('\n')

    '''
        "ab_intra":[
            {
                "LEMMA": "ab",
            },
            {
                "LEMMA": "intra"
            }
        ],
    '''
    with open(write_patterns_path, 'w') as f:
        pattern_dict = {}
        for item in i2e:
                key = item
                lemma_list = item.split('_')
                lemma_dict_list = []
                for lemma in lemma_list:
                    lemma_dict = {"LEMMA": lemma}
                    lemma_dict_list.append(lemma_dict)
                pattern_dict.update({key : lemma_dict_list})
        f.write(json.dumps(pattern_dict))

write_vocab_path = "data/my_conceptnet/concept.txt"
write_patterns_path = "data/my_conceptnet/matcher_patterns.json"

read_path = [
    "data/DailyDialog/test/raw.json"
        ]
write_path = [
        "2_dialog_generation/concepts_grounded_data/concepts_grounded_dd_test.json"
        ]

# 提取数据集中每一句话的concepts，用于后面在图谱上匹配路径
for dataset_path, output_path in zip(read_path, write_path):
    ground_TG(dataset_path, write_vocab_path, write_patterns_path, output_path, cpu_count(), False)

# 217288/217288
# nohup python -u 2_dialog_generation/concepts_grounded.py > 2_dialog_generation/logs/concepts_grounded_dev.log 2>&1
