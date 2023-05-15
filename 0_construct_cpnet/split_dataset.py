# 划分数据集train.txt, dev.txt and test.txt by ratio of 0.9:0.05:0.05 
import random

def split(all_data_path, train_path, dev_path, test_path, ratio=[0.9, 0.05, 0.05]):
    with open(all_data_path, "r") as f:
        data = f.readlines()
    random.shuffle(data)
    train_data = data[:int(len(data)*ratio[0])]
    dev_data = data[int(len(data)*ratio[0]):int(len(data)*(ratio[0]+ratio[1]))]
    test_data = data[int(len(data)*(ratio[0]+ratio[1])):]
    with open(train_path, "w") as f:
        f.write("".join(train_data))
    with open(dev_path, "w") as f:
        f.write("".join(dev_data))
    with open(test_path, "w") as f:
        f.write("".join(test_data))
    print("train: {}, dev: {}, test: {}".format(len(train_data), len(dev_data), len(test_data)))

if __name__ == "__main__":
    all_data_path = "../0_construct_cpnet/local_paths/sample_paths.txt"
    train_path = "../0_construct_cpnet/local_paths/train.txt"
    dev_path = "../0_construct_cpnet/local_paths/dev.txt"
    test_path = "../0_construct_cpnet/local_paths/test.txt"
    split(all_data_path, train_path, dev_path, test_path)