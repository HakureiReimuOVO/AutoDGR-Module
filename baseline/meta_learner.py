import os
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier
from acc_loader import *
import metric
from config import *
from slice_dataset import get_all_datasets_and_idx
from feature_loader import *
from train_test_indice import *
from evaluate_metrics import *
import itertools

model_save_path = 'saved_models'


def generate_combs(n):
    indices = list(range(n))
    combinations = list(itertools.combinations(indices, 2))
    return combinations


combs = generate_combs(len(model_configs))

print(combs)


def padding(idxs, elements, k=8):
    assert len(idxs) == len(elements)
    res = [0] * k
    for i in range(len(idxs)):
        res[idxs[i]] = elements[i]
    return res


num_tasks = 160
feature_length = 512

# Models count
num_classes = 10

wd_list = []
y_list = []

datasets = []
models = model_configs

meta = 'img'
# meta = 'mrm'
# meta = 'sgr'

accuracies = extract_accuracies('result/best_accuracies.csv')
for dataset_config in dataset_configs:
    items = get_all_datasets_and_idx(dataset_name=dataset_config['name'])
    for _, _, dataset_name in items:
        datasets.append(dataset_name)
        # meta_feature = get_imgdataset2vec_features(dataset_name)
        # meta_feature = get_mrm_features(dataset_name)
        # meta_feature = get_sgr_features(dataset_name)
        if meta == 'img':
            # ImageDataset2Vec
            meta_feature = get_imgdataset2vec_features(dataset_name)
        elif meta == 'mrm':
            # AutoMRM
            meta_feature = get_mrm_features(dataset_name)
            print(meta_feature.shape)
        elif meta == 'sgr':
            # AutoSGR
            meta_feature = get_sgr_features(dataset_name)

        acc = accuracies[dataset_name]
        vec = accuracies_to_idx(acc)

        wd_list.append(meta_feature)
        y_list.append(vec)

label_encoder = LabelEncoder()
y_list = label_encoder.fit_transform(y_list)
unique_y = label_encoder.classes_

wd_list = np.array(wd_list)
y_list = np.array(y_list)

# y_list = np.random.randint(0, num_classes, size=wd_list.shape[0])
# label_encoder = LabelEncoder()
# y_list = label_encoder.fit_transform(y_list)
# base_labels = np.arange(num_classes)
# remaining_labels = wd_list.shape[0] - num_classes
# random_labels = np.random.randint(0, num_classes, size=remaining_labels)
# y_list = np.concatenate([base_labels, random_labels])
# np.random.shuffle(y_list)

# (45, 2048)
print(wd_list.shape)

# (45)
print(y_list.shape)
print(y_list)


class Model(object):
    def __init__(self, name, SDS_acc, real_acc):
        self.name = name
        self.SDS_acc = SDS_acc
        self.real_acc = real_acc


#  Split the training and testing sets into 4:1 and repeat 100 times to obtain experimental results
acc_sum = 0.0
acc_list = []
ndcg5_t = []
mrr_t = []
map_t = []

# Select the meta learner to use Ensemble learning or not according to the task
# model = RandomForestClassifier()
# model = BaggingClassifier(base_estimator=SVC(probability=True), max_samples=0.8, max_features=0.8, n_estimators=100,
#                           bootstrap_features=True, n_jobs=-1)
model = XGBClassifier(random_state=1, learning_rate=0.1, use_label_encoder=False)
pipe_lr = make_pipeline(StandardScaler(), model)

res = []
res_map = []
X_train_list = []
X_test_list = []
y_train_list = []
y_test_list = []

x_fin = []
y_fin = []

for i in fin_indices:
    x_fin.append(wd_list[i])
    y_fin.append(y_list[i])

for j in train_indices:
    X_train_list.append(wd_list[j])
    y_train_list.append(y_list[j])

for k in test_indices:
    X_test_list.append(wd_list[k])
    y_test_list.append(y_list[k])

X_train = np.array(X_train_list)
X_test = np.array(X_test_list)
y_train = np.array(y_train_list)
y_test = np.array(y_test_list)
x_fin = np.array(x_fin)
y_fin = np.array(y_fin)

train_labels = Counter(y_train)
test_labels = Counter(y_test)

print("Training Labels:", train_labels)
print("Testing Labels:", test_labels)

if fin_test:
    epoch = 50
    with open(os.path.join(model_save_path, f"{meta}_meta_learner_epoch_{epoch}.pkl"), 'rb') as m:
        pipe_lr = pickle.load(m)
    start_time = time.time()
    prediction_pro = pipe_lr.predict_proba(x_fin)
    end_time = time.time()
    print(f"Total Time: {end_time - start_time} seconds")
    rmv_cnt = 0
    rmv2_cnt = 0
    rmv3_cnt = 0
    rmv4_cnt = 0
    rmv5_cnt = 0
    precision_cnt = 0
    recall_cnt = 0
    mrr_cnt = 0
    mrr2_cnt = 0
    mrr3_cnt = 0
    mrr4_cnt = 0
    mrr5_cnt = 0
    map_cnt = 0
    map2_cnt = 0
    map3_cnt = 0
    map4_cnt = 0
    map5_cnt = 0
    ndcg_cnt = 0
    ndcg2_cnt = 0
    ndcg3_cnt = 0
    ndcg4_cnt = 0
    ndcg5_cnt = 0
    binary_acc = 0
    binary_cnt = 0

    for x in range(len(prediction_pro)):
        model_list = []

        m = max(prediction_pro[x])
        a = list(prediction_pro[x]).index(m)
        idx = unique_y[a]
        model = models[idx]
        dataset = datasets[x]

        p = prediction_pro[x]
        pad_p = padding(unique_y, p, k=len(model_configs))
        l = list(accuracies[dataset].values())

        for comb in combs:
            p_x = pad_p[comb[0]]
            p_y = pad_p[comb[1]]
            l_x = l[comb[0]]
            l_y = l[comb[1]]
            binary_cnt += 1
            if l_x == l_y:
                binary_acc += 1
            elif p_x > p_y and l_x > l_y:
                binary_acc += 1
            elif p_x < p_y and l_x < l_y:
                binary_acc += 1

        rmv_cnt += rmv(pad_p, l)
        rmv2_cnt += rmv(pad_p, l, 2)
        rmv3_cnt += rmv(pad_p, l, 3)
        rmv4_cnt += rmv(pad_p, l, 4)
        rmv5_cnt += rmv(pad_p, l, 5)
        precision_cnt += precision_at_k(pad_p, l, 3)
        recall_cnt += recall_at_k(pad_p, l, 3)
        a = mrr_at_k(pad_p, l, 1)
        mrr_cnt += mrr_at_k(pad_p, l, 1)
        mrr2_cnt += mrr_at_k(pad_p, l, 2)
        mrr3_cnt += mrr_at_k(pad_p, l, 3)
        mrr4_cnt += mrr_at_k(pad_p, l, 4)
        mrr5_cnt += mrr_at_k(pad_p, l, 5)
        map_cnt += map_at_k(pad_p, l, 1)
        map2_cnt += map_at_k(pad_p, l, 2)
        map3_cnt += map_at_k(pad_p, l, 3)
        map4_cnt += map_at_k(pad_p, l, 4)
        map5_cnt += map_at_k(pad_p, l, 5)
        ndcg_cnt += ndcg_at_k(pad_p, l, 1)
        ndcg2_cnt += ndcg_at_k(pad_p, l, 2)
        ndcg3_cnt += ndcg_at_k(pad_p, l, 3)
        ndcg4_cnt += ndcg_at_k(pad_p, l, 4)
        ndcg5_cnt += ndcg_at_k(pad_p, l, 5)

    rmv_cnt = rmv_cnt / len(prediction_pro)
    rmv2_cnt = rmv2_cnt / len(prediction_pro)
    rmv3_cnt = rmv3_cnt / len(prediction_pro)
    rmv4_cnt = rmv4_cnt / len(prediction_pro)
    rmv5_cnt = rmv5_cnt / len(prediction_pro)
    precision_cnt = precision_cnt / len(prediction_pro)
    recall_cnt = recall_cnt / len(prediction_pro)
    mrr_cnt = mrr_cnt / len(prediction_pro)
    mrr2_cnt = mrr2_cnt / len(prediction_pro)
    mrr3_cnt = mrr3_cnt / len(prediction_pro)
    mrr4_cnt = mrr4_cnt / len(prediction_pro)
    mrr5_cnt = mrr5_cnt / len(prediction_pro)
    map_cnt = map_cnt / len(prediction_pro)
    map2_cnt = map2_cnt / len(prediction_pro)
    map3_cnt = map3_cnt / len(prediction_pro)
    map4_cnt = map4_cnt / len(prediction_pro)
    map5_cnt = map5_cnt / len(prediction_pro)
    ndcg_cnt = ndcg_cnt / len(prediction_pro)
    ndcg2_cnt = ndcg2_cnt / len(prediction_pro)
    ndcg3_cnt = ndcg3_cnt / len(prediction_pro)
    ndcg4_cnt = ndcg4_cnt / len(prediction_pro)
    ndcg5_cnt = ndcg5_cnt / len(prediction_pro)

    tmp = pipe_lr.score(X_test, y_test)
    acc_sum += tmp
    acc_list.append(tmp)
    print(f'acc: {tmp}')
    print(f'rmv: {rmv_cnt}')
    print(f'rmv2: {rmv2_cnt}')
    print(f'rmv3: {rmv3_cnt}')
    print(f'rmv4: {rmv4_cnt}')
    print(f'rmv5: {rmv5_cnt}')
    print(f'precision: {precision_cnt}')
    print(f'recall: {recall_cnt}')
    print(f'mrr: {mrr_cnt}')
    print(f'mrr2: {mrr2_cnt}')
    print(f'mrr3: {mrr3_cnt}')
    print(f'mrr4: {mrr4_cnt}')
    print(f'mrr5: {mrr5_cnt}')
    print(f'map: {map_cnt}')
    print(f'map2: {map2_cnt}')
    print(f'map3: {map3_cnt}')
    print(f'map4: {map4_cnt}')
    print(f'map5: {map5_cnt}')
    print(f'ndcg: {ndcg_cnt}')
    print(f'ndcg2: {ndcg2_cnt}')
    print(f'ndcg3: {ndcg3_cnt}')
    print(f'ndcg4: {ndcg4_cnt}')
    print(f'ndcg5: {ndcg5_cnt}')

    print(f'binary acc: {binary_acc / binary_cnt}')

else:
    s_time = time.time()
    for i in range(100):
        # indices = np.arange(y_list.shape[0])
        # [indices_train, indices_test, y_train, y_test] = train_test_split(indices, y_list, test_size=0.20, stratify=y_list)
        # X_train_list = []
        # X_test_list = []
        # for j in indices_train:
        #     X_train_list.append(wd_list[j])
        # X_train = np.array(X_train_list)
        # for k in indices_test:
        #     X_test_list.append(wd_list[k])
        # X_test = np.array(X_test_list)

        pipe_lr.fit(X_train, y_train)
        prediction_pro = pipe_lr.predict_proba(X_test)

        # res = df.iloc[indices_test]
        # res = res.reset_index(drop=True)

        rmv_cnt = 0
        rmv2_cnt = 0
        rmv3_cnt = 0
        rmv4_cnt = 0
        rmv5_cnt = 0
        precision_cnt = 0
        recall_cnt = 0
        mrr_cnt = 0
        mrr2_cnt = 0
        mrr3_cnt = 0
        mrr4_cnt = 0
        mrr5_cnt = 0
        map_cnt = 0
        map2_cnt = 0
        map3_cnt = 0
        map4_cnt = 0
        map5_cnt = 0
        ndcg_cnt = 0
        ndcg2_cnt = 0
        ndcg3_cnt = 0
        ndcg4_cnt = 0
        ndcg5_cnt = 0
        binary_acc = 0
        binary_cnt = 0

        for x in range(len(prediction_pro)):
            model_list = []

            m = max(prediction_pro[x])
            a = list(prediction_pro[x]).index(m)
            idx = unique_y[a]
            model = models[idx]
            dataset = datasets[x]
            # acc = accuracies[dataset][model]
            # max_acc = max(accuracies[dataset].values())
            # rmv_list.append(acc / max_acc)

            p = prediction_pro[x]
            pad_p = padding(unique_y, p, k=len(model_configs))
            l = list(accuracies[dataset].values())

            for comb in combs:
                p_x = pad_p[comb[0]]
                p_y = pad_p[comb[1]]
                l_x = l[comb[0]]
                l_y = l[comb[1]]
                binary_cnt += 1
                if l_x == l_y:
                    binary_acc += 1
                elif p_x > p_y and l_x > l_y:
                    binary_acc += 1
                elif p_x < p_y and l_x < l_y:
                    binary_acc += 1

            # map_list.append(map_k(pad_p, l, 3))

            rmv_cnt += rmv(pad_p, l)
            rmv2_cnt += rmv(pad_p, l, 2)
            rmv3_cnt += rmv(pad_p, l, 3)
            rmv4_cnt += rmv(pad_p, l, 4)
            rmv5_cnt += rmv(pad_p, l, 5)
            precision_cnt += precision_at_k(pad_p, l, 3)
            recall_cnt += recall_at_k(pad_p, l, 3)
            mrr_cnt += mrr_at_k(pad_p, l, 1)
            mrr2_cnt += mrr_at_k(pad_p, l, 2)
            mrr3_cnt += mrr_at_k(pad_p, l, 3)
            mrr4_cnt += mrr_at_k(pad_p, l, 4)
            mrr5_cnt += mrr_at_k(pad_p, l, 5)
            map_cnt += map_at_k(pad_p, l, 1)
            map2_cnt += map_at_k(pad_p, l, 2)
            map3_cnt += map_at_k(pad_p, l, 3)
            map4_cnt += map_at_k(pad_p, l, 4)
            map5_cnt += map_at_k(pad_p, l, 5)
            ndcg_cnt += ndcg_at_k(pad_p, l, 1)
            ndcg2_cnt += ndcg_at_k(pad_p, l, 2)
            ndcg3_cnt += ndcg_at_k(pad_p, l, 3)
            ndcg4_cnt += ndcg_at_k(pad_p, l, 4)
            ndcg5_cnt += ndcg_at_k(pad_p, l, 5)

            # arr = np.array(prediction_pro)
            # print(np.argmax(arr, axis=1))

            for y in range(len(prediction_pro[0])):
                model_name = y
                mSDS_acc = prediction_pro[x][y]
                mReal_acc = 1
            # ndcg5.append(metric.NDCG(model_list, 5, 1)[0])
            # mrr.append(metric.MRR(model_list, 1)[0])
            # map.append(metric.MAP(model_list, 3, 1)[0])

        rmv_cnt = rmv_cnt / len(prediction_pro)
        rmv2_cnt = rmv2_cnt / len(prediction_pro)
        rmv3_cnt = rmv3_cnt / len(prediction_pro)
        rmv4_cnt = rmv4_cnt / len(prediction_pro)
        rmv5_cnt = rmv5_cnt / len(prediction_pro)
        precision_cnt = precision_cnt / len(prediction_pro)
        recall_cnt = recall_cnt / len(prediction_pro)
        mrr_cnt = mrr_cnt / len(prediction_pro)
        mrr2_cnt = mrr2_cnt / len(prediction_pro)
        mrr3_cnt = mrr3_cnt / len(prediction_pro)
        mrr4_cnt = mrr4_cnt / len(prediction_pro)
        mrr5_cnt = mrr5_cnt / len(prediction_pro)
        map_cnt = map_cnt / len(prediction_pro)
        map2_cnt = map2_cnt / len(prediction_pro)
        map3_cnt = map3_cnt / len(prediction_pro)
        map4_cnt = map4_cnt / len(prediction_pro)
        map5_cnt = map5_cnt / len(prediction_pro)
        ndcg_cnt = ndcg_cnt / len(prediction_pro)
        ndcg2_cnt = ndcg2_cnt / len(prediction_pro)
        ndcg3_cnt = ndcg3_cnt / len(prediction_pro)
        ndcg4_cnt = ndcg4_cnt / len(prediction_pro)
        ndcg5_cnt = ndcg5_cnt / len(prediction_pro)

        tmp = pipe_lr.score(X_test, y_test)
        acc_sum += tmp
        acc_list.append(tmp)
        print(f'epoch{i}, acc: {tmp}')
        print(f'rmv: {rmv_cnt}')
        print(f'rmv2: {rmv2_cnt}')
        print(f'rmv3: {rmv3_cnt}')
        print(f'rmv4: {rmv4_cnt}')
        print(f'rmv5: {rmv5_cnt}')
        print(f'precision: {precision_cnt}')
        print(f'recall: {recall_cnt}')
        print(f'mrr: {mrr_cnt}')
        print(f'mrr2: {mrr2_cnt}')
        print(f'mrr3: {mrr3_cnt}')
        print(f'mrr4: {mrr4_cnt}')
        print(f'mrr5: {mrr5_cnt}')
        print(f'map: {map_cnt}')
        print(f'map2: {map2_cnt}')
        print(f'map3: {map3_cnt}')
        print(f'map4: {map4_cnt}')
        print(f'map5: {map5_cnt}')
        print(f'ndcg: {ndcg_cnt}')
        print(f'ndcg2: {ndcg2_cnt}')
        print(f'ndcg3: {ndcg3_cnt}')
        print(f'ndcg4: {ndcg4_cnt}')
        print(f'ndcg5: {ndcg5_cnt}')
        print(f'binary acc: {binary_acc / binary_cnt}')
        print('=================')
        # ndcg5_t.append(sum(ndcg5) / len(ndcg5))
        # mrr_t.append(sum(mrr) / len(mrr))
        # map_t.append(sum(map) / len(map))
        if (i + 1) % 10 == 0:
            with open(os.path.join(model_save_path, f"{meta}_meta_learner_epoch_{i + 1}.pkl"), 'wb') as model_file:
                pickle.dump(pipe_lr, model_file)
            print(f'Model saved at epoch {i + 1}.')
    e_time = time.time()
    print(f'total time:{e_time - s_time}')
    acc_avg = acc_sum / len(acc_list)
    print("acc: ", acc_avg)
