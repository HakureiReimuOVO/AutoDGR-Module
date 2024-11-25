import argparse
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from torchvision import models
from mmcv import Config
from mmcls.models import build_classifier
from mmcls.datasets import build_dataset, build_dataloader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import os

os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"

def parse_args():
    parser = argparse.ArgumentParser(description="Build dataset graph")
    parser.add_argument("config", help="Dataset config file path")
    args = parser.parse_args()
    return args

def extract_features(model, dataloader, device):
    model.eval()

    features = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            images = data['img'].to(device)
            backbone_features = model.backbone(images)
            if isinstance(backbone_features, tuple):
                backbone_features = backbone_features[0]
            neck_features = model.neck(backbone_features)
            features.append(neck_features.cpu())
    features = torch.cat(features, dim=0)
    return features

def get_model_features(dataset_config, model_config):
    cfg = Config.fromfile(dataset_config)
    dataset = build_dataset(cfg.data.train)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config.fromfile(model_config)
    model = build_classifier(cfg.model)
    model = model.to(device)
    
    print(model)
    print(type(model))

    features = extract_features(model, dataloader, device)
    mean_features = features.mean(dim=0).numpy()

    return mean_features


def build_graph(model_configs):
    embeddings = []
    model_names = []

    for config in model_configs:
        mean_features = get_model_features(config)
        embeddings.append(mean_features)
        model_names.append(os.path.basename(config))

    embeddings = np.array(embeddings)
    similarity_matrix = cosine_similarity(embeddings)

    G = nx.Graph()
    for i, name in enumerate(model_names):
        G.add_node(i, name=name, embedding=embeddings[i])

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            G.add_edge(i, j, weight=similarity_matrix[i, j])

    return G

def add_node_to_graph(G, config_path):
    mean_features = get_model_features(config_path)
    new_node_id = len(G.nodes)
    G.add_node(new_node_id, name=os.path.basename(config_path), embedding=mean_features)

    for node_id in G.nodes:
        if node_id != new_node_id:
            existing_embedding = G.nodes[node_id]['embedding']
            similarity = cosine_similarity([mean_features], [existing_embedding])[0][0]
            G.add_edge(new_node_id, node_id, weight=similarity)

def visualize_graph(G):
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'name')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=700, node_color='skyblue', font_size=10, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

if __name__ == "__main__":
    model_config = 'configs/models/resnet18.py'
    dataset_config = 'configs/datasets/test.py'
    res = get_model_features(dataset_config=dataset_config, model_config=model_config)
    
    print(res)
    
    # G = build_graph(model_configs)
    # visualize_graph(G)

    # # 添加新的节点
    # new_config_path = 'configs/models/model4.py'
    # add_node_to_graph(G, new_config_path)
    # visualize_graph(G)