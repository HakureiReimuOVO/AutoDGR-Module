import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import pickle as pkl
from torchvision import models
from mmcv import Config
from mmcls.datasets import build_dataset, build_dataloader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

import random

# TODO: import together
from mmcls_addon.datasets.caltech import CALTECH

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Build dataset graph")
    parser.add_argument("folder", help="Folder containing dataset config files")
    parser.add_argument("output", help="Output file to save the graph")
    args = parser.parse_args()
    return args


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        # Test
        dataloader = random.sample(list(dataloader), 20)
        for data in tqdm(dataloader):
            images = data["img"].to(device)
            outputs = model(images)
            features.append(outputs.cpu())
    features = torch.cat(features, dim=0)
    return features


def get_dataset_features(config_path):
    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.train)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()
    model = model.to(device)
    features = extract_features(model, dataloader, device)
    mean_features = features.mean(dim=0).numpy()
    return mean_features


def build_graph(folder):
    features = []
    config_files = [
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".py")
    ]
    for config_file in config_files:
        mean_features = get_dataset_features(config_file)
        features.append(mean_features)
    features = np.array(features)
    similarity_matrix = cosine_similarity(features)
    print(similarity_matrix)
    G = nx.Graph()
    for i, config_file in enumerate(config_files):
        G.add_node(i, name=os.path.basename(config_file), feature=features[i])
    for i in range(len(config_files)):
        for j in range(i + 1, len(config_files)):
            G.add_edge(i, j, weight=similarity_matrix[i, j])
    return G


def save_graph(G, output_file):
    with open(output_file, "wb") as f:
        pkl.dump(G, f)


if __name__ == "__main__":
    args = parse_args()
    G = build_graph(args.folder)
    save_graph(G, args.output)
    print(f"Dataset graph saved to {args.output}")
