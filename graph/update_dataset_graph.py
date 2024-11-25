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


def parse_args():
    parser = argparse.ArgumentParser(description="Update dataset graph")
    parser.add_argument("config", help="Dataset config file path")
    parser.add_argument("graph", help="Existing graph file")
    parser.add_argument("output", help="Output file to save the updated graph")
    args = parser.parse_args()
    return args


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
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


def update_graph(graph_file, config_file):
    with open(graph_file, "rb") as f:
        G = pkl.load(f)
    mean_features = get_dataset_features(config_file)
    new_node_id = len(G.nodes)
    G.add_node(new_node_id, name=os.path.basename(config_file), feature=mean_features)
    for node_id in G.nodes:
        if node_id != new_node_id:
            existing_feature = G.nodes[node_id]["feature"]
            similarity = cosine_similarity([mean_features], [existing_feature])[0][0]
            G.add_edge(new_node_id, node_id, weight=similarity)
    return G


def save_graph(G, output_file):
    with open(output_file, "wb") as f:
        pkl.dump(G, f)


if __name__ == "__main__":
    args = parse_args()
    G = update_graph(args.graph, args.config)
    save_graph(G, args.output)
    print(f"Updated dataset graph saved to {args.output}")
