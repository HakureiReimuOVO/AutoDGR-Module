import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from mmcv import Config
from mmcls.models import build_classifier
from mmcls.datasets import build_dataset, build_dataloader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pickle as pkl


def average_pool(feature, target_length=512):
    if feature.dim() != 1:
        raise ValueError("Feature must be a 1D tensor")

    feature = feature.unsqueeze(0).unsqueeze(0)
    pooled_feature = F.adaptive_avg_pool1d(feature, target_length)
    pooled_feature = pooled_feature.squeeze(0).squeeze(0)

    return pooled_feature


def parse_args():
    parser = argparse.ArgumentParser(description="Update model graph")
    parser.add_argument("config", help="Model config file path")
    parser.add_argument("dataset_config", help="Dataset config file path")
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
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Config.fromfile(model_config)
    model = build_classifier(cfg.model)
    model = model.to(device)

    features = extract_features(model, dataloader, device)
    mean_features = features.mean(dim=0).numpy()

    return mean_features


def update_graph(graph_file, dataset_config, config_file):
    with open(graph_file, "rb") as f:
        G = pkl.load(f)
    mean_features = get_model_features(dataset_config, config_file)
    pooled_features = average_pool(
        torch.tensor(mean_features), target_length=512
    ).numpy()
    new_node_id = len(G.nodes)
    G.add_node(new_node_id, name=os.path.basename(config_file), feature=pooled_features)
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
    G = update_graph(args.graph, args.dataset_config, args.config)
    save_graph(G, args.output)
    print(f"Updated model graph saved to {args.output}")
