import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
import numpy as np
import networkx as nx
from mmcv import Config
from mmcls.models import build_classifier
from mmcls.datasets import build_dataset, build_dataloader
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def average_pool(feature, target_length=512):
    if feature.dim() != 1:
        raise ValueError("Feature must be a 1D tensor")

    feature = feature.unsqueeze(0).unsqueeze(0)
    pooled_feature = F.adaptive_avg_pool1d(feature, target_length)
    pooled_feature = pooled_feature.squeeze(0).squeeze(0)

    return pooled_feature


def parse_args():
    parser = argparse.ArgumentParser(description="Build model graph")
    parser.add_argument("folder", help="Folder containing model config files")
    parser.add_argument("dataset_config", help="Dataset config file path")
    parser.add_argument("output", help="Output file to save the graph")
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


def build_graph(folder, dataset_config):
    features = []
    config_files = [
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".py")
    ]
    for config_file in config_files:
        mean_features = get_model_features(dataset_config, config_file)
        pooled_features = average_pool(torch.tensor(mean_features), target_length=512).numpy()
        features.append(pooled_features)

    features = np.array(features)
    similarity_matrix = cosine_similarity(features)
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
    G = build_graph(args.folder, args.dataset_config)
    save_graph(G, args.output)
    print(f"Model graph saved to {args.output}")
