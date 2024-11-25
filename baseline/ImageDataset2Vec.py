import time
import timm
import clip
import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from config import *
from slice_dataset import get_all_datasets_and_idx

META_FEATURE_PATH = 'meta_features/ImageDataset2Vec'

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_meta_features(task_features, num_classes=5) -> np.ndarray:
    task_features_normalized = task_features.cpu().numpy()
    kmeans = KMeans(n_clusters=num_classes)
    kmeans.fit(task_features_normalized)
    centroids = kmeans.cluster_centers_
    wk = centroids.mean(axis=0)
    return wk


# Preprocess image and extract the metafeature
def extract_meta_features(dataset_images, preprocess, mnasnet_model, batch_size=64, num_classes=5):
    meta_features_list = []

    for i in tqdm(range(0, len(dataset_images), batch_size)):
        batch_images = dataset_images[i:i + batch_size]
        batch_input = [preprocess(image).unsqueeze(0).to(device) for image in batch_images]
        batch_input = torch.cat(batch_input)

        with torch.no_grad():
            batch_features = mnasnet_model(batch_input)

        batch_features /= batch_features.norm(dim=-1, keepdim=True)
        meta_features_list.append(batch_features.cpu())

    all_features = torch.cat(meta_features_list)
    print(all_features.shape)
    meta_features = make_meta_features(all_features, num_classes=num_classes)

    return meta_features


if __name__ == '__main__':
    pretrained_cfg_overlay = {'file': r"/home/pod/.cache/huggingface/hub/models--timm--mnasnet_100.rmsp_in1k/pytorch_model.bin"}
    model = timm.create_model('mnasnet_100', pretrained=True, pretrained_cfg_overlay=pretrained_cfg_overlay).to(device)

    _, preprocessor = clip.load('ViT-B/32', device)
    # model = timm.create_model('mnasnet_100', pretrained=True).to(device)
    model.reset_classifier(0)

    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        image_key = dataset_config['image_key']
        label_key = dataset_config['label_key']

        items = get_all_datasets_and_idx(dataset_name)
        for dataset, _, dataset_comb_name in tqdm(items):
            num_classes = dataset_comb_name.count('_')
            label_dict = dataset.features[label_key].names
            imgs = []

            # Test option
            # dataset = dataset.select(range(64))

            for idx, item in enumerate(dataset):
                label = label_dict[item[label_key]]
                img = item[image_key]
                imgs.append(img)

            s_time = time.time()
            meta_features = extract_meta_features(imgs, preprocessor, model, num_classes=num_classes)
            e_time = time.time()
            print(f'time:{e_time - s_time}')
            np.save(f'{META_FEATURE_PATH}/{dataset_comb_name}', meta_features)