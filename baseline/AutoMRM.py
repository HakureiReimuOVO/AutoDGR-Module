import time

import clip
import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from config import *
from slice_dataset import get_all_datasets_and_idx

META_FEATURE_PATH = 'meta_features/AutoMRM/'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def make_task_meta_features(task_features: np.ndarray, num_classes) -> np.ndarray:
    task_features_normalized = task_features.cpu().numpy()
    nfeature = np.vsplit(task_features_normalized, num_classes)
    for i in range(num_classes):
        w = make_task_meta_features2(nfeature[i])
        if i == 0:
            ww = w
        else:
            ww = np.append(ww, w)
    return ww


def make_task_meta_features2(task_features: np.ndarray) -> np.ndarray:
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(task_features)
    centroids = kmeans.cluster_centers_
    wk = centroids[0]
    return wk


def extract_meta_features(dataset_images, dataset_labels, preprocess, model, batch_size=64):
    wi_list = []
    wt_list = []

    for i in tqdm(range(0, len(dataset_images), batch_size)):
        batch_images = dataset_images[i:i + batch_size]
        batch_labels = dataset_labels[i:i + batch_size]
        batch_input = [preprocess(image).unsqueeze(0).to(device) for image in batch_images]
        batch_input = torch.cat(batch_input)

        # Image
        with torch.no_grad():
            batch_image_features = model.encode_image(batch_input)
        batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
        wi_list.append(batch_image_features.cpu())

        # Text
        text_input = torch.cat([clip.tokenize(f"photos of {label}.") for label in batch_labels]).to(device)
        with torch.no_grad():
            batch_text_features = model.encode_text(text_input)

        # Not normalized in the original AutoMRM
        batch_text_features /= batch_text_features.norm(dim=-1, keepdim=True)
        wt_list.append(batch_text_features.cpu())

    all_image_features = torch.cat(wi_list)
    all_text_features = torch.cat(wt_list)

    unique_labels = set(labels)
    all_labels_str = ', '.join(unique_labels)
    task_input = clip.tokenize(f"photos of {all_labels_str}.").to(device)

    with torch.no_grad():
        task_feature = model.encode_text(task_input)
    task_feature /= task_feature.norm(dim=-1, keepdim=True)
    task_feature = task_feature.cpu().numpy()

    image_meta_features = make_task_meta_features(all_image_features, num_classes=len(unique_labels))
    text_meta_features = make_task_meta_features(all_text_features, num_classes=len(unique_labels))

    f_meta_features = 0.5 * image_meta_features + 0.5 * text_meta_features
    f_meta_features = np.concatenate((f_meta_features, task_feature[0]), axis=0)

    return f_meta_features


if __name__ == '__main__':
    _, preprocessor = clip.load('ViT-B/32', device)
    model = clip.load('ViT-B/32', device)[0]

    for dataset_config in dataset_configs:
        dataset_name = dataset_config['name']
        image_key = dataset_config['image_key']
        label_key = dataset_config['label_key']

        items = get_all_datasets_and_idx(dataset_name)
        for dataset, _, dataset_comb_name in tqdm(items):
            print(dataset_comb_name)
            label_dict = dataset.features[label_key].names
            imgs = []
            labels = []

            # Test option
            # dataset = dataset.select(range(64))

            for idx, item in enumerate(dataset):
                label = label_dict[item[label_key]]
                labels.append(label)
                img = item[image_key]
                imgs.append(img)

            sorted_indices = np.argsort(labels)
            sorted_imgs = [imgs[i] for i in sorted_indices]
            sorted_labels = [labels[i] for i in sorted_indices]
            s_time = time.time()
            meta_features = extract_meta_features(sorted_imgs, sorted_labels, preprocessor, model)
            e_time = time.time()
            print(f'time:{e_time - s_time}')
            np.save(f'{META_FEATURE_PATH}/{dataset_comb_name}', meta_features)
