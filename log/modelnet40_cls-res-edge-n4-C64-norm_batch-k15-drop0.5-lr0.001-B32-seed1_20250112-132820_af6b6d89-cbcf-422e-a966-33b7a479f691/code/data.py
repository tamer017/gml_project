#!/usr/bin/env python
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch


# def download(data_dir):
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#     if not os.path.exists(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048')):
#         www = 'https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip?download=true'
#         zipfile = os.path.basename(www)
#         os.system('wget %s; unzip %s' % (www, zipfile))
#         os.system('mv %s %s' % (zipfile[:-18], data_dir))
#         os.system('rm %s' % (zipfile))

# def load_data(data_dir, partition):
#     download(data_dir)
#     all_data = []
#     all_label = []
#     for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
#         with h5py.File(h5_name, 'r') as f:
#             data = f['data'][:].astype('float32')
#             label = f['label'][:].astype('int64')
#         all_data.append(data)
#         all_label.append(label)
#     all_data = np.concatenate(all_data, axis=0)
#     all_label = np.concatenate(all_label, axis=0)
#     return all_data, all_label


def download(data_dir, dataset_name="ModelNet40"):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if dataset_name == "ModelNet40":
        dataset_dir = os.path.join(data_dir, 'modelnet40_ply_hdf5_2048')
        if not os.path.exists(dataset_dir):
            print("Downloading ModelNet40 dataset...")
            url = 'https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip?download=true'
            zipfile = os.path.basename(url)
            os.system(f'wget {url}; unzip {zipfile}')
            os.system(f'mv {zipfile[:-18]} {data_dir}')
            os.system(f'rm {zipfile}')
            print("ModelNet40 dataset downloaded and extracted.")

    elif dataset_name == "ShapeNetPart":
        dataset_dir = os.path.join(data_dir, 'shapenetpart-hdf5-2048')
        if not os.path.exists(dataset_dir):
            print("Downloading ShapeNetPart dataset...")
            zip_file_path = os.path.join(data_dir, "shapenetpart-hdf5-2048.zip")
            kaggle_dataset = "horsek/shapenetpart-hdf5-2048"
            os.system(f'kaggle datasets download -d {kaggle_dataset} -p {data_dir}')
            os.system(f'unzip {zip_file_path} -d {data_dir}/shapenetpart-hdf5-2048')
            os.system(f'rm {zip_file_path}')
            print("ShapeNetPart dataset downloaded and extracted.")
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_data(data_dir, partition, dataset_name="ModelNet40"):
    download(data_dir, dataset_name)
    all_data = []
    all_labels = []

    if dataset_name == "ModelNet40":
        path = os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', f'ply_data_{partition}*.h5')
    elif dataset_name == "ShapeNetPart":
        path = os.path.join(data_dir, 'shapenetpart-hdf5-2048', f'*{partition}*.h5')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    for h5_name in glob.glob(path):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            labels = f['label'][:].astype('int64')
        all_data.append(data)
        all_labels.append(labels)

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0) 

    return all_data, all_labels


def translate_pointcloud(pointcloud):
    """
    for scaling and shifting the point cloud
    :param pointcloud:
    :return:
    """
    scale = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, scale), shift).astype('float32')
    return translated_pointcloud

def rotate_pointcloud(pointcloud):
    theta = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return np.dot(pointcloud, rotation_matrix)

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    noise = np.clip(sigma * np.random.randn(*pointcloud.shape), -clip, clip)
    return pointcloud + noise


class ModelNet40(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    def __init__(self, num_points=1024, data_dir="/data/deepgcn/modelnet40", partition='train'):
        self.data, self.label = load_data(data_dir, partition, dataset_name="ModelNet40")
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        # Normalize point cloud
        pointcloud = self.normalize_pointcloud(pointcloud)
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label
        
    # def __getitem__(self, item):
    #     pointcloud = self.data[item][:self.num_points]
    #     label = self.label[item]

    #     # Normalize point cloud
    #     pointcloud = self.normalize_pointcloud(pointcloud)

    #     # Apply augmentations if in training mode
    #     if self.partition == 'train':
    #         pointcloud = translate_pointcloud(pointcloud)
    #         pointcloud = rotate_pointcloud(pointcloud)
    #         pointcloud = jitter_pointcloud(pointcloud)
    #         np.random.shuffle(pointcloud)

    #     return torch.tensor(pointcloud, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
    
    @staticmethod
    def normalize_pointcloud(pointcloud):
        """
        Normalize the point cloud to be centered and scaled within a unit sphere.
        """
        centroid = np.mean(pointcloud, axis=0)
        pointcloud -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(pointcloud**2, axis=1)))
        pointcloud /= furthest_distance
        return pointcloud

    def __len__(self):
        return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1

class ShapeNetPart(Dataset):
    """
    ShapeNetPart dataset loader.
    """
    def __init__(self, data_dir="/data/deepgcn/modelnet40", num_points=2048, partition='train'):
        self.data, self.labels = load_data(data_dir, partition, dataset_name="ShapeNetPart")
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, index):
        pointcloud = self.data[index][:self.num_points]
        label = self.labels[index]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return len(self.data)

    def num_classes(self):
        return np.max(self.labels) + 1
    
if __name__ == '__main__':
    train = ModelNet40(2048)
    test = ModelNet40(2048, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
