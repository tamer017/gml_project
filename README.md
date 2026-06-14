# 3D Point Cloud Classification — DeepGCN on ModelNet40

> A geometric deep learning system for 3D point cloud classification using **Graph Convolutional Networks (GCNs)** on the ModelNet40 benchmark.

[![Framework](https://img.shields.io/badge/Framework-PyTorch-red?style=flat-square)](https://pytorch.org/)
[![Library](https://img.shields.io/badge/Library-PyTorch%20Geometric-blue?style=flat-square)](https://pytorch-geometric.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-ModelNet40-green?style=flat-square)](https://modelnet.cs.princeton.edu/)

---

## Overview

This project implements a deep learning-based **3D point cloud classification system** using Graph Convolutional Networks (GCNs). The primary objective is to classify point clouds into **40 distinct object categories** (chairs, airplanes, lamps, etc.) using the **ModelNet40 dataset** — a standard benchmark in geometric deep learning.

The system employs a **DeepGCN architecture** that constructs dynamic graph representations from unordered point clouds and applies stacked graph convolutional layers to extract hierarchical 3D structural features, enabling robust shape recognition without volumetric voxelization.

---

## Architecture — DeepGCN

```
[Input: N × 3 Point Cloud]
         |
         v
  [KNN Graph Construction]     <- knn_graph (k=20, Dilated KNN)
         |
         v
  [GCN Layer 1]                <- EdgeConv / Graph Conv + BN + ReLU
         |
         v
  [GCN Layer 2]
         |
         v
       [...]                   <- Residual connections between layers
         |
         v
  [Global Aggregation]         <- Max pooling over all nodes
         |
         v
  [Fully Connected Head]       <- FC(512) -> FC(256) -> FC(40)
         |
         v
  [Output: 40-class Softmax]
```

---

## Technical Highlights

### Dynamic Graph Construction
Uses `knn_graph` from `torch_cluster` to dynamically construct k-nearest-neighbor graphs from raw 3D point coordinates. The **DilatedKnnGraph** class supports receptive field expansion across layers without increasing k, analogous to dilated convolutions in 2D CNNs.

### Graph Convolutional Layers
Each GCN layer aggregates features from neighboring nodes using edge-conditioned convolutions, capturing local geometric relationships between points. Stacking multiple layers captures progressively global structure.

### Pairwise Distance Computation
The `pairwise_distance` function efficiently computes all-pairs squared Euclidean distances in the feature space, enabling dynamic graph re-construction at each layer based on learned embeddings rather than fixed spatial coordinates.

### Residual Connections
Deep residual skip connections between GCN layers mitigate the vanishing gradient problem, enabling the training of deeper graph networks without performance degradation.

---

## Dataset — ModelNet40

| Property | Value |
|---|---|
| Categories | 40 object classes |
| Training samples | 9,843 |
| Test samples | 2,468 |
| Points per cloud | 1,024 (standard) |
| Format | HDF5 (`.h5`) |

Also compatible with **ShapeNetPart** for part segmentation tasks.

---

## Project Structure

```
gml_project/
├── src/
│   ├── architecture/       # DeepGCN model definition
│   ├── data/               # ModelNet40 / ShapeNetPart data loaders
│   ├── main.py             # Training & evaluation entry point
│   └── config/             # Hyperparameter configs
└── log/                    # Experiment logs & checkpoints
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/tamer017/gml_project.git
cd gml_project

# Install dependencies
pip install torch torch-geometric torch-cluster torchmetrics torchsummary h5py numpy

# Download ModelNet40 dataset (HDF5 format)
# Place in src/data/modelnet40_ply_hdf5_2048/

# Train the model
python src/main.py --config configs/deepgcn.yaml
```

---

## Skills Demonstrated

- **Geometric Deep Learning:** Graph Neural Networks (GNNs), DeepGCN, EdgeConv, dynamic graph construction
- **3D Computer Vision:** Point cloud processing, ModelNet40 classification, 3D shape understanding
- **PyTorch Ecosystem:** PyTorch, PyTorch Geometric, TorchMetrics, `torch_cluster`
- **Data Engineering:** HDF5 I/O with `h5py`, point cloud augmentation, normalization pipelines
- **Research Engineering:** Hyperparameter management with `argparse`, experiment logging

---

## References

- [DeepGCN: Can GCNs Go as Deep as CNNs? — Li et al., ICCV 2019](https://arxiv.org/abs/1904.03751)
- [ModelNet — Princeton University](https://modelnet.cs.princeton.edu/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
