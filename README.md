# GML Project — 3D Point Cloud Classification with DeepGCN

> **Geometric Deep Learning system classifying 3D point clouds into 40 categories using dynamic graph construction and dilated EdgeConv — no voxelization required.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-latest-orange.svg)](https://pyg.org/)

---

## Overview

This project applies **Geometric Deep Learning** to the task of 3D object classification and graph-level tasks using the **ModelNet40** benchmark. Three GNN architectures are systematically compared: GCN, GraphSAGE, and Graph Attention Networks (GAT).

The core innovation is operating **directly on raw XYZ coordinates** of unordered 3D point clouds without voxelization or 2D projection — preserving full geometric fidelity.

---

## Architecture — DeepGCN with Dilated EdgeConv

```
Raw Point Cloud (N × 3 XYZ)
        │
  Dynamic k-NN Graph
  (re-built each layer)
        │
  Dilated EdgeConv Layer 1
  (expanded receptive field)
        │
  Dilated EdgeConv Layer 2
        │
  Global Pooling (Max + Avg)
        │
  MLP Classifier → 40 classes
```

**Key design decisions:**
- **Dynamic graph re-construction** at each layer based on updated feature embeddings — captures topology-aware representations
- **Dilated KNN graphs** expand receptive field without increasing k, analogous to dilated convolutions in 2D CNNs
- **No voxelization** — preserves full geometric resolution

---

## Models Benchmarked

| Model | Node Classification | Link Prediction AUC |
|---|---|---|
| **GCN** | 81% | — |
| **GraphSAGE** | 82% | — |
| **GAT** | **83%** | **90%** |

GAT's attention mechanism provides the best performance by adaptively weighting neighbor contributions based on feature similarity.

---

## Dataset — ModelNet40

- **40 object categories**: chair, table, airplane, lamp, bathtub, etc.
- **12,311 CAD models** (9,843 train / 2,468 test)
- Each object represented as a point cloud sampled from mesh surfaces
- Features: XYZ coordinates + surface normals

---

## Installation

```bash
git clone https://github.com/tamer017/gml_project.git
cd gml_project
pip install torch torch_geometric h5py numpy
```

---

## Skills & Concepts

`Geometric Deep Learning` `Graph Neural Networks` `GCN` `GraphSAGE` `GAT` `PyTorch Geometric` `3D Computer Vision` `Point Clouds` `EdgeConv` `ModelNet40` `Link Prediction`

---

## Author

**Ahmed Tamer Assy** — [GitHub](https://github.com/tamer017) | Machine Learning Researcher @ Volkswagen AG
