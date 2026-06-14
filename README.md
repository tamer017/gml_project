# 3D Point Cloud Classification Using Graph Neural Networks

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-orange.svg)](https://pytorch.org/)
[![Final](https://img.shields.io/badge/Status-Final-purple.svg)](https://img.shields.io/badge/Status-Final-blue.svg)
[![AI-Usage Card](https://img.shields.io/badge/AI_Usage_Card-pdf-blue.svg)](ai-usage-card.pdf)

---

## Table of Contents
1. [Introduction](#introduction)
2. [Background](#background)
   - [Point Clouds](#point-clouds)
   - [ModelNet40 Dataset](#modelnet40-dataset)
3. [Methods](#methods)
   - [EdgeConv Layer](#edgeconv-layer)
   - [TransformerConv](#transformerConv)
   - [Skip Connections](#skip-connections)
   - [Optimizer and Scheduler](#optimizer-and-scheduler)
   - [Data Augmentation](#data-augmentation)
4. [Experiments](#experiments)
5. [Results](#results)
   - [Our Results](#our-results)
   - [Comparison with Other Methods](#comparison-with-other-methods)
6. [Fine-Tuning on ShapeNetPart](#fine-tuning-on-shapenetpart)
7. [How to Run](#how-to-run)
   - [Setup](#setup)
   - [Training and Testing](#training-and-testing)
   - [Options in `train_main.sh`](#options-in-train_mainsh)
8. [AI Card](#ai-card)
9. [References](#references)
10. [Contact](#contact)

---

## Introduction
This project focuses on **3D point cloud classification** using **Graph Neural Networks (GNNs)**. Point clouds, which represent 3D objects as sets of points, are challenging to process due to their irregular and unordered structure. To address this, we use graph-based methods like **EdgeConv** and **TransformerConvs** to capture both local and global relationships between points.

Our approach leverages the **ModelNet40 dataset**, a benchmark for 3D shape recognition, and achieves competitive results. We also explore fine-tuning on the **ShapeNetPart dataset** to evaluate generalization. Implemented with **PyTorch** and **PyTorch Geometric**, the project provides a modular framework for experimenting with GNN architectures and datasets.

---

## Background

### Point Clouds
Point clouds are collections of data points in a three-dimensional (3D) space, typically representing the surface of an object or environment. Each point in a point cloud is defined by its coordinates $(x, y, z)$ in 3D space and may also include additional attributes such as color, intensity, or normal vectors. Point clouds are commonly generated using 3D scanning technologies, such as **LiDAR**, **structured light scanners**, or **photogrammetry**.

![image](https://github.com/user-attachments/assets/ff771566-3dd2-4827-9017-4ee0b46b5877)

### ModelNet40 Dataset
The **ModelNet40** dataset is a widely used benchmark for 3D shape recognition and classification. It consists of 12,311 3D models across 40 common object categories, such as tables, chairs, airplanes, and cars.

#### About the Dataset
- **Number of Models**: 12,311
- **Number of Classes**: 40
- **Training Set**: 9,843 models
- **Test Set**: 2,468 models (~20%)
- **Point Cloud Size**: 2,048 points per model

![image](https://github.com/user-attachments/assets/02e30678-7da8-439b-914a-0760e79413f5)

![image](https://github.com/user-attachments/assets/251d2710-bb12-4db9-9aae-bf8e025da824)

##### Who Collected It?
Created by researchers at **Princeton University** (Zhirong Wu, Shuran Song, Aditya Khosla, Xiaoou Tang, Jianxiong Xiao). Introduced in 2015 as part of the **3D ShapeNets** project.

---

## Methods

### EdgeConv Layer
The **EdgeConv** layer captures local geometric structures by constructing a k-Nearest Neighbors (kNN) graph. It dynamically updates edge features based on the relationships between neighboring points.

Given a point cloud $X = \{x_1, x_2, ..., x_n\}$, EdgeConv constructs a graph by connecting each point $x_i$ to its $k$ nearest neighbors. The edge feature is computed as:

$$h_{ij} = h_\Theta(x_i, x_j - x_i)$$

The output feature for $x_i$ is then aggregated as:

$$x_i' = \max_{j \in \mathcal{N}(i)} h_{ij}$$

#### Why EdgeConv for Point Clouds
- **Captures Local Geometry**: Explicitly models local geometric relationships
- **Dynamic Feature Learning**: Adapts to local structure at each layer
- **Permutation Invariance**: Max-pooling ensures order invariance

#### EdgeConv vs. Traditional GCN

| **Aspect** | **EdgeConv** | **Traditional GCN** |
|---|---|---|
| **Graph Structure** | Dynamic kNN, recomputed each layer | Static, fixed connectivity |
| **Edge Features** | Explicitly modeled via learnable MLP | Not explicitly modeled |
| **Aggregation** | Max-pooling of edge features | Weighted sum based on adjacency |
| **Point Cloud Suitability** | Highly suitable | Less suitable |
| **Flexibility** | Adapts to varying densities | Limited by static structure |

---

### TransformerConv

The **TransformerConv** layer integrates Transformer attention mechanisms with graph convolution:

![Pasted_image-removebg-preview](https://github.com/user-attachments/assets/8440b4ae-27ba-4e0b-bfa1-c7b81bc9c74e)

Attention coefficient:
$$\alpha_{i,j} = \text{softmax} \left( \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)} {\sqrt{d}} \right)$$

**Key benefits:** Multi-head attention, global context integration, permutation invariance.

---

### Skip Connections

$$x_i' = x_i + \text{Block}(x_i)$$

Residual connections prevent vanishing gradients in deep architectures and preserve fine-grained features.

![image](https://github.com/user-attachments/assets/e1362728-f367-4fa1-82ae-4af9be4dd039)

---

### Optimizer and Scheduler

**AdamW** with decoupled weight decay:
$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

**Cosine Annealing** learning rate:
$$\eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t}{T} \pi\right)\right)$$

---

### Data Augmentation

1. **Translation (Scaling & Shifting)**: Scale `[2/3, 3/2]`, shift `[-0.2, 0.2]`
2. **Shuffling**: Random reordering of points for permutation invariance
3. **Normalization**: Center around origin, scale to unit sphere

---

## Experiments

![concate](https://github.com/user-attachments/assets/d0523993-d23b-4349-ab7b-1f557bfd394c)

<details>
<summary>Experiment 1 — EdgeConv with Skip Connections</summary>

**Hyperparameters**: Batch 32, Epochs 400, LR 0.001, AdamW wd=0.0001, Dropout 0.5, k=15, 14 blocks, 64 filters

</details>

<details>
<summary>Experiment 2 — TransformerConv with Dense Connections</summary>

**Hyperparameters**: Batch 32, Epochs 200, LR 0.0001, AdamW wd=0.0001, Dropout 0.5, k=15, 7 blocks, 256 filters

</details>

---

## Results

### ModelNet40 Classification

![image](https://github.com/user-attachments/assets/8d1469db-25d0-47d9-8526-0a0bc6ceb6b7)

### Benchmark Comparison

| **Model** | **Overall Accuracy (%)** | **Category** |
|---|---|---|
| RS-CNN | 93.6 | Graph-Based |
| MLVCNN | 94.16 | Non-Graph-Based |
| MHBN | 94.7 | Non-Graph-Based |
| RotationNet | 97.37 | Non-Graph-Based |
| LDGCNN | 92.9 | Graph-Based |
| **EdgeConv (Ours)** | **93.27** | Graph-Based |
| **TransformerConv (Ours)** | **92.75** | Graph-Based |

### EdgeConv Result
- **Test Overall Accuracy**: 93.27% | **Avg Class Accuracy**: 90.09%
- **Parameters**: 2.2M | **Inference**: 7.55 ms/point cloud

### TransformerConv Result
- **Test Overall Accuracy**: 92.75% | **Avg Class Accuracy**: 89.36%
- **Parameters**: 15.8M | **Inference**: 8.34 ms/point cloud

> EdgeConv is 7.2× smaller than TransformerConv (2.2M vs 15.8M params) while being 10% faster

### Loss Graphs

#### TransformerConv
![image](https://github.com/user-attachments/assets/74ca964d-d026-4800-919c-1c8f8d38c666)

#### EdgeConv
![image](https://github.com/user-attachments/assets/4b1c40e6-220c-4d5e-ae8e-3641a32ed3a4)

### Cross-Dataset Generalization on ShapeNet
Fine-tuned pretrained models achieved **97.6% accuracy in 8 epochs** with rapid convergence.

---

## How to Run

### Setup
```bash
source gml_env_install.sh
```

### Training and Testing

```bash
# Train EdgeConv
source run_main.sh --phase train --multi_gpus --block res --n_blocks 14 --data_dir /Dataset --n_filter 64 --batch_size 32 --conv edge

# Train TransformerConv
source run_main.sh --phase train --n_blocks 7 --block dense --data_dir /Dataset --n_filter 256 --batch_size 32 --conv trans --dynamic True --multi_gpu

# Test EdgeConv
source run_main.sh --phase test --multi_gpus --block res --n_blocks 14 --data_dir /Dataset --n_filter 64 --conv edge --batch_size 32 --pretrained_model path/to/model.pth
```

### Model Weights
[Download Pre-trained Model Weights](https://drive.google.com/drive/folders/1hm0q7_I8cLXCDSXgNoBQ-mlyvRxIgqOz)

---

## AI Card

AI aided the development of this project. See our AI-Usage card [here](ai-usage-card.pdf) (generated from [https://ai-cards.org/](https://ai-cards.org/)).

---

## References

1. [3D ShapeNets Paper](https://arxiv.org/abs/1406.5670)
2. [ModelNet40 Dataset — Princeton University](https://modelnet.cs.princeton.edu/)
3. Wang et al. (2019). Dynamic Graph CNN for Learning on Point Clouds. [DOI:10.1145/3326362](https://doi.org/10.1145/3326362)
4. Qi et al. (2017). PointNet. [DOI:10.1109/CVPR.2017.16](https://doi.org/10.1109/CVPR.2017.16)
5. Guo et al. (2020). Deep Learning for 3D Point Clouds: A Survey. [DOI:10.1109/TPAMI.2020.3005434](https://doi.org/10.1109/TPAMI.2020.3005434)
6. Loshchilov & Hutter. SGDR. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
7. Loshchilov & Hutter. Decoupled Weight Decay. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
8. Shi et al. Masked Label Prediction. [arXiv:2009.03509](https://arxiv.org/abs/2009.03509)
9. Zhao et al. Point Transformer. [arXiv:2012.09164](https://arxiv.org/abs/2012.09164)
10. Chen et al. PointGPT. [arXiv:2305.11487](https://arxiv.org/abs/2305.11487)
11. Qi et al. ShapeLLM. [arXiv:2402.17766](https://arxiv.org/abs/2402.17766)
12. Ma et al. PointMLP. [arXiv:2202.07123](https://arxiv.org/abs/2202.07123)
13. Li et al. DeepGCNs. [arXiv:1904.03751](https://arxiv.org/abs/1904.03751)
14. Qi et al. PointNet++. [arXiv:1706.02413](https://arxiv.org/abs/1706.02413)

---

## Contact

- **Ahmed Assy**: [ahmed.assy@stud.uni-goettingen.de](mailto:ahmed.assy@stud.uni-goettingen.de)
- **Mahmoud Abdellahi**: [mahmoud.abdellahi@stud.uni-goettingen.de](mailto:mahmoud.abdellahi@stud.uni-goettingen.de)
