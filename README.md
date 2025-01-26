
# 3D Point Cloud Classification Using Graph Neural Networks

This project focuses on classifying dense 3D point clouds using Graph Neural Networks (GNNs). We leverage the **ModelNet40** dataset, which contains 12,311 CAD models across 40 object categories, to benchmark our models. The project explores graph-based approaches, including **EdgeConv** layers and **Graph Transformers**, and compares them with traditional methods like **PointNet**.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Background](#background)
   - [Point Clouds](#point-clouds)
   - [ModelNet40 Dataset](#modelnet40-dataset)
3. [Methods](#methods)
   - [EdgeConv Layer](#edgeconv-layer)
   - [Graph Transformer](#graph-transformer)
   - [Skip Connections](#skip-connections)
   - [Optimizer and Scheduler](#optimizer-and-scheduler)
   - [Data Augmentation](#data-augmentation)
4. [Results](#results)
   - [Our Results](#our-results)
   - [Comparison with Other Methods](#comparison-with-other-methods)
5. [Fine-Tuning on ShapeNetPart](#fine-tuning-on-shapenetpart)
6. [How to Run](#how-to-run)
   - [Setup](#setup)
   - [Training and Testing](#training-and-testing)
   - [Options in `train_main.sh`](#options-in-train_mainsh)
7. [AI Card](#ai-card)
8. [References](#references)
9. [Contact](#contact)

---

## Introduction
This project aims to classify 3D point clouds using Graph Neural Networks (GNNs). Point clouds are a common representation of 3D data, and their irregular structure makes them challenging to process with traditional deep learning methods. We explore graph-based approaches to capture the local and global structures of point clouds effectively.

---

## Background

### Point Clouds
Point clouds are collections of points in 3D space, often representing the surface of objects. They are widely used in applications like autonomous driving, robotics, and 3D modeling. Point clouds are typically collected using LiDAR sensors or generated from CAD models.

### ModelNet40 Dataset
The **ModelNet40** dataset is a widely used benchmark for 3D object classification. It contains 12,311 CAD models across 40 categories, such as chairs, tables, and cars. The dataset is split into:
- **Training set**: 9,843 models
- **Test set**: 2,485 models

---

## Methods

### EdgeConv Layer
The **EdgeConv** layer is a graph convolution operation that captures local geometric structures by constructing a k-Nearest Neighbors (kNN) graph. It dynamically updates edge features based on the relationships between neighboring points.

### Graph Transformer
We replace some **EdgeConv** layers with **Graph Transformers** to capture long-range dependencies in the point cloud. Transformers are effective in modeling global relationships, which complements the local features extracted by EdgeConv.

### Skip Connections
Skip connections are used to improve gradient flow and feature aggregation across different layers. They help in preserving fine-grained details and preventing information loss.

### Optimizer and Scheduler
We use the **AdamW** optimizer with a **cosine annealing learning rate scheduler** to stabilize training and improve convergence.

### Data Augmentation
To improve generalization, we apply data augmentation techniques such as random rotation, scaling, and jittering to the point clouds during training.

---

## Results

### Our Results
- **Model 1**: Repeated EdgeConv + kNN with skip connections. Accuracy: **93.27%**
- **Model 2**: EdgeConv replaced with Graph Transformers. Accuracy: **92.83%**

### Comparison with Other Methods
We compare our results with both graph-based and non-graph-based methods on the ModelNet40 dataset:

| Method Type       | Method Name       | Accuracy |
|-------------------|-------------------|----------|
| Graph-Based        | DGCNN             | 92.9%    |
| Graph-Based        | Ours (EdgeConv)   | 93.27%   |
| Non-Graph-Based    | PointNet          | 89.2%    |
| Non-Graph-Based    | PointNet++        | 90.7%    |

---

## Fine-Tuning on ShapeNetPart
To further evaluate our model's generalization, we fine-tune it on the **ShapeNetPart** dataset, which contains part-level annotations for 3D objects. This helps in understanding how well our model performs on more granular tasks.

---

## How to Run

### Setup
To set up the environment, run the following command:

```bash
source gml_env_install.sh
```

This script will install all the necessary dependencies, including PyTorch, PyTorch Geometric, and other required libraries. It will also set up the CUDA paths and create a Conda environment named `gml`.

### Training and Testing
To train or test the model, use the following command:

```bash
# For training
python main.py --phase train --multi_gpus

# For testing
python main.py --phase test --pretrained_model path/to/pretrained/model.pth
```

You can customize the training and testing process using various options available in the `train_main.sh` script.

---

## Options in `train_main.sh`

The `train_main.sh` script provides several options for customizing the training and testing process. Below is a list of the most important options and their explanations:

### Base Options
- **`--phase`**: Specifies the phase of the process. Use `train` for training and `test` for testing. Default: `train`.
- **`--exp_name`**: Name of the experiment. This will be used to create a directory for logs and checkpoints. Default: `''`.
- **`--use_cpu`**: If set, the model will run on the CPU instead of the GPU. Default: `False`.
- **`--root_dir`**: Root directory where experiment results, checkpoints, and logs will be saved. Default: `log`.

### Dataset Options
- **`--data_dir`**: Directory where the dataset is stored or will be downloaded. Default: `Dataset`.
- **`--dataset`**: Dataset to use. Options: `ModelNet40` or `ShapeNetPart`. Default: `ShapeNetPart`.
- **`--num_points`**: Number of points to use from each point cloud. Default: `2048`.
- **`--in_channels`**: Dimension of the input features. Default: `3` (x, y, z coordinates).

### Training Options
- **`--batch_size`**: Mini-batch size for training. Default: `32`.
- **`--epochs`**: Number of epochs to train. Default: `400`.
- **`--use_sgd`**: If set, SGD optimizer will be used instead of AdamW. Default: `False`.
- **`--weight_decay`**: L2 regularization (weight decay) for the optimizer. Default: `1e-4`.
- **`--lr`**: Learning rate. Default: `1e-3`.
- **`--seed`**: Random seed for reproducibility. Default: `1`.
- **`--multi_gpus`**: If set, the model will use multiple GPUs for training. Default: `False`.

### Testing Options
- **`--test_batch_size`**: Mini-batch size for testing. Default: `50`.
- **`--pretrained_model`**: Path to a pretrained model for testing or fine-tuning. Default: `''`.

### Model Options
- **`--k`**: Number of nearest neighbors for graph construction. Default: `15`.
- **`--block`**: Type of graph backbone block. Options: `res`, `plain`, `dense`. Default: `res`.
- **`--conv`**: Type of graph convolution layer. Options: `edge`, `mr`. Default: `edge`.
- **`--act`**: Activation function to use. Options: `relu`, `prelu`, `leakyrelu`. Default: `relu`.
- **`--norm`**: Normalization layer to use. Options: `batch`, `instance`. Default: `batch`.
- **`--bias`**: If set, bias will be added to the convolution layers. Default: `True`.
- **`--n_blocks`**: Number of basic blocks in the backbone. Default: `14`.
- **`--n_filters`**: Number of channels for deep features. Default: `64`.
- **`--emb_dims`**: Dimension of the embeddings. Default: `1024`.
- **`--dropout`**: Dropout rate. Default: `0.5`.
- **`--dynamic`**: If set, dynamic graph construction will be used. Default: `False`.

### Fine-Tuning Options
- **`--fine_tune`**: If set, the model will be fine-tuned on a new dataset. Default: `False`.
- **`--fine_tune_num_classes`**: Number of classes for fine-tuning. Default: `40`.

### Other Options
- **`--no_dilation`**: If set, dilated kNN will not be used. Default: `False`.
- **`--epsilon`**: Stochastic epsilon for graph construction. Default: `0.2`.
- **`--no_stochastic`**: If set, stochastic graph construction will not be used. Default: `False`.

---

## AI Card
This project is part of our research in Graph Machine Learning. We aim to explore the potential of GNNs in processing 3D data and improving classification accuracy.

---

## References
1. Princeton University, ModelNet40 Dataset. [Link](https://modelnet.cs.princeton.edu/)
2. Qi et al., PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. [Link](https://arxiv.org/abs/1612.00593)
3. Wang et al., Dynamic Graph CNN for Learning on Point Clouds. [Link](https://arxiv.org/abs/1801.07829)

---

## Contact
For any questions or feedback, feel free to reach out to us:
- **Ahmed Assy**: [ahmed.assy@stud.uni-goettingen.de](mailto:ahmed.assy@stud.uni-goettingen.de)
- **Mahmoud Abdellatif**: [mahmoud.abdellahi@stud.uni-goettingen.de](mailto:mahmoud.abdellahi@stud.uni-goettingen.de)

