
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
This project focuses on **3D point cloud classification** using **Graph Neural Networks (GNNs)**. Point clouds, which represent 3D objects as sets of points, are challenging to process due to their irregular and unordered structure. To address this, we use graph-based methods like **EdgeConv** and **Graph Transformers** to capture both local and global relationships between points.

Our approach leverages the **ModelNet40 dataset**, a benchmark for 3D shape recognition, and achieves competitive results. We also explore fine-tuning on the **ShapeNetPart dataset** to evaluate generalization. Implemented with **PyTorch** and **PyTorch Geometric**, the project provides a modular framework for experimenting with GNN architectures and datasets.

---

## Background

### Point Clouds
Point clouds are collections of data points in a three-dimensional (3D) space, typically representing the surface of an object or environment. Each point in a point cloud is defined by its coordinates $$(x, y, z)$$ in 3D space and may also include additional attributes such as color, intensity, or normal vectors. Point clouds are commonly generated using 3D scanning technologies, such as **LiDAR (Light Detection and Ranging)**, **structured light scanners**, or **photogrammetry**. LiDAR systems emit laser pulses and measure the time it takes for the light to reflect back, creating precise distance measurements that form the point cloud. Structured light scanners project a pattern of light onto an object and use cameras to capture the distortions in the pattern, which are then converted into 3D points. Photogrammetry, on the other hand, uses multiple 2D images taken from different angles to reconstruct the 3D geometry of an object or scene. Point clouds are widely used in applications such as autonomous driving, robotics, augmented reality, and 3D modeling, as they provide a detailed and accurate representation of real-world environments.
![image](https://github.com/user-attachments/assets/ff771566-3dd2-4827-9017-4ee0b46b5877)


### ModelNet40 Dataset
The **ModelNet40** dataset is a widely used benchmark for 3D shape recognition and classification. It consists of 12,311 3D models across 40 common object categories, such as tables, chairs, airplanes, and cars. Each model is represented as a point cloud, making it ideal for training and evaluating machine learning models for 3D shape analysis tasks like classification, segmentation, and retrieval.

This project provides a Python script (`dataset.py`) that handles **downloading**, **loading**, and **processing** the ModelNet40 dataset. The script also includes a PyTorch-compatible dataset loader for seamless integration into machine learning workflows.

#### About the Dataset
- **Number of Models**: 12,311
- **Number of Classes**: 40
- **Training Set**: 9,843 models
- **Test Set**: 2,468 models
- **Point Cloud Size**: 2,048 points per model

##### Who Collected It?
The ModelNet40 dataset was created by researchers at **Princeton University**, including **Zhirong Wu**, **Shuran Song**, **Aditya Khosla**, **Xiaoou Tang**, and **Jianxiong Xiao**.

##### How Was It Collected?
The dataset was generated from 3D CAD models, which were collected from online repositories and manually aligned to ensure consistency. Each model was converted into a point cloud representation, making it suitable for 3D deep learning tasks.

##### When Was It Created?
The dataset was introduced in 2015 as part of the **3D ShapeNets** project.

##### Why Was It Developed?
ModelNet40 was created to address the lack of large-scale, well-organized 3D shape datasets for training and evaluating machine learning models. It has since become a standard benchmark for 3D shape analysis tasks, enabling advancements in 3D object recognition, segmentation, and retrieval.


---

## Methods

### EdgeConv Layer
The **EdgeConv** layer is a graph convolution operation that captures local geometric structures by constructing a k-Nearest Neighbors (kNN) graph. It dynamically updates edge features based on the relationships between neighboring points, making it particularly well-suited for processing point cloud data.

Given a point cloud $$X = \{x_1, x_2, ..., x_n\}$$, where $$x_i \in \mathbb{R}^d$$, EdgeConv constructs a graph by connecting each point $$x_i$$ to its $$k$$ nearest neighbors. The feature of an edge $$(x_i, x_j)$$ is computed as:

$$h_{ij} = h_\Theta(x_i, x_j - x_i)$$

where $$h_\Theta$$ is a learnable function (e.g., a multi-layer perceptron), and $$x_j - x_i$$ represents the relative position of the neighbor $$x_j$$ with respect to $$x_i$$. The output feature for $$x_i$$ is then aggregated as:

$$x_i' = \max_{j \in \mathcal{N}(i)} h_{ij}$$

where $$\mathcal{N}(i)$$ is the set of neighbors of $$x_i$$, and $$\max$$ is a symmetric aggregation function (e.g., max-pooling).

#### Why EdgeConv is Perfect for Point Clouds
Point clouds are inherently irregular and unordered, making traditional convolutional operations unsuitable. EdgeConv addresses this by:
1. **Capturing Local Geometry**: By constructing a kNN graph, EdgeConv explicitly models the local geometric relationships between points, which is crucial for understanding the structure of point clouds.
2. **Dynamic Feature Learning**: The dynamic nature of EdgeConv allows it to adapt to the local structure of the point cloud, enabling the network to learn features that are invariant to transformations and robust to noise.
3. **Permutation Invariance**: The aggregation function (e.g., max-pooling) ensures that the operation is invariant to the order of points, a key requirement for processing point clouds.

#### The Role of Dynamic kNN in Edge Reconstruction
The dynamic kNN graph is a critical component of EdgeConv. Unlike static graphs, the kNN graph is recomputed at each layer based on the feature space, allowing the network to:
1. **Adapt to Hierarchical Features**: As the network learns higher-level features, the kNN graph evolves to reflect the changing relationships between points, improving the representation of the point cloud.
2. **Enhance Edge Reconstruction**: By dynamically updating the neighborhood relationships, EdgeConv can better reconstruct edges and surfaces in the point cloud, leading to more accurate geometric representations.
3. **Improve Robustness**: The dynamic nature of kNN makes the network more robust to variations in point density and distribution, which are common challenges in point cloud data.


### Comparison Between EdgeConv and Traditional Graph Convolution
EdgeConv and traditional graph convolution (e.g., Graph Convolutional Networks, GCNs) are both methods for processing graph-structured data, but they differ significantly in their approach, especially when applied to point clouds. Below is a detailed comparison:

#### 1. **Input Representation**
- **EdgeConv**:
  - Operates on point clouds, where each point is represented as a node in a graph.
  - Constructs a **dynamic k-Nearest Neighbors (kNN) graph** to define edges between points based on their feature space or spatial proximity.
  - Focuses on capturing **local geometric structures** by explicitly modeling edge features between neighboring points.

- **Traditional Graph Convolution (GCN)**:
  - Operates on general graphs, where nodes and edges are predefined and fixed.
  - Assumes a **static graph structure**, meaning the connectivity between nodes does not change during the forward pass.
  - Does not explicitly model edge features; instead, it aggregates information from neighboring nodes based on the graph's adjacency matrix.

#### 2. **Feature Aggregation**
- **EdgeConv**:
  - Computes edge features between a central node and its neighbors using a learnable function (e.g., a Multi-Layer Perceptron (MLP)).
  - Aggregates edge features using a symmetric function like **max-pooling** or **summation** to update node features.
  - Captures **relative geometric relationships** between points by incorporating the difference in features (e.g., $$x_j - x_i$$) as part of the edge feature computation.

- **Traditional Graph Convolution (GCN)**:
  - Aggregates features from neighboring nodes using a weighted sum, where the weights are determined by the graph's adjacency matrix (often normalized).
  - Does not explicitly model edge features or relative relationships between nodes.
  - Typically uses a simpler aggregation mechanism, such as averaging or summing neighbor features.

#### 3. **Dynamic vs. Static Graph Structure**
- **EdgeConv**:
  - Uses a **dynamic kNN graph**, where the neighborhood of each node is recomputed at each layer based on the current feature space.
  - This allows the graph structure to adapt to the hierarchical features learned by the network, making it more flexible and powerful for tasks like point cloud processing.

- **Traditional Graph Convolution (GCN)**:
  - Uses a **static graph structure**, where the connectivity between nodes is fixed and predefined before training.
  - The graph structure does not change during the forward pass, which can limit its ability to adapt to complex or evolving data structures.

#### 4. **Suitability for Point Clouds**
- **EdgeConv**:
  - Specifically designed for point clouds, which are irregular, unordered, and lack a fixed structure.
  - Captures local geometric relationships effectively, making it ideal for tasks like 3D shape classification, segmentation, and reconstruction.
  - The dynamic kNN graph allows it to adapt to the varying density and distribution of points in a point cloud.

- **Traditional Graph Convolution (GCN)**:
  - Less suitable for point clouds because it assumes a fixed graph structure, which does not align well with the irregular and unordered nature of point clouds.
  - Struggles to capture local geometric structures effectively, as it does not explicitly model edge features or relative relationships.


#### Summary of Key Differences
| **Aspect**               | **EdgeConv**                                                                 | **Traditional Graph Convolution (GCN)**                          |
|--------------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------|
| **Graph Structure**       | Dynamic kNN graph, recomputed at each layer                                 | Static graph, fixed connectivity                                 |
| **Edge Features**         | Explicitly modeled using a learnable function                              | Not explicitly modeled                                           |
| **Aggregation**           | Symmetric aggregation (e.g., max-pooling) of edge features                 | Weighted sum of node features based on adjacency matrix          |
| **Suitability for Point Clouds** | Highly suitable, captures local geometry effectively                  | Less suitable, struggles with irregular and unordered data       |
| **Flexibility**           | Adapts to hierarchical features and varying point densities                | Limited flexibility due to static graph structure                |

---
### Graph Transformer
We replace some **EdgeConv** layers with **Graph Transformers** to capture long-range dependencies in the point cloud. Transformers are effective in modeling global relationships, which complements the local features extracted by EdgeConv.

In the Graph Transformer, the self-attention mechanism is applied to the graph structure. For each node $$x_i$$, the attention weights $$\alpha_{ij}$$ are computed as:

$$\alpha_{ij} = \text{softmax}\left(\frac{(W_Q x_i)^T (W_K x_j)}{\sqrt{d_k}}\right)$$

where $$W_Q$$ and $$W_K$$ are learnable weight matrices, and $$d_k$$ is the dimension of the key vectors. The output feature for $$x_i$$ is then computed as:

$$x_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} (W_V x_j)$$

where $$W_V$$ is another learnable weight matrix.

---

### Skip Connections
Skip connections are used to improve gradient flow and feature aggregation across different layers. They help in preserving fine-grained details and preventing information loss. In our architecture, skip connections are added between the input and output of each **EdgeConv** or **Graph Transformer** block. The output of a block with skip connections is computed as:

$$x_i' = x_i + \text{Block}(x_i)$$

where $$\text{Block}$$ represents either an EdgeConv or Graph Transformer layer.

---

### Optimizer and Scheduler

#### **AdamW Optimizer**
The **AdamW** optimizer is a widely used optimization algorithm in modern deep learning tasks. It is a variant of the Adam optimizer but incorporates **decoupled weight decay regularization**, which improves training stability and prevents undesirable side effects on weight updates.

The update rule for AdamW is:

\[
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)
\]

Here are the components:
- $$\theta_t$$: The parameters of the model at step $$t$$.
- $$\eta$$: The learning rate, which determines the step size for updates.
- $$\hat{m}_t$$: The bias-corrected estimate of the first moment (mean of gradients).
- $$\hat{v}_t$$: The bias-corrected estimate of the second moment (uncentered variance of gradients).
- $$\epsilon$$: A small constant added for numerical stability, preventing division by zero.
- $$\lambda$$: The weight decay parameter, which controls the strength of regularization.

**Key Differences Between AdamW and Adam:**
1. **Weight Decay Regularization:**
   - In the standard Adam optimizer, weight decay is implemented as **L2 regularization**, which modifies the gradients directly during backpropagation.
   - In AdamW, weight decay is decoupled from the optimization step. Instead, it is applied as a separate term ($$\lambda \theta_t$$), directly scaling the parameters. This prevents interference with the adaptive moment estimation process, improving convergence and generalization.

2. **Impact on Optimization:**
   - Decoupling weight decay allows AdamW to handle large-scale models better, particularly in tasks where generalization is critical (e.g., NLP and vision).



#### **Learning Rate Scheduler: Cosine Annealing**
Cosine annealing is a learning rate schedule designed to reduce the learning rate gradually, following the shape of a cosine curve. The formula is:

\[
\eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t}{T} \pi\right)\right)
\]

**Parameters:**
- $$\eta_{\text{min}}$$: The minimum learning rate.
- $$\eta_{\text{max}}$$: The initial (maximum) learning rate.
- $$T$$: The total number of training steps.
- $$t$$: The current training step.

**Explanation:**
- At the beginning of training ($$t = 0$$), the learning rate starts at $$\eta_{\text{max}}$$.
- As $$t$$ increases, the learning rate decreases following a cosine curve, reaching $$\eta_{\text{min}}$$ at the end of training ($$t = T$$).
- The gradual reduction helps the model converge to a better minimum by taking smaller and smaller steps as training progresses.

**Comparison with Static Learning Rates:**
1. **Static Learning Rates:**
   - Static learning rates remain constant throughout training, which can lead to poor convergence. If the learning rate is too high, the optimizer may overshoot the optimal solution; if too low, it may converge too slowly or to a suboptimal point.
2. **Cosine Annealing:**
   - Provides a dynamic learning rate that adapts over time, leading to smoother convergence and better performance.



---
### Data Augmentation
To improve generalization, we apply data augmentation techniques such as random rotation, scaling, and jittering to the point clouds during training. These augmentations help the model become invariant to transformations and improve robustness.

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

1. **3D ShapeNets Paper**  
   This is the original paper that introduced the ModelNet40 dataset. It describes how the dataset was collected, processed, and used for 3D shape analysis tasks.  
   [Paper Link](https://arxiv.org/abs/1406.5670)

2. **ModelNet40 Dataset Website**  
   The official website for the ModelNet40 dataset, hosted by Princeton University, provides additional details and resources.  
   [Princeton University, ModelNet40 Dataset](https://modelnet.cs.princeton.edu/)

3. Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., & Solomon, J. M. (2019). **Dynamic Graph CNN for Learning on Point Clouds**. *ACM Transactions on Graphics (TOG)*, 38(5), 1-12. [DOI:10.1145/3326362](https://doi.org/10.1145/3326362)

4. Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 652-660. [DOI:10.1109/CVPR.2017.16](https://doi.org/10.1109/CVPR.2017.16)

5. Guo, Y., Wang, H., Hu, Q., Liu, H., Liu, L., & Bennamoun, M. (2020). **Deep Learning for 3D Point Clouds: A Survey**. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 43(12), 4338-4364. [DOI:10.1109/TPAMI.2020.3005434](https://doi.org/10.1109/TPAMI.2020.3005434)

6. Loshchilov, Ilya, and Hutter, Frank. *"SGDR: Stochastic Gradient Descent with Warm Restarts."* arXiv preprint arXiv:1608.03983 (2016). [Link](https://arxiv.org/abs/1608.03983)

7. Loshchilov, Ilya, and Hutter, Frank. *"Decoupled Weight Decay Regularization."* arXiv preprint arXiv:1711.05101 (2017). [Link](https://arxiv.org/abs/1711.05101)



---

## Contact
For any questions or feedback, feel free to reach out to us:
- **Ahmed Assy**: [ahmed.assy@stud.uni-goettingen.de](mailto:ahmed.assy@stud.uni-goettingen.de)
- **Mahmoud Abdellahi**: [mahmoud.abdellahi@stud.uni-goettingen.de](mailto:mahmoud.abdellahi@stud.uni-goettingen.de)

