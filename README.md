
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
Point clouds are collections of data points in a three-dimensional (3D) space, typically representing the surface of an object or environment. Each point in a point cloud is defined by its coordinates $$(x, y, z)$$ in 3D space and may also include additional attributes such as color, intensity, or normal vectors. Point clouds are commonly generated using 3D scanning technologies, such as **LiDAR (Light Detection and Ranging)**, **structured light scanners**, or **photogrammetry**. LiDAR systems emit laser pulses and measure the time it takes for the light to reflect back, creating precise distance measurements that form the point cloud. Structured light scanners project a pattern of light onto an object and use cameras to capture the distortions in the pattern, which are then converted into 3D points. Photogrammetry, on the other hand, uses multiple 2D images taken from different angles to reconstruct the 3D geometry of an object or scene. Point clouds are widely used in applications such as autonomous driving, robotics, augmented reality, and 3D modeling, as they provide a detailed and accurate representation of real-world environments.
![image](https://github.com/user-attachments/assets/ff771566-3dd2-4827-9017-4ee0b46b5877)


### ModelNet40 Dataset
The **ModelNet40** dataset is a widely used benchmark for 3D shape recognition and classification. It consists of 12,311 3D models across 40 common object categories, such as tables, chairs, airplanes, and cars. Each model is represented as a point cloud, making it ideal for training and evaluating machine learning models for 3D shape analysis tasks like classification, segmentation, and retrieval.

This project provides a Python script (`dataset.py`) that handles **downloading**, **loading**, and **processing** the ModelNet40 dataset. The script also includes a PyTorch-compatible dataset loader for seamless integration into machine learning workflows.

#### About the Dataset
- **Number of Models**: 12,311
- **Number of Classes**: 40
- **Training Set**: 9,843 models
- **Test Set**: 2,468 models (~20%)
- **Point Cloud Size**: 2,048 points per model
  
![image](https://github.com/user-attachments/assets/02e30678-7da8-439b-914a-0760e79413f5)

![image](https://github.com/user-attachments/assets/251d2710-bb12-4db9-9aae-bf8e025da824)




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



### **TransformerConv**

The **TransformerConv** layer, as implemented in [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/2.4.0/generated/torch_geometric.nn.conv.TransformerConv.html), integrates the attention mechanisms of Transformers with graph convolutional networks, making it particularly effective for processing non-Euclidean data structures like point clouds.

The operation of a **TransformerConv** layer is defined as:

![Pasted_image-removebg-preview](https://github.com/user-attachments/assets/8440b4ae-27ba-4e0b-bfa1-c7b81bc9c74e)

Where:

- $$\mathbf{x}^{\prime}_i$$: The updated feature of node $$i$$.
- $$\mathcal{N}(i)$$: The set of neighbors for node $$i$$.
- $$\mathbf{W}_1$$ and $$\mathbf{W}_2$$: Learnable weight matrices.
- $$\alpha_{i,j}$$: The attention coefficient between nodes $$i$$ and $$j$$, computed as:

$$\alpha_{i,j} = \text{softmax} \left( \frac{(\mathbf{W}_3\mathbf{x}_i)^{\top} (\mathbf{W}_4\mathbf{x}_j)} {\sqrt{d}} \right)$$

Here, $$d$$ represents the dimensionality of the key vectors, and $$\mathbf{W}_3$$ and $$\mathbf{W}_4$$ are additional learnable weight matrices.

**Key Features:**

- **Multi-Head Attention:** The layer supports multiple attention heads, allowing the model to capture various aspects of the local neighborhood.
- **Edge Features:** If edge features are present, the layer can incorporate them into the attention mechanism, enhancing its ability to model complex relationships.


#### **Why TransformerConv is Beneficial for Point Cloud Classification**

1. **Dynamic Attention:**
   - TransformerConv dynamically assigns weights to neighboring points, enabling the model to focus on the most relevant features within a point cloud's local structure.

2. **Global Context Integration:**
   - By leveraging self-attention, the layer captures both local and global contextual information, which is crucial for understanding complex 3D structures.

3. **Permutation Invariance:**
   - Point clouds are unordered sets of points. TransformerConv's design inherently handles this property, ensuring consistent performance regardless of point ordering.

#### **k-Nearest Neighbors (kNN) Layers**

In point cloud processing, constructing a meaningful graph structure is essential. We employ **k-Nearest Neighbors (kNN)** to build this graph by connecting each point to its $$k$$ closest neighbors based on Euclidean distance.

**Benefits of kNN in This Context:**

- **Preservation of Local Geometry:**
  - By connecting each point to its nearest neighbors, the local geometric relationships are maintained, which is vital for accurate feature learning.

- **Dynamic Graph Construction:**
  - The kNN approach allows the graph structure to adapt dynamically to the underlying data distribution, ensuring that the most relevant connections are established for each point.


#### **Integrating TransformerConv with kNN for Point Cloud Classification**

Our model follows these steps:

1. **Graph Construction:**
   - Utilize a kNN layer to construct a graph where each point is connected to its $$k$$ nearest neighbors.

2. **Feature Aggregation:**
   - Apply TransformerConv layers to aggregate features from neighboring points, with attention mechanisms assigning appropriate weights to each neighbor's contribution.

3. **Hierarchical Learning:**
   - Stack multiple TransformerConv layers to learn hierarchical representations, capturing both fine-grained local details and broader global structures.

---  

### Skip Connections

Skip connections are a critical architectural feature that enhance gradient flow and improve feature aggregation across different layers of deep networks. They address challenges like **vanishing gradients** in very deep models and help preserve fine-grained details while preventing information loss. 

In our architecture, skip connections are added between the input and output of each **EdgeConv** or **TransformerConv** block. The output of a block with skip connections is computed as:

$$x_i' = x_i + \text{Block}(x_i)$$ 

This form of skip connection, often referred to as a **residual connection**, enables the model to learn residual mappings instead of the complete transformation. This simplifies optimization and helps stabilize training.

#### **Why Residual Connections in Our Model?**
Our model architecture is notably deep, with a large number of stacked layers. While depth enables the network to learn more complex features, it also introduces the risk of the **vanishing gradient problem**, where gradients diminish as they are backpropagated through the network. This can lead to poor training dynamics and suboptimal performance. 

To address this, we replaced plain connections (i.e., simple skip connections that directly add inputs to outputs without transformation) with **residual and dense connections**, which are more effective for deep architectures:
- **Residual Connections:** Allow the model to propagate gradients more efficiently by adding the input directly to the output of a block. This ensures the gradient signal does not vanish, even in very deep networks.
- **Dense Connections:** In some cases, we experimented with connections that concatenate outputs from all preceding layers. This type of connection facilitates feature reuse and improves gradient flow further.

#### **Advantages of Residual and Dense Connections:**
1. **Improved Gradient Flow:** Residual connections mitigate the risk of vanishing gradients, enabling stable training for deep networks.
2. **Preservation of Information:** Features from earlier layers are preserved and aggregated across the network, preventing information loss and maintaining fine-grained details.
3. **Ease of Optimization:** By simplifying the learning objective to residual mappings, the network converges faster and more effectively.
4. **Enhanced Feature Reuse:** Dense connections promote the sharing of features across multiple layers, further enriching the learned representations.

By incorporating residual and dense connections into our architecture, we were able to overcome the challenges posed by our deep model design and achieve better convergence, stability, and performance.
![image](https://github.com/user-attachments/assets/e1362728-f367-4fa1-82ae-4af9be4dd039)


---

### Optimizer and Scheduler

#### **AdamW Optimizer**
The **AdamW** optimizer is a widely used optimization algorithm in modern deep learning tasks. It is a variant of the Adam optimizer but incorporates **decoupled weight decay regularization**, which improves training stability and prevents undesirable side effects on weight updates.

The update rule for AdamW is:

$$\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

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

$$\eta_t = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t}{T} \pi\right)\right)$$

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

Data augmentation is a crucial technique in machine learning, especially when working with limited datasets. It helps to increase the diversity of the training data, which can improve the generalization and robustness of the model. In the context of point cloud data, such as the **ModelNet40** and **ShapeNetPart** datasets, several augmentation techniques are applied to enhance the training process. Below is a brief overview of the simplified augmentation methods used in the updated code:

#### 1. **Translation (Shifting and Scaling)**
   - **Purpose**: To make the model invariant to the position and scale of the point cloud.
   - **Method**: 
     - **Scaling**: Each point in the point cloud is scaled by a random factor along the x, y, and z axes. The scaling factors are sampled uniformly from the range `[2/3, 3/2]`.
     - **Shifting**: The point cloud is shifted by a random offset along the x, y, and z axes. The shift values are sampled uniformly from the range `[-0.2, 0.2]`.
   - **Effect**: This augmentation helps the model to recognize objects regardless of their size and position in the 3D space.

#### 2. **Shuffling**
   - **Purpose**: To make the model invariant to the order of points in the point cloud.
   - **Method**: The points in the point cloud are randomly shuffled.
   - **Effect**: Since the order of points in a point cloud does not carry any meaningful information, shuffling ensures that the model does not rely on the specific order of points.

#### 3. **Normalization**
   - **Purpose**: To standardize the point cloud data, making it easier for the model to learn.
   - **Method**: The point cloud is centered around the origin by subtracting the centroid (mean) of the point cloud. Then, the point cloud is scaled so that the furthest point from the origin lies on the surface of a unit sphere.
   - **Effect**: Normalization ensures that all point clouds have a consistent scale and position, which can improve the convergence and stability of the training process.


---
## Experiments
![concate](https://github.com/user-attachments/assets/d0523993-d23b-4349-ab7b-1f557bfd394c)

<details>

<summary>Experiment 1</summary>

#### **Training on ModelNet40 with EdgeConv and Skip Connections**
Evaluate the performance of a **Graph Neural Network (GNN)** using **EdgeConv** layers with **skip connections** on the **ModelNet40** dataset.

---

### **Model Architecture**

The model consists of **EdgeConv blocks**, a **fusion layer**, and a **prediction head**:

#### **EdgeConv Blocks**
1. **Dynamic kNN Graph**:
   - Connects each point to its $$k=15$$ nearest neighbors, updated dynamically at each layer.
   
2. **EdgeConv Layer**:
   - Uses an MLP to compute edge features between points.
   - Includes:
     - Linear layers (input: 128, output: 64).
     - ReLU activation and batch normalization.
   - Aggregates features using **max-pooling**.

3. **Skip Connections**:
   - Residual connections between input and output to stabilize training.

#### **Fusion Layer**
- Aggregates features from all EdgeConv blocks.
- Includes:
  - A 2D convolution (input: 896, output: 1024, kernel: 1x1).
  - LeakyReLU activation and batch normalization.

#### **Prediction Head**
1. **First Convolution**:
   - Input: 2048, Output: 512.
   - LeakyReLU, batch normalization, and dropout (0.5).

2. **Second Convolution**:
   - Input: 512, Output: 256.
   - LeakyReLU, batch normalization, and dropout (0.5).

3. **Final Convolution**:
   - Input: 256, Output: 40 (number of classes).
   - Produces the final classification logits.

---

### **Hyperparameters**
- **Batch Size**: 32 (training), 50 (testing).
- **Epochs**: 400.
- **Learning Rate**: 0.001.
- **Optimizer**: AdamW with weight decay = 0.0001.
- **Dropout Rate**: 0.5.
- **k in kNN**: 15.
- **Number of Filters**: 64 per EdgeConv block.
- **Number of Blocks**: 14.
- **Activation Function**: ReLU (EdgeConv), LeakyReLU (fusion and prediction head).
- **Data Augmentation**: Translation, shuffling, and normalization.

</details>
<details>

<summary>Experiment 2</summary>

#### **Training on ModelNet40 with TransformerConv and Dense Connections**
Evaluate the performance of a **Graph Neural Network (GNN)** using **TransformerConv** layers with **dense connections** on the **ModelNet40** dataset.

---

### **Model Architecture**

The model consists of **TransformerConv blocks**, a **fusion layer**, and a **prediction head**:

#### **TransformerConv Blocks**
1. **Dynamic kNN Graph**:
   - Connects each point to its $$k=15$$ nearest neighbors, updated dynamically at each layer.
   
2. **TransformerConv Layer**:
   - Uses multi-head attention to compute edge features between points.
   - Includes:
     - TransformerConv layers with 4 attention heads.
     - Feed-forward networks with ReLU activation and dropout (0.1).
   - Aggregates features using **max-pooling**.

3. **Dense Connections**:
   - Concatenates outputs from all preceding layers to improve feature reuse and gradient flow.

#### **Fusion Layer**
- Aggregates features from all TransformerConv blocks.
- Includes:
  - A 2D convolution (input: 7168, output: 1024, kernel: 1x1).
  - LeakyReLU activation and batch normalization.

#### **Prediction Head**
1. **First Convolution**:
   - Input: 2048, Output: 512.
   - LeakyReLU, batch normalization, and dropout (0.5).

2. **Second Convolution**:
   - Input: 512, Output: 256.
   - LeakyReLU, batch normalization, and dropout (0.5).

3. **Final Convolution**:
   - Input: 256, Output: 40 (number of classes).
   - Produces the final classification logits.

---

### **Hyperparameters**
- **Batch Size**: 32 (training), 50 (testing).
- **Epochs**: 200.
- **Learning Rate**: 0.0001.
- **Optimizer**: AdamW with weight decay = 0.0001.
- **Dropout Rate**: 0.5.
- **k in kNN**: 15.
- **Number of Filters**: 256 per TransformerConv block.
- **Number of Blocks**: 7.
- **Activation Function**: ReLU (TransformerConv), LeakyReLU (fusion and prediction head).
- **Data Augmentation**: Translation, scaling, shuffling, and normalization.

</details>

## Results
### 🏆 ModelNet40 Classification (3D Point Clouds)
![image](https://github.com/user-attachments/assets/8d1469db-25d0-47d9-8526-0a0bc6ceb6b7)

### Benchmark Comparison

| **Model Name**              | **Overall Accuracy (%)** | **Category**           |
|-----------------------------|--------------------------|------------------------|
| RS-CNN                     | 93.6                    | Graph-Based            |
| MLVCNN                     | 94.16                   | Non-Graph-Based        |
| MHBN                       | 94.7                    |  Non-Graph-Based        |
| RotationNet                | 97.37                   | Non-Graph-Based        |
| PANORAMA-ENN               | 95.56                   |  Non-Graph-Based        |
| VRN Ensemble               | 95.54                   | NNon-Graph-Based        |
| MVCNN-New                  | 95.0                    |Non-Graph-Based        |
| SPNet                      | 92.63                   | Non-Graph-Based        |
| 3DCapsule                  | 92.7                    | Non-Graph-Based        |
| LDGCNN                     | 92.9                    | Graph-Based            |
| **EdgeConv (Our Model)**    | 93.27                   | Graph-Based (Our Model)|
| **TransformerConv (Our Model)** | 92.75              |  Graph-Based (Our Model)|
---
### EdgeConv Result
- **Test Time**: 7.55 ms per point cloud.
- **Number of Parameters**: 2.2M
- **Test Overall Accuracy**: 93.27%
- **Test Average Class Accuracy**: 90.09% 

---
### TransformerConv Result
- **Test Time**: 8.34 ms per point cloud.
- **Number of Parameters**: 15.8M
- **Test Overall Accuracy**: 92.75%
- **Test Average Class Accuracy**: 89.36%

---
### Key Insights

- EdgeConv achieves 93.27% OA – outperforms DGCNN by +0.37% with comparable parameters

- EdgeConv 7.2× Smaller than TransformerConv (2.2M vs 15.8M params) while being 10% faster
---

### Loss Graphs
Below are the training and validation loss graphs for both experiments to visualize convergence:
#### TransformerConv Graph
![image](https://github.com/user-attachments/assets/74ca964d-d026-4800-919c-1c8f8d38c666)
#### EdgeConv Graph
![image](https://github.com/user-attachments/assets/4b1c40e6-220c-4d5e-ae8e-3641a32ed3a4)

---
### Cross-Dataset Generalization on ShapeNet

To evaluate transfer learning capabilities, we fine-tuned our ModelNet40-pretrained EdgeConv and TransformerConv models on the ShapeNet classification task. Our pretrained architectures demonstrated Rapid Convergence as it Achieved 97.6% accuracy in 8 epochs with stabilized loss.

---


## How to Run

### Setup
To set up the environment, run the following command:

```bash
source gml_env_install.sh
```

This script will install all the necessary dependencies, including PyTorch, PyTorch Geometric, and other required libraries. It will also set up the CUDA paths and create a Conda environment named `gml`.

### Training and Testing
To train or test the models, use the following commands:

#### Training the EdgeConv Model
```bash
source run_main.sh --phase train --multi_gpus --block res --n_blocks 14 --data_dir /Dataset --n_filter 64 --batch_size 32 --conv edge
```
#### Training the TransformerConv Model
```bash
source run_main.sh --phase train --n_blocks 7 --block dense --data_dir /Dataset --n_filter 256 --batch_size 32 --conv trans --dynamic True --multi_gpu 
```
#### Test the EdgeConv Model
```bash
source run_main.sh --phase test --multi_gpus --block res --n_blocks 14 --data_dir /Dataset --n_filter 64 --conv edge --batch_size 32 --pretrained_model path/to/pretrained/edgeconv_model.pth
```
#### Test the TransformerConv Model
```bash
source run_main.sh --phase test --n_blocks 7 --block dense --data_dir /Dataset --n_filter 256 --conv trans --multi_gpu --batch_size 32 --dynamic True --pretrained_model path/to/pretrained/transformerconv_model.pth
```


You can customize the training and testing process using various options available in the `run_main.sh` or `config.py` scripts.

---
### Model Weights  
You can find the pre-trained model weights at the following link:  
[Download Model Weights](https://drive.google.com/drive/folders/1hm0q7_I8cLXCDSXgNoBQ-mlyvRxIgqOz)

### Options in `run_main.sh`
<details>
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
  
</details>

---

## AI Card

Artificial Intelligence (AI) aided the development of this project. Please find our AI-Usage card [here](ai-usage-card.pdf) (generated from [https://ai-cards.org/](https://ai-cards.org/)).
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

8. Shi, Yunsheng, et al. "Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification." arXiv preprint arXiv:2009.03509 (2020). [Link](https://arxiv.org/abs/2009.03509)

9. Zhao, Hengshuang, et al. "Point Transformer." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021. [Link](https://arxiv.org/abs/2012.09164)
10. Chen, G., Wang, M., Yang, Y., Yu, K., Yuan, L., & Yue, Y. (2023, May 19). PointGPT: Auto-regressively Generative Pre-training from Point Clouds. arXiv.org. [Link]([https://arxiv.org/abs/2012.09164](https://arxiv.org/abs/2305.11487))
11. Qi, Z., Dong, R., Zhang, S., Geng, H., Han, C., Ge, Z., Yi, L., & Ma, K. (2024, February 27). ShapeLLM: Universal 3D Object Understanding for Embodied Interaction. [Link](https://arxiv.org/abs/2402.17766)
12. Zeid, K. A., Schult, J., Hermans, A., & Leibe, B. (2023, March 29). Point2VEC for Self-Supervised Representation Learning on Point Clouds. [Link](https://arxiv.org/abs/230)
13. Ma, X., Qin, C., You, H., Ran, H., & Fu, Y. (2022, February 15). Rethinking network design and local geometry in point Cloud: a simple residual MLP framework. [Link](https://arxiv.org/abs/2202.07123)
14. Li, G., Müller, M., Thabet, A., & Ghanem, B. (2019, April 7). DeepGCNS: Can GCNs go as deep as CNNs? [Link](https://arxiv.org/abs/1904.03751)
15. Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017, June 7). PointNet++: deep hierarchical feature learning on point sets in a metric space. [Link](https://arxiv.org/abs/1706.02413)
16. Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2016, December 2). PointNet: Deep learning on point sets for 3D classification and segmentation. [Link](https://arxiv.org/abs/1612.00593)



---

## Contact
For any questions or feedback, feel free to reach out to us:
- **Ahmed Assy**: [ahmed.assy@stud.uni-goettingen.de](mailto:ahmed.assy@stud.uni-goettingen.de)
- **Mahmoud Abdellahi**: [mahmoud.abdellahi@stud.uni-goettingen.de](mailto:mahmoud.abdellahi@stud.uni-goettingen.de)

