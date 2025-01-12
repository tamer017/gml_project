import torch
from torch import nn
from torch_cluster import knn_graph
import torch.nn.functional as F


class DenseDilated(nn.Module):
    """
    Implements dilated neighbor selection from a given neighbor list.

    Attributes:
        k (int): Number of neighbors to consider.
        dilation (int): Dilation rate for selecting neighbors.
        stochastic (bool): If True, enables stochastic dilation.
        epsilon (float): Probability of random neighbor selection when stochastic is enabled.

    Methods:
        forward(edge_index): Applies dilation to the neighbor indices.
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        """
        Forward pass for selecting dilated neighbors.

        Args:
            edge_index (Tensor): Neighbor indices of shape (2, batch_size, num_points, k).

        Returns:
            Tensor: Dilated neighbor indices.
        """
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


def pairwise_distance(x):
    """
    Computes pairwise Euclidean distance for a batch of point clouds.

    Args:
        x (Tensor): Input tensor of shape (batch_size, num_points, num_dims).

    Returns:
        Tensor: Pairwise distance matrix of shape (batch_size, num_points, num_points).
    """
    x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


def dense_knn_matrix(x, k=16):
    """
    Computes K-nearest neighbors based on pairwise distances.

    Args:
        x (Tensor): Input tensor of shape (batch_size, num_dims, num_points, 1).
        k (int): Number of nearest neighbors to compute.

    Returns:
        Tensor: Neighbor indices of shape (2, batch_size, num_points, k).
    """
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        _, nn_idx = torch.topk(-pairwise_distance(x.detach()), k=k)
        center_idx = torch.arange(0, n_points, device=x.device).expand(batch_size, k, -1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilatedKnnGraph(nn.Module):
    """
    Constructs a graph based on dilated KNN for dense data.

    Attributes:
        k (int): Number of neighbors to consider.
        dilation (int): Dilation rate for neighbor selection.
        stochastic (bool): If True, enables stochastic dilation.
        epsilon (float): Probability of random neighbor selection when stochastic is enabled.

    Methods:
        forward(x): Computes the graph by finding dilated KNN neighbors.
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = dense_knn_matrix

    def forward(self, x):
        """
        Forward pass to compute dilated KNN graph.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_dims, num_points, 1).

        Returns:
            Tensor: Dilated KNN neighbor indices.
        """
        edge_index = self.knn(x, self.k * self.dilation)
        return self._dilated(edge_index)


class DilatedKnnGraph(nn.Module):
    """
    Constructs a graph based on dilated KNN for sparse data using torch_cluster's knn_graph.

    Attributes:
        k (int): Number of neighbors to consider.
        dilation (int): Dilation rate for neighbor selection.
        stochastic (bool): If True, enables stochastic dilation.
        epsilon (float): Probability of random neighbor selection when stochastic is enabled.

    Methods:
        forward(x): Computes the graph by finding dilated KNN neighbors for each batch.
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = knn_graph

    def forward(self, x):
        """
        Forward pass to compute dilated KNN graph for sparse data.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_dims, num_points, 1).

        Returns:
            Tensor: Dilated KNN neighbor indices of shape (2, batch_size, num_points, k).
        """
        x = x.squeeze(-1)
        B, C, N = x.shape
        edge_index = []
        for i in range(B):
            edgeindex = self.knn(x[i].contiguous().transpose(1, 0).contiguous(), self.k * self.dilation)
            edgeindex = edgeindex.view(2, N, self.k * self.dilation)
            edge_index.append(edgeindex)
        edge_index = torch.stack(edge_index, dim=1)
        return self._dilated(edge_index)
