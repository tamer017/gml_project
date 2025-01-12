import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, Conv2d

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    Create an activation layer based on the specified type.

    Args:
        act (str): Activation type ('relu', 'leakyrelu', 'prelu').
        inplace (bool): Whether to perform the operation in-place (default: False).
        neg_slope (float): Negative slope for LeakyReLU and initial value for PReLU (default: 0.2).
        n_prelu (int): Number of learnable parameters for PReLU (default: 1).

    Returns:
        nn.Module: The activation layer.
    """
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(f"Activation layer '{act}' is not implemented.")

def norm_layer(norm, nc):
    """
    Create a normalization layer based on the specified type.

    Args:
        norm (str): Normalization type ('batch2d', 'batch', 'instance').
        nc (int): Number of channels.

    Returns:
        nn.Module: The normalization layer.
    """
    norm = norm.lower()
    if norm == 'batch2d':
        return nn.BatchNorm2d(nc, affine=True)
    elif norm == 'batch':
        return nn.BatchNorm1d(nc, affine=True)
    elif norm == 'instance':
        return nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError(f"Normalization layer '{norm}' is not implemented.")

class BasicConv(Seq):
    """
    Basic Convolutional Block with optional activation, normalization, and dropout layers.

    Args:
        channels (list[int]): List of input and output channels for each convolutional layer.
        act (str): Activation type ('relu', 'leakyrelu', 'prelu', or 'none').
        norm (str): Normalization type ('batch2d', 'batch', 'instance', or 'none').
        bias (bool): Whether to use bias in convolutional layers (default: True).
        drop (float): Dropout rate (default: 0.0).
    """
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        layers = []
        for i in range(1, len(channels)):
            layers.append(Conv2d(channels[i - 1], channels[i], kernel_size=1, bias=bias))
            if act and act.lower() != 'none':
                layers.append(act_layer(act))
            if norm and norm.lower() != 'none':
                layers.append(norm_layer(norm, channels[i]))
            if drop > 0:
                layers.append(nn.Dropout2d(drop))
        super().__init__(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights and biases for the layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class BasicLayer(Seq):
    """
    Basic Linear Layer Block with activation and normalization layers.

    Args:
        channels (tuple[int, int]): Input and output channel dimensions.
        act (str): Activation type ('relu', 'leakyrelu', 'prelu', or 'none').
        norm (str): Normalization type ('batch', 'none').
        bias (bool): Whether to use bias in linear layers (default: True).
    """
    def __init__(self, channels, act='relu', norm=None, bias=True):
        in_channels, out_channels = channels
        layers = [
            Lin(in_channels, out_channels, bias=bias),
            act_layer(act),
            norm_layer(norm, out_channels),
            Lin(out_channels, out_channels, bias=bias),
            act_layer(act),
            norm_layer(norm, out_channels),
        ]
        super().__init__(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights and biases for the layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.InstanceNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

def convert_idx_to_edge_index(idx):
    """
    Converts a tensor of indices (2, B, N, l) into an edge index tensor (2, number_of_edges).

    Args:
        idx (Tensor): Neighbor indices of shape (2, B, N, l), where:
                      - 2: Source and target indices.
                      - B: Batch size.
                      - N: Number of nodes.
                      - l: Number of neighbors.

    Returns:
        Tensor: Edge index tensor of shape (2, number_of_edges).
    """
    batch_size, num_nodes, _ = idx.shape[1:4]

    # Create batch offsets for unique indexing
    batch_offsets = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_nodes

    # Add batch offsets to source and target indices
    src_idx = idx[0] + batch_offsets
    tgt_idx = idx[1] + batch_offsets

    # Flatten and combine indices
    edge_index = torch.stack([src_idx.view(-1), tgt_idx.view(-1)], dim=0)
    return edge_index
