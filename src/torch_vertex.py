import torch
from torch import nn
from .layers import convert_idx_to_edge_index, norm_layer,BasicLayer, act_layer
from .torch_edge import DenseDilatedKnnGraph, DilatedKnnGraph
from torch_geometric.nn import TransformerConv, EdgeConv, DynamicEdgeConv


class DynamicEdgeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k, act='relu', norm=None, bias=True, aggr='max'):
        """
        Dynamic Edge Convolution Layer using PyTorch Geometric's DynamicEdgeConv.

        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            k (int): Number of nearest neighbors for kNN graph.
            act (str): Activation function name.
            norm (str or None): Normalization type ('batchnorm', 'layernorm', or None).
            bias (bool): Whether to include bias in the MLP layers.
            aggr (str): Aggregation type ('max', 'mean', or 'add') for DynamicEdgeConv.
        """
        super(DynamicEdgeConvLayer, self).__init__()

        # MLP for edge features
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels, bias=bias),
            act_layer(act),
            nn.Linear(out_channels, out_channels, bias=bias),
            act_layer(act)
        )

        # DynamicEdgeConv layer
        self.conv = DynamicEdgeConv(self.mlp, k, aggr)

        # Normalization
        self.norm = norm_layer(norm, out_channels) if norm else None

    def forward(self, x, edge_index=None):
        """
        Forward pass for DynamicEdgeConvLayer.

        Args:
            x (torch.Tensor): Node features of shape [B, C, N, 1].

        Returns:
            torch.Tensor: Updated node features of shape [B * N, out_channels].
        """
        B, C, N, _ = x.shape
        
        # Create batch indices
        batch = torch.arange(B, device=x.device).repeat_interleave(N)

        x = x.squeeze(-1)  # Remove trailing dimension [B, C, N, 1] -> [B, C, N]
        # Reshape input to [N_total, F]
        # x is initially [B, C, N] -> Permute to [B, N, C]
        x = x.permute(0, 2, 1).reshape(-1, x.size(1))

        # Pass through DynamicEdgeConv
        out = self.conv(x, batch)

        # Apply normalization
        if self.norm:
            out = self.norm(out)
        
        # Reshape back to original format
        out_channels = out.shape[-1]
        out = out.view(B, N, out_channels).permute(0, 2, 1).unsqueeze(-1)  # [B, C_out, N, 1]

        return out


class EdgeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='max'):
        """
        EdgeConvLayer with customizable activation, normalization, and aggregation.

        Args:
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            act (str): Activation function name.
            norm (str or None): Normalization type.
            bias (bool): Whether to include bias in the layers.
            aggr (str): Aggregation type for EdgeConv.
        """
        super(EdgeConvLayer, self).__init__()
        # Define EdgeConv
        self.edge_conv = EdgeConv(
            nn=BasicLayer([in_channels*2, out_channels], act, norm, bias),
            aggr=aggr
        )

        # Normalization
        self.norm = norm_layer(norm, out_channels) if norm else None

    def forward(self, x, edge_index):
        """
        Forward pass for EdgeConvLayer.

        Args:
            x (Tensor): Input node features of shape [B, C, N, 1].
            edge_index (Tensor): Edge indices of shape [2, B, N, l].

        Returns:
            Tensor: Output node features of shape [B, out_channels, N, 1].
        """
        # Convert edge indices to PyG edge_index format
        edge_index = convert_idx_to_edge_index(edge_index)

        # Reshape x to match PyG's input format
        B, C, N, _ = x.shape
        x = x.squeeze(-1).permute(0, 2, 1).reshape(B * N, C)  # [B*N, C]
        # Apply EdgeConv
        out = self.edge_conv(x, edge_index)  # [B*N, out_channels]

        # # Apply normalization
        if self.norm:
            out = self.norm(out)

        # Reshape back to original format
        out_channels = out.shape[-1]
        out = out.view(B, N, out_channels).permute(0, 2, 1).unsqueeze(-1)  # [B, C_out, N, 1]
        return out
   

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim=None, heads=4, dropout=0.2):
        super(GraphTransformerLayer, self).__init__()
        
        # TransformerConv for attention mechanism
        self.transformer = TransformerConv(
            in_channels,
            out_channels // heads,
            edge_dim=edge_dim,
            heads=heads,
            dropout=dropout
        )
        # self.norm = nn.BatchNorm1d(out_channels)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.ReLU() ,
            nn.Dropout(dropout),
            nn.Linear(out_channels * 2, out_channels)
        )
        
        # Optional activation after the FFN
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_attr=None):
        # Reshape node features
        B, C, N, _ = x.shape
        x = x.squeeze(-1).permute(0, 2, 1).reshape(B * N, C)

        # Ensure edge_index has the correct dimensions
        if edge_index.dim() == 4:
            edge_index = edge_index.permute(1, 0, 2, 3).reshape(2, -1)
        elif edge_index.dim() == 3:
            edge_index = edge_index.permute(1, 0, 2).reshape(2, -1)

        # Attention mechanism via TransformerConv
        out = self.transformer(x, edge_index, edge_attr)  # Shape: [B*N, out_channels]

        # out = self.norm(out)

        # Feed-forward processing
        out = self.feed_forward(out)  # Shape: [B*N, out_channels]
        
        # Optional non-linearity
        out = self.activation(out)

        # Reshape back to [B, out_channels, N, 1]
        out_channels = out.shape[1]
        out = out.view(B, N, out_channels).permute(0, 2, 1).unsqueeze(-1)
        return out
   
class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='gcn', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConvLayer(in_channels, out_channels, act, norm, bias)
        elif conv == 'trans':
            self.gconv = GraphTransformerLayer(in_channels, out_channels)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))
    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DynConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        if knn == 'matrix':
            self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        else:
            self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)

    
class PlainDynBlock2d(nn.Module):
    """
    Plain Dynamic graph convolution block
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, knn='matrix', dynamic=True):
        super(PlainDynBlock2d, self).__init__()
        if dynamic == True:
            self.body = DynamicEdgeConvLayer(in_channels, in_channels, kernel_size, act, norm, bias)
        else:
            self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x, edge_index=None):
        return self.body(x, edge_index)
    
    
class ResDynBlock2d(nn.Module):
    """
    Residual Dynamic graph convolution block
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, knn='matrix', res_scale=1, dynamic=True):
        super(ResDynBlock2d, self).__init__()
        if dynamic == True:
            self.body = DynamicEdgeConvLayer(in_channels, in_channels, kernel_size, act, norm, bias)
        else:
            self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)
        self.res_scale = res_scale

    def forward(self, x, edge_index=None):
        return self.body(x, edge_index) + x*self.res_scale


class DenseDynBlock2d(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channels, out_channels=64,  kernel_size=9, dilation=1, conv='edge',
                 act='relu', norm=None,bias=True, stochastic=False, epsilon=0.0, knn='matrix', dynamic=True):
        super(DenseDynBlock2d, self).__init__()
        if conv == 'trans':
            self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

        elif dynamic == True:
            self.body = DynamicEdgeConvLayer(in_channels, in_channels, kernel_size, act, norm, bias)
        else:
            self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x, edge_index=None):
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1)
