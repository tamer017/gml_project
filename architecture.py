#!/usr/bin/env python
# -*- coding: utf-8 -*-
import __init__
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from src import GraphConv2d, BasicConv, ResDynBlock2d, DenseDynBlock2d, PlainDynBlock2d, DenseDilatedKnnGraph, DynamicEdgeConvLayer



class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        knn = 'matrix'  # implement knn using matrix multiplication
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        c_growth = channels
        emb_dims = opt.emb_dims
        dilation = 1
        self.n_blocks = opt.n_blocks
        self.dynamic = opt.dynamic
        self.opt = opt

        
        if self.dynamic:
            # Head using DynamicEdgeConvLayer
            self.dynamic_head = DynamicEdgeConvLayer(
                in_channels=opt.in_channels,
                out_channels=channels,
                k=k,
                act=act,
                norm=norm,
                bias=bias
            )
        else:
            self.knn = DenseDilatedKnnGraph(k, dilation, stochastic, epsilon)
            self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias=False)

        if opt.block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                  norm, bias, stochastic, epsilon, knn, dynamic=self.dynamic)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks-1)) * self.n_blocks // 2)

        elif opt.block.lower() == 'res':
            if opt.use_dilation:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, i + dilation, conv, act, norm,
                                                    bias, stochastic, epsilon, knn, dynamic=self.dynamic)
                                      for i in range(self.n_blocks - 1)])
            else:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, 1, conv, act, norm,
                                                    bias, stochastic, epsilon, knn, dynamic=self.dynamic)
                                      for _ in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        else:
            # Plain GCN. No dilation, no stochastic, no residual connections
            stochastic = False

            self.backbone = Seq(*[PlainDynBlock2d(channels, k, 1, conv, act, norm,
                                                  bias, stochastic, epsilon, knn, dynamic=self.dynamic)
                                  for i in range(self.n_blocks - 1)])

            fusion_dims = int(channels+c_growth*(self.n_blocks-1))

        self.fusion_block = BasicConv([fusion_dims, emb_dims], 'leakyrelu', norm='batch2d', bias=False)
        self.prediction = Seq(*[BasicConv([emb_dims * 2, 512], 'leakyrelu', norm='batch2d', drop=opt.dropout),
                                BasicConv([512, 256], 'leakyrelu', norm='batch2d', drop=opt.dropout),
                                BasicConv([256, opt.n_classes], None, None)])
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        if self.dynamic:
            feats = [self.dynamic_head(inputs)]
        else:
            feats = [self.head(inputs, self.knn(inputs[:, 0:3]))]

        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))
        
        feats = torch.cat(feats, dim=1)
        fusion = self.fusion_block(feats)
        x1 = F.adaptive_max_pool2d(fusion, 1)
        x2 = F.adaptive_avg_pool2d(fusion, 1)
        return self.prediction(torch.cat((x1, x2), dim=1)).squeeze(-1).squeeze(-1)
    
    def update_num_classes(self, new_num_classes):
        """
        Replace the last layer of the prediction module with a new layer 
        that has the same structure but different output dimensions.

        Args:
            new_num_classes (int): The number of output classes for the new layer.
        """
        last_layer = self.prediction[-1]
        if isinstance(last_layer, BasicConv):
            # Replace Last BasicConv layer
            self.prediction[-1] = BasicConv([256, new_num_classes], None, None).to(self.opt.device)
        else:
            raise ValueError(f"Unsupported layer type: {type(last_layer)}")


