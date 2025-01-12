#!/usr/bin/env bash
# make sure command is : source gml_env_install.sh

# install anaconda3.
# cd ~/
# wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
# bash Anaconda3-2019.07-Linux-x86_64.sh

source ~/.bashrc
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"  # Adjust based on your GPU (e.g., A100: 8.0; RTX 30xx: 8.6)

# Update paths for CUDA 12.4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Create and activate a new Conda environment
conda create -n gml python=3.8 -y
conda activate gml

# Install PyTorch compatible with CUDA 12.4
conda install -y pytorch=2.4.1 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install PyTorch Geometric dependencies
CUDA=cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.1+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.1+${CUDA}.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+${CUDA}.html

# Install PyTorch Geometric
pip install torch-geometric

# Install additional dependencies
pip install tqdm requests

# Additional package required for DGL implementation
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install h5py kaggle scikit-learn torchmetrics
