#!/bin/bash
#SBATCH --job-name=point-cloud-classification  # Name of the job
#SBATCH -t 40:00:00                            # Maximum runtime of 40 hours
#SBATCH -p gpu                                 # Use the GPU partition
#SBATCH -G RTX5000:4                           # Request 4 RTX5000 GPUs
#SBATCH --mem=24G                              # Request 24GB of memory
#SBATCH --nodes=1                              # Use 1 node
#SBATCH --ntasks=1                             # Run 1 task
#SBATCH --cpus-per-task=4                      # Use 4 CPU cores per task
#SBATCH --output=./slurm_files/slurm-%x-%j.out # Output log file (job name and ID)
#SBATCH --error=./slurm_files/slurm-%x-%j.err  # Error log file (job name and ID)
#SBATCH --mail-type=ALL                        # Send emails for all job events
#SBATCH --mail-user=ahmed.assy@stud.uni-goettingen.de  # Email address for notifications

# Activate the Conda environment named 'gml'
conda activate gml

# Run the main Python script with parameters
# -------------------------
# General parameters
# -------------------------
# --phase: Phase of the pipeline, either train or test (default: train)
# --exp_name: Experiment name, used for logs and checkpoints (default: '')
# --use_cpu: Set True to use CPU, False to use GPU(s) (default: False)
# --root_dir: Directory to save logs and checkpoints (default: 'log')

# -------------------------
# File and dataset parameters
# -------------------------
# --data_dir: Directory containing the dataset (default: 'Dataset')
# --dataset: Dataset to use (e.g., ShapeNetPart or ModelNet40) (default: 'ShapeNetPart')
# --num_points: Number of points per point cloud (default: 2048)
# --in_channels: Input dimensions (e.g., x, y, z coordinates) (default: 3)

# -------------------------
# Training parameters
# -------------------------
# --batch_size: Batch size for training (default: 32)
# --epochs: Number of epochs to train (default: 400)
# --use_sgd: Set True to use SGD optimizer, False to use AdamW (default: False)
# --weight_decay: L2 regularization factor for weights (default: 1e-4)
# --lr: Learning rate for training (default: 1e-3)
# --seed: Random seed for reproducibility (default: 1)
# --multi_gpus: Set True to enable multi-GPU training (default: False)

# -------------------------
# Testing parameters
# -------------------------
# --test_batch_size: Batch size used during evaluation (default: 50)
# --pretrained_model: Path to a pretrained model (leave empty if not used) (default: '')

# -------------------------
# Graph and model architecture parameters
# -------------------------
# --k: Number of nearest neighbors for constructing the graph (default: 15)
# --block: Type of backbone block (e.g., res, plain, or dense) (default: 'res')
# --conv: Type of graph convolution (e.g., edge or mr) (default: 'edge')
# --act: Activation function (relu, prelu, or leakyrelu) (default: 'relu')
# --norm: Type of normalization (batch or instance) (default: 'batch')
# --bias: Add bias to convolution layers (set to True or False) (default: True)
# --n_blocks: Number of blocks in the backbone architecture (default: 14)
# --n_filters: Number of channels for deep features (default: 64)
# --emb_dims: Dimension of the embeddings (default: 1024)
# --dropout: Dropout rate for regularization (default: 0.5)
# --dynamic: Set True to use dynamic graph construction (default: False)

# -------------------------
# Advanced features
# -------------------------
# --fine_tune: Set True to enable fine-tuning on a new dataset (default: False)
# --fine_tune_num_classes: Number of classes in the new dataset (for fine-tuning) (default: 40)
# --no_dilation: Set True to disable dilated kNN (default: False)
# --epsilon: Stochastic epsilon value for graph construction (default: 0.2)
# --no_stochastic: Set True to disable stochastic graph construction (default: False)

python main.py --multi_gpus