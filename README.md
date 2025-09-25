# Implicit Regularization In Group Robustness

This project aims to investigate the effect of training hyperparameters on group robustness.

## Getting Started

### Repository Setup
To start working with this repository, navigate to the root directory and create the necessary directories for experiments and data:

```bash
cd training_dynamics
mkdir -p experiments/checkpoints
```

### Virtual Environment Setup
It is recommended to use a virtual environment to manage dependencies. 
Make sure you are using Python 3.10.8.
Create and activate a virtual environment using:

```bash
virtualenv .venv
source .venv/bin/activate  # On MacOS/Linux
.venv\Scripts\activate    # On Windows
```

Install the necessary dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```
or
```bash
cat requirements.txt | xargs -n 1 pip3 install
```

### 📚 Dataset Support

This repo supports training on the following datasets:

`cifar10`
`cmnist (Colored MNIST)`
`domino (compositional CIFAR10 + MNIST)`
`celeba (CelebA with spurious gender/attribute)`
`waterbirds (Waterbirds dataset from WILDS)`
All data will be automatically downloaded and cached in the data/ directory unless provided explicitly.

### 🛠️ Training Run

You can manually train a model:

### 🧪 Example Command
```
python run.py --dataset cifar10 --architecture resnet18 --epochs 100 --random_seed 42 --spurious_correlation 0.5 --wandb_run test_run
```

#### 🔍 Common Arguments

Argument	Description	Default
- **`--dataset`**  
  Dataset name: `cifar10`, `cmnist`, `domino`, `celeba`, `waterbirds`  
  **Default**: `cifar10`

- **`--architecture`**  
  Model architecture: `resnet18` or `resnet50`  
  **Default**: `resnet18`

- **`--epochs`**  
  Number of training epochs  
  **Default**: `100`

- **`--spurious_correlation`**  
  Strength of spurious correlation in training set  
  **Default**: `0.5`

- **`--setup`**  
  Validation setup: `unknown` (spurious) or `known` (clean)  
  **Default**: `unknown`

- **`--model_selection`**  
  wga or acc, use acc for unknown and wga for known.  
  **Default**: `acc`

#### 📊 Using Weights & Biases (wandb)

We used `wandb` for logging the training and evaluation process.  
In our setup, we relied on a **local wandb server**, but you can either:  
- use your own self-hosted wandb server, or  
- connect to the **online wandb service** for centralized experiment tracking.  

Each run logs metrics like losses and accuracies, and naming runs (via `--wandb_run`) helps organize and compare experiments in the wandb dashboard.
