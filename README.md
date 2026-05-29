# CleanViewGAD

Minimal reproduction guide for CleanViewGAD / SAREM-GAD-style unsupervised graph anomaly detection.

## 1. Environment

Create the same environment used in our experiments:

```bash
conda create -p D:\conda_envs_unified\adgcl python=3.8.13 -y
conda activate D:\conda_envs_unified\adgcl
```

Install PyTorch 1.12.1 with CUDA 11.3:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Install graph learning dependencies:

```bash
pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric==2.6.1
pip install dgl-cu113==0.9.1 dgl==0.4.3
```

Install common scientific packages:

```bash
pip install numpy==1.23.5 scipy==1.9.1 pandas==1.5.3 scikit-learn==1.2.2 matplotlib==3.7.1 networkx==2.8.8 tqdm==4.64.1 pygod==1.1.0 openpyxl==3.1.5 seaborn==0.12.2
```

## 2. Dataset Format

Put datasets under `./dataset/`:

```text
dataset/cora.mat
dataset/citeseer.mat
dataset/pubmed.mat
dataset/BlogCatalog.mat
```

Each `.mat` file should contain:

```text
adj       adjacency matrix
features  node feature matrix
label     anomaly labels, or labels
```

Labels are only used for evaluation.

## 3. Reproduction Commands

Run one dataset:

```bash
python run_sarem_pack.py --dataset cora --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
```

Run four main datasets with five seeds:

```bash
python run_paper_experiments_sarem.py \
  --data_dir ./dataset \
  --datasets cora,citeseer,pubmed,BlogCatalog \
  --variant sarem_score \
  --seeds 0,1,2,3,4 \
  --output_root ./sarem_results
```

Expected output example:

```text
sarem_results/cora/sarem_score/seed_0/
  metrics.json
  node_scores.csv
  roc_curve.csv
  pr_curve.csv
  train_log.csv
  clean_view_info.json
  plot_data.json
  best_model.pkl
```
