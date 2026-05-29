# CleanViewGAD

Minimal and accurate reproduction guide. The current entry script is `train.py`.

## 1. Environment

Create the same Conda environment used for the experiments:

```bash
conda create -p D:\conda_envs_unified\adgcl python=3.8.13 -y
conda activate D:\conda_envs_unified\adgcl
```

Install PyTorch and graph-learning dependencies:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.9 torch-sparse==0.6.15 torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install torch-geometric==2.6.1
pip install dgl-cu113==0.9.1 dgl==0.4.3
```

Install common packages:

```bash
pip install numpy==1.23.5 scipy==1.9.1 pandas==1.5.3 scikit-learn==1.2.2 matplotlib==3.7.1 networkx==2.8.8 tqdm==4.64.1 pygod==1.1.0 openpyxl==3.1.5 seaborn==0.12.2
```

## 2. Dataset Format

Put `.mat` datasets under `./dataset/`:

```text
dataset/cora.mat
dataset/citeseer.mat
dataset/pubmed.mat
dataset/BlogCatalog.mat
```

The loader in `sarem/utils.py` supports the following keys:

```text
Features: Attributes or X
Graph:    Network or A
Labels:   Label, gnd, or label
```

Labels are used only for evaluation.

## 3. Reproduction Commands

Run one dataset with one seed:

```bash
python train.py --data_dir ./dataset --result_root ./sarem_results --datasets cora --runs 1 --seed_start 0 --device cuda:0
```

Run four datasets with five seeds:

```bash
python train.py \
  --data_dir ./dataset \
  --result_root ./sarem_results \
  --datasets cora,citeseer,pubmed,BlogCatalog \
  --runs 5 \
  --seed_start 0 \
  --device cuda:0
```

CPU version:

```bash
python train.py --data_dir ./dataset --result_root ./sarem_results --datasets cora --runs 1 --seed_start 0 --device cpu
```

Optional PNG plots:

```bash
python train.py --data_dir ./dataset --result_root ./sarem_results --datasets cora --runs 1 --seed_start 0 --device cuda:0 --save_png
```

Main configurable arguments:

```text
--datasets            comma-separated dataset names
--runs                number of repeated runs
--seed_start          first random seed; seeds are seed_start, seed_start+1, ...
--device              cuda:0 or cpu
--num_epoch           override default training epochs
--lr                  override default learning rate
--batch_size          default 128
--auc_test_rounds     default 64
--clean_drop_ratio    default 0.1
--clean_sub_weight    default 0.5
--alpha               contrastive evidence weight, default 1.0
--beta                calibrated reconstruction weight, default 0.6
--gamma               substructure evidence weight, default 0.3
```

Expected output structure:

```text
sarem_results/
  run_config.json
  summary_runs_live.csv
  summary_runs.csv
  summary_mean_std.csv
  dataset_plot_json_files.json
  cora/
    seed_0/
      metrics.json
      node_scores.csv
      roc_curve.csv
      pr_curve.csv
      train_log.csv
      clean_view_info.json
      plot_data.json
      best_model.pkl
      plots/                  # generated only when --save_png is used
  plot_json/
    cora_YYYYMMDD_HHMMSS.json
```

Example terminal output:

```text
[RUN] dataset=cora seed=0 device=cuda:0 lr=0.001 epochs=100 subgraph=2
[1/4] Computing original ego-substructure evidence: cora/seed0
[2/4] Building clean graph: cora/seed0
[3/4] Computing clean-view ego-substructure evidence: cora/seed0
[4/4] Starting model training/testing: cora/seed0
[DONE] cora         seed=0 AUC=0.xxxx AP=0.xxxx FPR95=0.xxxx FDR@K=0.xxxx
All experiments finished.
Dataset-level plotting JSON files are saved in: sarem_results/plot_json
```
