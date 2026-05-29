# CleanViewGAD

CleanViewGAD is an unsupervised graph anomaly detection project for attributed graphs. The code supports node-level anomaly detection with clean-view structural refinement, reliability-aware evidence analysis, and reproducible evaluation on common graph anomaly detection benchmarks.

This repository follows a CleanViewGAD / RGRGAD-style workflow: raw attributed graphs are loaded from `.mat` files, model variants are trained under an unsupervised setting, and node-level anomaly scores are evaluated using ROC-AUC, PR-AUC, Top-K, and diagnostic visualization analyses.

## 1. Environment Installation

Create a Conda environment:

```bash
conda create -n cleanviewgad python=3.9 -y
conda activate cleanviewgad
```

Install PyTorch. For CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only environments:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

If some visualization dependencies are missing, install them manually:

```bash
pip install numpy scipy pandas scikit-learn matplotlib networkx tqdm h5py
```

## 2. Dataset Preparation

Put the datasets under:

```text
dataset/
```

Main datasets:

```text
dataset/cora.mat
dataset/citeseer.mat
dataset/pubmed.mat
dataset/BlogCatalog.mat
```

Optional datasets, if supported by local scripts:

```text
dataset/ACM.mat
dataset/books.mat
dataset/disney.mat
dataset/enron.mat
dataset/Flickr.mat
```

Each `.mat` file should contain fields similar to:

```text
adj       adjacency matrix
features  node feature matrix
label     anomaly labels
labels    alternative anomaly-label field
```

Labels are used only for evaluation, not for training.

## 3. Dataset Description

### Cora

Cora is a citation network. Nodes represent papers, edges represent citation links, and node attributes are bag-of-words features. It is commonly used to evaluate graph anomaly detection on sparse attributed citation graphs.

### Citeseer

Citeseer is a citation network with high-dimensional sparse text attributes and relatively sparse graph connectivity. It is useful for testing whether a detector can handle weak or noisy local contexts.

### PubMed

PubMed is a larger citation network with more nodes and edges than Cora and Citeseer. It is suitable for evaluating scalability and global anomaly ranking on larger attributed graphs.

### BlogCatalog

BlogCatalog is a social network. Nodes represent users, edges represent social relations, and node attributes describe user interests or profile features. Compared with citation networks, BlogCatalog usually has denser and noisier local structures.

## 4. Repository Structure

```text
CleanViewGAD/
  dataset/
  sarem_results/
  nature_vis/
  run_sarem_pack.py
  run_paper_experiments_sarem.py
  run_all_sarem_datasets.py
  model.py
  clean_view.py
  reliability.py
  substructure.py
  utils.py
  requirements.txt
  README.md
```

Main files:

```text
run_sarem_pack.py              Single-dataset training and evaluation entry.
run_all_sarem_datasets.py      Batch execution over multiple datasets.
run_paper_experiments_sarem.py Paper-level experiment runner.
model.py                       Core GNN and reconstruction model definitions.
clean_view.py                  Clean-view construction and edge filtering.
reliability.py                 Reliability-calibrated reconstruction utilities.
substructure.py                Ego-substructure evidence extraction.
utils.py                       Dataset loading, metrics, logging, and helper functions.
nature_vis/                    Nature-style node-level mechanism visualization scripts.
```

## 5. Run One Dataset

Example for Cora:

```bash
python run_sarem_pack.py --dataset cora --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
```

Example for Citeseer:

```bash
python run_sarem_pack.py --dataset citeseer --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
```

Example for PubMed:

```bash
python run_sarem_pack.py --dataset pubmed --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
```

Example for BlogCatalog:

```bash
python run_sarem_pack.py --dataset BlogCatalog --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
```

Check actual arguments with:

```bash
python run_sarem_pack.py --help
```

## 6. Run Four Main Datasets

Recommended command:

```bash
python run_all_sarem_datasets.py \
  --datasets cora,citeseer,pubmed,BlogCatalog \
  --data_dir ./dataset \
  --variant sarem_score \
  --seeds 0,1,2,3,4 \
  --output_root ./sarem_results
```

If your batch script does not support comma-separated arguments, run the datasets one by one:

```bash
python run_sarem_pack.py --dataset cora --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
python run_sarem_pack.py --dataset citeseer --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
python run_sarem_pack.py --dataset pubmed --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
python run_sarem_pack.py --dataset BlogCatalog --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
```

## 7. Paper-Level Experiment Command

```bash
python run_paper_experiments_sarem.py \
  --data_dir ./dataset \
  --datasets cora,citeseer,pubmed,BlogCatalog \
  --variant sarem_score \
  --seeds 0,1,2,3,4 \
  --output_root ./sarem_results
```

## 8. Expected Output

The expected output directory is:

```text
sarem_results/{dataset}/{variant}/seed_{seed}/
```

A typical run may generate:

```text
metrics.json
node_scores.csv
roc_curve.csv
pr_curve.csv
train_log.csv
clean_view_info.json
plot_data.json
best_model.pkl
```

Important files:

```text
metrics.json          Final ROC-AUC, PR-AUC, and other metrics.
node_scores.csv       Node-level anomaly scores and evidence values.
clean_view_info.json  CleanView edge statistics and removed/kept edge information.
plot_data.json        Optional plotting data for visualization scripts.
```

## 9. Nature-Style Mechanistic Visualization

If the `nature_vis/` module is available, generate four-dataset mechanism figures with:

```bash
python nature_vis/05_build_all_nature_figures.py \
  --result_root ./sarem_results \
  --data_dir ./dataset \
  --datasets cora,citeseer,pubmed,BlogCatalog \
  --variant sarem_score \
  --seed 0 \
  --top_k 1 \
  --output_root ./nature_vis_outputs \
  --overwrite
```

Windows PowerShell example:

```powershell
python nature_vis/05_build_all_nature_figures.py ^
  --result_root ./sarem_results ^
  --data_dir "D:\DESK\金融欺诈\AAA金融欺诈\SAREM_GAD_final_pack\dataset" ^
  --datasets cora,citeseer,pubmed,BlogCatalog ^
  --variant sarem_score ^
  --seed 0 ^
  --top_k 1 ^
  --output_root ./nature_vis_outputs ^
  --overwrite
```

Expected outputs:

```text
nature_vis_outputs/figures/four_dataset_node_case_study_grid.pdf
nature_vis_outputs/figures/four_dataset_node_case_study_grid.png
nature_vis_outputs/figures/four_dataset_evidence_distribution_grid.pdf
nature_vis_outputs/figures/four_dataset_evidence_distribution_grid.png
nature_vis_outputs/figures/four_dataset_cleanview_edge_refinement_grid.pdf
nature_vis_outputs/figures/four_dataset_cleanview_edge_refinement_grid.png
nature_vis_outputs/figures/four_dataset_raw_degree_vs_clean_degree_grid.pdf
nature_vis_outputs/figures/four_dataset_raw_degree_vs_clean_degree_grid.png
nature_vis_outputs/paper_text/visual_analysis.tex
nature_vis_outputs/paper_text/visual_analysis_cn.txt
```

## 10. RGRGAD-Style Reproducible Protocol

Recommended protocol:

```text
Datasets: cora, citeseer, pubmed, BlogCatalog
Seeds: 0, 1, 2, 3, 4
Metrics: ROC-AUC, PR-AUC, FPR@95TPR, Precision@K, Recall@K
Result root: ./sarem_results
Variant name: sarem_score
```

Recommended command:

```bash
python run_paper_experiments_sarem.py \
  --data_dir ./dataset \
  --datasets cora,citeseer,pubmed,BlogCatalog \
  --variant sarem_score \
  --seeds 0,1,2,3,4 \
  --output_root ./sarem_results
```

Collect metrics from:

```text
sarem_results/{dataset}/sarem_score/seed_{seed}/metrics.json
```

Report mean and standard deviation over five random seeds.

## 11. Reproducibility Checklist

Before reporting results, confirm that:

```text
1. All datasets are stored under dataset/.
2. The same seeds are used for all compared methods.
3. Anomaly labels are not used during training.
4. node_scores.csv and metrics.json are generated for each run.
5. Mean and standard deviation are reported over five seeds.
6. CleanView construction does not use anomaly labels.
7. The exact command line and environment are recorded.
```

## 12. Troubleshooting

Missing dataset file:

```bash
ls ./dataset
```

Windows PowerShell:

```powershell
dir .\dataset
```

Missing Python package:

```bash
pip install networkx
```

CUDA out of memory:

```bash
export CUDA_VISIBLE_DEVICES=""
```

Windows PowerShell:

```powershell
$env:CUDA_VISIBLE_DEVICES=""
```

Script arguments do not match:

```bash
python run_sarem_pack.py --help
python run_paper_experiments_sarem.py --help
```

## 13. Citation

If you use this repository, please cite the corresponding CleanViewGAD / SAREM-GAD / RGRGAD paper when it becomes available.
