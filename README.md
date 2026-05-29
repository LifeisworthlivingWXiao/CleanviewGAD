# CleanViewGAD

Minimal reproduction guide for CleanViewGAD / SAREM-GAD-style unsupervised graph anomaly detection.

## 1. Environment

```bash
conda create -n cleanviewgad python=3.9 -y
conda activate cleanviewgad

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Dependencies
pip install -r requirements.txt
pip install numpy scipy pandas scikit-learn matplotlib networkx tqdm h5py
```

For CPU-only environments:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
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

## 3. Run One Dataset

```bash
python run_sarem_pack.py --dataset cora --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
```

Other datasets:

```bash
python run_sarem_pack.py --dataset citeseer --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
python run_sarem_pack.py --dataset pubmed --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
python run_sarem_pack.py --dataset BlogCatalog --data_dir ./dataset --variant sarem_score --seed 0 --output_root ./sarem_results
```

## 4. Run All Main Datasets

```bash
python run_all_sarem_datasets.py \
  --datasets cora,citeseer,pubmed,BlogCatalog \
  --data_dir ./dataset \
  --variant sarem_score \
  --seeds 0,1,2,3,4 \
  --output_root ./sarem_results
```

If the batch script is unavailable, run `run_sarem_pack.py` separately for each dataset and seed.

## 5. Paper-Level Reproduction

```bash
python run_paper_experiments_sarem.py \
  --data_dir ./dataset \
  --datasets cora,citeseer,pubmed,BlogCatalog \
  --variant sarem_score \
  --seeds 0,1,2,3,4 \
  --output_root ./sarem_results
```

## 6. Expected Outputs

Each run writes results to:

```text
sarem_results/{dataset}/{variant}/seed_{seed}/
```

Example:

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

Main files:

```text
metrics.json          ROC-AUC, PR-AUC, and other metrics.
node_scores.csv       Node-level anomaly scores.
clean_view_info.json  CleanView edge information.
roc_curve.csv         ROC curve points.
pr_curve.csv          Precision-Recall curve points.
```

## 7. Nature-Style Visualization

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

Windows example:

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

Expected visualization outputs:

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

## 8. Quick Check

```bash
python run_sarem_pack.py --help
python run_paper_experiments_sarem.py --help
```
