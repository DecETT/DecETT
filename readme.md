# DecETT: Accurate App Fingerprinting under Encrypted Tunnels via Dual Decouple-based Semantic Enhancement

Welcome to the repository of DecETT, a novel App Fingerprinting (AF) model for encrypted tunnel traffic. 

Since this paper has not yet been accepted, we provide an evaluation demo for interested reviewers to validate this work, including a trained model and part of our dataset.

## Requirements
Python 3.8.16

Pytorch 1.12.1+cu113

numpy 1.24.2


## Data
Part of our dataset is released for evaluation. The data is stored at  `saved_data/v2ray_corr/data.pkl`.

## Usage

```bash run.sh```

Before running run.sh, you should replace the `PHTHON_PATH` and `MAIN_FILE` as your own Python interpreter path and file path, respectively. 

## Results
The evaluation results can be found in `saved_results`. We provide an evaluation result in `saved_results/24-10-21_16-16-45_DRL_v2ray_corr_test` as an example.
