<h1 align="center">
  FINE: Factorized Multimodal Sentiment Analysis via Mutual Information Estimation
</h1>

<p align="center">
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37176">📘 Paper</a>
</p>

## 📢 News

[2025.11.08] 🎉🎉🎉 Our FINE has been accepted at **AAAI 2026 Oral**!

## 🚀 Overview

This repository contains the **official implementation** of the paper:  
**FINE: Factorized Multimodal Sentiment Analysis via Mutual Information Estimation**.

FINE proposes a novel **factorized framework** for multimodal sentiment analysis, which decomposes multimodal representations into **modality-specific** and **shared components** through **mutual information estimation**.  
This approach enhances interpretability and robustness across diverse multimodal contexts.

## Model Architecture

- **BertEncoder**: Text feature encoding based on RoBERTa
- **MoQformer**: Mixture-of-Experts Q-Former for modality feature extraction and compression
- **FTREncoder**: Modality-invariant and modality-specific representation learning (refer to MISA)
- **Transformer Fusion**: Cross-modal fusion module
- **Contrastive Learning**: AccCon-based contrastive loss for enhanced discriminative representations

## Supported Datasets

| Dataset | `--dataset_name` |
|---------|-----------------|
| CMU-MOSEI | `MOSEI` |
| CMU-MOSI | `MOSI` |
| CH-SIMS | `SIMS` |
| URFUNNY | `URFUNNY` |

## 📦 Pretrained Weights & Datasets

Pretrained weights and datasets are available at: [Google Drive](https://drive.google.com/drive/folders/1KMPMUE9hff8iyZThOS6AKJnVsImTp5zn?usp=drive_link)

Please place the downloaded weights under the corresponding directory in `best_results/`.

## Requirements

- Python 3.9.13
- PyTorch 2.3.0+cu121
- Transformers 4.49.0
- NumPy 1.26.3

Install dependencies:
```bash
pip install torch==2.3.0 transformers==4.49.0 numpy==1.26.3
```

## Project Structure

```
├── train.py              # Training entry point
├── test.py               # Testing entry point
├── get_opts.py           # Configuration options
├── models/
│   ├── fine_model.py     # FINE model definition
│   ├── fine_mosei_model.py
│   ├── fine_urfunny_model.py
│   └── networks/
│       ├── Qformer.py    # MoQformer module
│       ├── FTREncoder.py # Modality representation encoder
│       ├── MoE.py        # Mixture-of-Experts
│       ├── queue.py      # Feature queue
│       └── ...
├── dataloader/           # Data loading
├── scripts/              # Training/testing scripts
├── utils/                # Utility functions
└── visualize/            # Visualization
```

## Usage

### Training

For CMU-MOSEI:
```bash
bash scripts/train_mosei_best.sh <run_idx> <gpu_id> [description]
```

Example:
```bash
bash scripts/train_mosei_best.sh 1 0 "baseline"
```

For other datasets:
```bash
bash scripts/train_mosi_best.sh <run_idx> <gpu_id> [description]
bash scripts/train_sims_best.sh <run_idx> <gpu_id> [description]
bash scripts/train_urfunny_best.sh <run_idx> <gpu_id> [description]
```

### Testing

```bash
bash scripts/test_mosei.sh <run_idx> <gpu_id>
```

Make sure pretrained weights are placed under `best_results/` before testing.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embd_size_t/v/a` | 256 | Text/visual/acoustic embedding dimension |
| `--num_experts` | 8 | Number of MoE experts |
| `--topk_ratio` | 0.75 | MoE top-k selection ratio |
| `--query_token_len` | 8 | Number of Q-Former query tokens |
| `--batch_size` | 64 | Training batch size |
| `--lr` | 3e-5 | Learning rate |
| `--bert_pretrain` | roberta-base | Pretrained language model |
| `--cl_weight` | 3.0 | Contrastive learning loss weight |
| `--temperature` | 0.05 | Contrastive learning temperature |

## ✒️ Citation

If you find our work helpful for your research, please consider giving a star and citation.

```tex
@misc{liu2025finefactorizedmultimodalsentiment,
      title={FINE: Factorized multimodal sentiment analysis via mutual INformation Estimation}, 
      author={Yadong Liu and Shangfei Wang},
      year={2025},
      eprint={2511.20167},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://ojs.aaai.org/index.php/AAAI/article/view/37176}, 
}
```
