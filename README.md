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

## 模型架构

- **BertEncoder**: 基于 RoBERTa 的文本特征编码
- **MoQformer**: 基于 Mixture-of-Experts 的 Q-Former，对各模态特征进行提取与压缩
- **FTREncoder**: 模态不变与模态特异性表征学习（参考 MISA）
- **Transformer Fusion**: 跨模态融合模块
- **对比学习**: 基于 AccCon 的对比损失，增强表征判别能力

## 支持的数据集

| 数据集 | 参数 `--dataset_name` |
|--------|----------------------|
| CMU-MOSEI | `MOSEI` |
| CMU-MOSI | `MOSI` |
| CH-SIMS | `SIMS` |
| URFUNNY | `URFUNNY` |

## 📦 预训练权重与数据集

预训练权重和数据集下载地址：[Google Drive](https://drive.google.com/drive/folders/1KMPMUE9hff8iyZThOS6AKJnVsImTp5zn?usp=drive_link)

下载后将权重文件放置于 `best_results/` 对应目录下。

## 环境依赖

- Python 3.9.13
- PyTorch 2.3.0+cu121
- Transformers 4.49.0
- NumPy 1.26.3

安装依赖：
```bash
pip install torch==2.3.0 transformers==4.49.0 numpy==1.26.3
```

## 项目结构

```
├── train.py              # 训练入口
├── test.py               # 测试入口
├── get_opts.py           # 参数配置
├── models/
│   ├── fine_model.py     # FINE 模型定义
│   ├── fine_mosei_model.py
│   ├── fine_urfunny_model.py
│   └── networks/
│       ├── Qformer.py    # MoQformer 模块
│       ├── FTREncoder.py # 模态表征编码
│       ├── MoE.py        # Mixture-of-Experts
│       ├── queue.py      # 特征队列
│       └── ...
├── dataloader/           # 数据加载
├── scripts/              # 训练/测试脚本
├── utils/                # 工具函数
└── visualize/            # 可视化
```

## 使用方法

### 训练

以 CMU-MOSEI 为例：
```bash
bash scripts/train_mosei_best.sh <run_idx> <gpu_id> [description]
```

示例：
```bash
bash scripts/train_mosei_best.sh 1 0 "baseline"
```

其他数据集：
```bash
bash scripts/train_mosi_best.sh <run_idx> <gpu_id> [description]
bash scripts/train_sims_best.sh <run_idx> <gpu_id> [description]
bash scripts/train_urfunny_best.sh <run_idx> <gpu_id> [description]
```

### 测试

```bash
bash scripts/test_mosei.sh <run_idx> <gpu_id>
```

测试前需确保 `best_results/` 下已有预训练权重。

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--embd_size_t/v/a` | 256 | 文本/视觉/声学嵌入维度 |
| `--num_experts` | 8 | MoE 专家数量 |
| `--topk_ratio` | 0.75 | MoE top-k 选择比例 |
| `--query_token_len` | 8 | Q-Former query token 数量 |
| `--batch_size` | 64 | 训练 batch size |
| `--lr` | 3e-5 | 学习率 |
| `--bert_pretrain` | roberta-base | 预训练语言模型 |
| `--cl_weight` | 3.0 | 对比学习损失权重 |
| `--temperature` | 0.05 | 对比学习温度参数 |

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
