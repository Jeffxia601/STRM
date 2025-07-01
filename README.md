# Efficient Pretraining & Fine-Tuning of a Large Spatio-Temporal Reasoning Model (STRM, ViT-1.2B)
​**Jan 2024 - May 2024**​ | `PyTorch` • `FSDP` • `Parallel Adapter` • `Structured Pruning` • `TensorRT-LLM` | Python 3.11.5  

### Large-Scale Spatio-Temporal Modeling for Trajectory Intelligence
This project focuses on the ​efficient development and deployment​ of a large-scale Vision Transformer (ViT-1.2B) adapted for spatio-temporal trajectory data analysis, powering two key taxi operation predictions:  
- Driver Identification (20-class Classification)​
- ​Occupancy Status Detection (Binary Classification)​

Key contributions lie in ​significantly optimizing​ the computationally intensive processes of ​pretraining, fine-tuning, and inference​ for the spatio-temporally adapted model.

## Technical Contributions & Optimization Achievements
| ​**Area**​               | ​**Optimization Techniques**​                          |
|-----------------------------|------------------------------------------|
| ​**Large Model Pretraining (ViT-1.2B)​**​           | ​Spatio-Temporal Adaptation: Modified ViT architecture (Patch Embedding, Self-Attention) for 3D structure (Time × Region × Feature). ​Distributed Training: Combined Fully Sharded Data Parallel (FSDP) with ​Gradient Checkpointing. |
| ​**Efficient Downstream Fine-Tuning**​       | Parameter-Efficient Tuning: Integrated ​Parallel Adapters​ (r=32, inserted after FFN blocks). |
| ​**Model Compression & Accelerated Inference**​  | Applied ​Structured Channel Pruning. Deployed optimized engine via ​TensorRT-LLM. |

## Getting Started (Quick Setup)
1. Place datasets under `data/pretrain/` and `data/finetune/`.
2. Adjust any hyperparameters in `src/config.py`.
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run pretraining:
    ```bash
    python src/train/pretrain.py
    ```
5. Run fine-tuning (choose task)
    ```bash
    python src/train/finetune.py --task driver
    python src/train/finetune.py --task status
    ```
6. Prune & build TensorRT engine:
    ```bash
    python src/prune_infer.py
    ```
