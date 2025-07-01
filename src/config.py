import os

# Project root dir
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data and checkpoint dir
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# Global settings
CONFIG = {
    'paths': {
        'pretrain_data': os.path.join(DATA_DIR, 'pretrain'),
        'finetune_data': os.path.join(DATA_DIR, 'finetune'),
        'save_dir': CHECKPOINT_DIR,
    },
    'pretrain': {
        'batch_size': 16,
        'epochs': 100,
        'learning_rate': 1e-4,
        'fsdp': True,
        'grad_checkpoint': True,
    },
    'finetune': {
        'batch_size': 32,
        'epochs': 20,
        'learning_rate': 5e-5,
        'adapter_rank': 32,
    },
    'prune': {
        'prune_ratio': 0.3,
    },
    'infer': {
        'tensorrt_engine_path': os.path.join(PROJECT_ROOT, 'models', 'engine.trt'),
    },
}
