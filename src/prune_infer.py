import os
import torch
import torch.nn as nn
import torch.onnx
import tensorrt as trt
from torch_pruning import DependencyGraph

from src.config import CONFIG
from src.utils import set_seed, init_logger


def prune_model(model, example_input):
    """Structured pruning on Conv3d layers based on DependencyGraph."""
    logger = init_logger('prune')
    dg = DependencyGraph().build_dependency(model, example_inputs=example_input)
    prune_ratio = CONFIG['prune']['prune_ratio']
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            plan = dg.get_pruning_plan(m, 'weight', lambda x: int(x.shape[0] * (1 - prune_ratio)))
            plan.exec()
            logger.info(f"Pruned {m} channels by {prune_ratio * 100}%")
    return model


def convert_to_trt(model, engine_path, example_input):
    """Export to ONNX and build TensorRT engine."""
    logger = init_logger('infer')
    onnx_path = 'model.onnx'
    torch.onnx.export(model, example_input, onnx_path,
                      input_names=['input'], output_names=['output'], opset_version=13)
    logger.info("Exported ONNX model")

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30
    engine = builder.build_engine(network, config)

    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    logger.info(f"Saved TensorRT engine to {engine_path}")
    return engine


def main():
    set_seed()
    cfg = CONFIG
    paths = cfg['paths']
    engine_path = cfg['infer']['tensorrt_engine_path']

    # Load pre-trained PyTorch model
    ckpt_path = os.path.join(paths['save_dir'], 'pretrain_epoch0.pth')
    checkpoint = torch.load(ckpt_path)
    model = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    example_input = torch.randn(1, 3, 8, 128, 128).cuda()
    model = prune_model(model, example_input)
    convert_to_trt(model, engine_path, example_input)


if __name__ == '__main__':
    main()
