import os
import argparse

import torch
import torch.nn as nn
import timm
import matplotlib.pyplot as plt
from timm.layers.activations import HardSwish as TimmHardSwish
# For FX Graph Mode PTQ
import torch.ao.quantization
from torch.ao.quantization.quantize_fx import (
    prepare_fx,
    convert_fx,
)
from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping

def compare_params_histograms(float_model, quant_model, bins=50, rows=3, save_dir="./histograms_params_compare"):
    print("Creating directory for parameter-comparison histograms: ", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    float_params = {}
    for name, param in float_model.named_parameters():
        float_params[name] = param.detach().cpu().numpy().ravel()
    
    quant_params = {}
    for name, param in quant_model.named_parameters():
        if param.is_quantized:
            quant_params[name] = param.dequantize().cpu().numpy().ravel()
        else:
            quant_params[name] = param.detach().cpu().numpy().ravel()
    
    common_names = [n for n in float_params.keys() if n in quant_params]
    common_names.sort()
    
    group_size = rows
    for start_idx in range(0, len(common_names), group_size):
        end_idx = min(start_idx + group_size, len(common_names))
        chunk_names = common_names[start_idx:end_idx]
        
        fig, axes = plt.subplots(len(chunk_names), 2, figsize=(10, 3 * len(chunk_names)))
        if len(chunk_names) == 1:
            axes = [axes]
        
        for row_idx, param_name in enumerate(chunk_names):
            float_vals = float_params[param_name]
            quant_vals = quant_params[param_name]
            
            ax_float = axes[row_idx][0]
            ax_quant = axes[row_idx][1]
            
            ax_float.hist(float_vals, bins=bins, color='blue', alpha=0.7)
            ax_float.set_title(f"Float: {param_name}", fontsize=9)
            ax_float.tick_params(axis='both', which='major', labelsize=8)
            
            ax_quant.hist(quant_vals, bins=bins, color='red', alpha=0.7)
            ax_quant.set_title(f"Quant: {param_name}", fontsize=9)
            ax_quant.tick_params(axis='both', which='major', labelsize=8)
        
        fig.tight_layout()
        group_id = start_idx // group_size
        save_path = os.path.join(save_dir, f"params_compare_{group_id}.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    
    print("Done comparing parameters.\n")

def compare_activations_histograms(float_model, quant_model, input_tensor, bins=50, rows=3, save_dir="./histograms_activations_compare"):
    print("Creating directory for activation-comparison histograms: ", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    for m in float_model.modules():
        if hasattr(m, "_forward_hooks"):
            m._forward_hooks.clear()
    for m in quant_model.modules():
        if hasattr(m, "_forward_hooks"):
            m._forward_hooks.clear()
    
    float_acts = {}
    quant_acts = {}
    
    def float_hook(layer_name):
        def hook_fn(module, inp, out):
            if hasattr(out, 'int_repr') and out.is_quantized:
                float_acts[layer_name] = out.int_repr().cpu().numpy().ravel()
            else:
                float_acts[layer_name] = out.detach().cpu().numpy().ravel()
        return hook_fn
    
    def quant_hook(layer_name):
        def hook_fn(module, inp, out):
            if hasattr(out, 'int_repr') and out.is_quantized:
                quant_acts[layer_name] = out.int_repr().cpu().numpy().ravel()
            else:
                quant_acts[layer_name] = out.detach().cpu().numpy().ravel()
        return hook_fn
    
    for name, module in float_model.named_modules():
        module.register_forward_hook(float_hook(name))
    
    for name, module in quant_model.named_modules():
        module.register_forward_hook(quant_hook(name))
        
    with torch.no_grad():
        _ = float_model(input_tensor)
        _ = quant_model(input_tensor)
    
    common_names = [n for n in float_acts.keys() if n in quant_acts]
    common_names.sort()
    
    group_size = rows
    for start_idx in range(0, len(common_names), group_size):
        end_idx = min(start_idx + group_size, len(common_names))
        chunk_names = common_names[start_idx:end_idx]
        
        fig, axes = plt.subplots(len(chunk_names), 2, figsize=(10, 3 * len(chunk_names)))
        if len(chunk_names) == 1:
            axes = [axes]
        
        for row_idx, layer_name in enumerate(chunk_names):
            float_vals = float_acts[layer_name]
            quant_vals = quant_acts[layer_name]
            
            ax_float = axes[row_idx][0]
            ax_quant = axes[row_idx][1]
            
            ax_float.hist(float_vals, bins=bins, color='green', alpha=0.7)
            ax_float.set_title(f"Float: {layer_name}", fontsize=9)
            ax_float.tick_params(axis='both', which='major', labelsize=8)
            
            ax_quant.hist(quant_vals, bins=bins, color='purple', alpha=0.7)
            ax_quant.set_title(f"Quant: {layer_name}", fontsize=9)
            ax_quant.tick_params(axis='both', which='major', labelsize=8)
        
        fig.tight_layout()
        group_id = start_idx // group_size
        save_path = os.path.join(save_dir, f"activations_compare_{group_id}.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    print("Done comparing activations.\n")
    
def quantize_model_fx(model, example_data, backend="qnnpack"): # in case of x86, use "fbgemm"; in case of macOS, use "qnnpack"
    torch.backends.quantized.engine = backend
    qconfig_mapping = get_default_qconfig_mapping(backend)
    
    model_prepared = prepare_fx(model, qconfig_mapping, example_data)
    
    model_prepared.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model_prepared(example_data)
    
    model_quantized = convert_fx(model_prepared)
    model_quantized.eval()
    return model_quantized

def replace_hardswish(model):
    for name, module in model.named_children():
        if isinstance(module, (nn.Hardswish, nn.quantized.Hardswish, TimmHardSwish)):
            print(f"Replacing {name} (HardSwish) with ReLU.")
            setattr(model, name, nn.ReLU())
        else:
            replace_hardswish(module)
    return model

def main():
    parser = argparse.ArgumentParser(description="Compare Float vs Quantized model histograms.")
    parser.add_argument("--model_name", type=str, default="mobilenetv3_small_100",
                        help="Name of the TIMM model to load.")
    parser.add_argument("--quant_backend", type=str, default="qnnpack",
                        choices=["qnnpack", "fbgemm"],
                        help="Which quantization backend to use (CPU: 'fbgemm' or mobile-friendly: 'qnnpack').")
    parser.add_argument("--no_replace_hardswish", action="store_true",
                        help="If set, do NOT replace Hardswish with ReLU.")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size (height=width).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for dummy input.")
    args = parser.parse_args()
    
    print(f"Loading float (FP32) model: {args.model_name}")
    float_model = timm.create_model(args.model_name, pretrained=True)
    float_model.eval()
    
    if not args.no_replace_hardswish:
        float_model = replace_hardswish(float_model)
    
    example_data = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    
    print(f"Quantizing model using backend: {args.quant_backend}")
    quant_model = quantize_model_fx(float_model, example_data, backend=args.quant_backend)
    
    print("Comparing float vs quant model params ...")
    compare_params_histograms(float_model, quant_model, bins=50, rows=3, save_dir="./histograms_params_compare")
    
    print("Comparing float vs quant model activations ...")
    compare_activations_histograms(float_model, quant_model, example_data, bins=50, rows=3, save_dir="./histograms_activations_compare")
    
    # print("Tracing quantized model for TorchScript...")
    # with torch.no_grad():
    #     traced_quantized = torch.jit.trace(quant_model, example_data)
    #
    # print("Converting to CoreML (INT8) ...")
    # mlmodel = ct.convert(
    #     traced_quantized,
    #     source="pytorch",
    #     inputs=[ct.TensorType(shape=example_data.shape)],
    #     convert_to="mlprogram",
    #     compute_units=ct.ComputeUnit.CPU_ONLY,
    #     minimum_deployment_target=ct.target.iOS17,
    # )
    # mlmodel_path = "mobilenetv3_small_100_int8.mlpackage"
    # mlmodel.save(mlmodel_path)
    # print(f"CoreML model saved at: {mlmodel_path}")
    
    print("Done.\n")

if __name__ == "__main__":
    main()