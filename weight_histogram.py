import os
import timm
import torch
import matplotlib.pyplot as plt

def save_params_histograms(model, bins=50, grid_rows=3, grid_cols=3, save_dir="./histograms/params"):
    print("Creating directory for saving histograms: ", save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    param_names = []
    param_values = []
    for name, param in model.named_parameters():
        param_names.append(name)
        param_values.append(param.detach().cpu().numpy().ravel())
        
    total_params = len(param_values)
    group_size = grid_rows * grid_cols
    
    print(f"Plotting parameter histograms in groups of {group_size}...")
    
    for start_idx in range(0, total_params, group_size):
        end_idx = min(start_idx + group_size, total_params)
        chunk_names = param_names[start_idx:end_idx]
        chunk_values = param_values[start_idx:end_idx]
        
        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4*grid_cols, 3*grid_rows))
        axes = axes.flatten()
        
        for ax_idx, (name, values) in enumerate(zip(chunk_names, chunk_values)):
            ax = axes[ax_idx]
            ax.hist(values, bins=bins, color='blue', alpha=0.7)
            ax.set_title(f"{name}", fontsize=9)
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)
        
        for empty_ax_idx in range(len(chunk_names), group_size):
            fig.delaxes(axes[empty_ax_idx])
        
        fig.tight_layout()
        
        group_id = start_idx // group_size
        save_path = os.path.join(save_dir, f"params_histogram_{group_id}.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    
    print("Done plotting and saving parameter histograms.\n")

def activation_hook(layer_name, activations_dict):
    def hook(module, input, output):
        # Flatten and store activation values in the dictionary
        activations_dict[layer_name] = output.detach().cpu().numpy().ravel()
    return hook

def save_activations_histograms(model, input_tensor, bins=50, grid_rows=3, grid_cols=3, save_dir="./histograms/activations"):
    print("Creating directory for activation histograms: ", save_dir)
    os.makedirs(save_dir, exist_ok=True)

    activations = {}

    for name, module in model.named_modules():
        module.register_forward_hook(activation_hook(name, activations))

    print("Running forward pass for activation collection...")
    with torch.no_grad():
        _ = model(input_tensor)

    layer_names = list(activations.keys())
    layer_acts = list(activations.values())
    total_layers = len(layer_acts)
    group_size = grid_rows * grid_cols

    print(f"Plotting activation histograms in groups of {group_size}...")
    for start_idx in range(0, total_layers, group_size):
        end_idx = min(start_idx + group_size, total_layers)
        chunk_names = layer_names[start_idx:end_idx]
        chunk_values = layer_acts[start_idx:end_idx]

        fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4*grid_cols, 3*grid_rows))
        axes = axes.flatten()

        for ax_idx, (name, values) in enumerate(zip(chunk_names, chunk_values)):
            ax = axes[ax_idx]
            ax.hist(values, bins=bins, color='green', alpha=0.7)
            ax.set_title(f"{name}", fontsize=9)
            ax.set_xlabel("Value", fontsize=8)
            ax.set_ylabel("Frequency", fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=8)

        for empty_ax_idx in range(len(chunk_names), group_size):
            fig.delaxes(axes[empty_ax_idx])

        fig.tight_layout()

        group_id = start_idx // group_size
        save_path = os.path.join(save_dir, f"activations_hist_group_{group_id}.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)

    print("Done plotting and saving activation histograms.\n")


def main():
    # ---------------------
    # 1. Create and load the TIMM model
    # ---------------------
    model_name = "mobilenetv3_small_100"
    print(f"Loading TIMM model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    # ---------------------
    # 2. Plot Parameter Histograms
    # ---------------------
    save_params_histograms(model, bins=50, grid_rows=3, grid_cols=3, save_dir="./histograms/params")

    # ---------------------
    # 3. Define an input tensor for activations
    # ---------------------
    # For classification models, typically expect [batch_size, 3, height, width].
    # For MobileNetV3, usually 3x224x224 is standard.
    dummy_input = torch.randn(1, 3, 224, 224)

    # ---------------------
    # 4. Plot Activation Histograms
    # ---------------------
    save_activations_histograms(model, dummy_input, bins=50, grid_rows=3, grid_cols=3, save_dir="./histograms/activations")


if __name__ == "__main__":
    main()