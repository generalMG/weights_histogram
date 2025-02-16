import timm
import torch
import matplotlib.pyplot as plt

def plot_params_histograms(model, bins=50):
    print("Plotting weight histograms...")
    
    for name, param in model.named_parameters():
        param_data = param.detach().cpu().numpy().ravel()
        
        plt.figure(figsize=(6, 4))
        plt.hist(param_data, bins=bins, color='blue', alpha=0.7)
        plt.title(f'Parameter histogram: {name}')
        plt.xlabel('Parameter value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()
        
    print("Done plotting weight histograms.")

def activation_hook(layer_name, activations_dict):
    def hook(module, input, output):
        # Flatten and store activation values in the dictionary
        activations_dict[layer_name] = output.detach().cpu().numpy().ravel()
    return hook

def plot_activations_histograms(model, input_tensor, bins=50):
    print("Registering forward hooks for activation histograms...")
    activations = {}

    for name, module in model.named_modules():
        # For demonstration purposes, the code hooks all activation layers:
        module.register_forward_hook(activation_hook(name, activations))
    
    print("Running forward pass...")
    with torch.no_grad():
        _ = model(input_tensor)

    print("Plotting activation histograms...")
    for layer_name, act_values in activations.items():
        plt.figure(figsize=(6, 4))
        plt.hist(act_values, bins=bins, color="green", alpha=0.7)
        plt.title(f"Activation Histogram: {layer_name}")
        plt.xlabel("Activation Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    print("Done plotting activation histograms.\n")

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
    plot_params_histograms(model, bins=50)

    # ---------------------
    # 3. Define an input tensor for activations
    # ---------------------
    # For classification models, typically expect [batch_size, 3, height, width].
    # For MobileNetV3, usually 3x224x224 is standard.
    dummy_input = torch.randn(1, 3, 224, 224)

    # ---------------------
    # 4. Plot Activation Histograms
    # ---------------------
    plot_activations_histograms(model, dummy_input, bins=50)


if __name__ == "__main__":
    main()