"""
Visualize convolutional filters for Part 1
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from models.my_net import MyCIFAR10Net
from models.vgg import VGG_A


def load_trained_model(model_class, weights_path, device='cpu'):
    """加载训练好的模型权重"""
    model = model_class()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def visualize_filters(model, layer_name='conv1', save_path='filters.png'):
    """可视化卷积核"""
    # 获取指定层的权重
    for name, param in model.named_parameters():
        if layer_name in name and 'weight' in name:
            weights = param.data.cpu().numpy()
            break
    else:
        print(f"Layer {layer_name} not found")
        return
    
    # weights shape: [out_channels, in_channels, height, width]
    n_filters = min(weights.shape[0], 64)  # 最多显示64个
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i in range(n_filters):
        # 取第一个输入通道的核
        filter_img = weights[i, 0, :, :]
        # 归一化到[0,1]
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
        
        axes[i].imshow(filter_img, cmap='gray')
        axes[i].set_title(f'Filter {i}')
        axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Visualization of {layer_name} filters', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Filter visualization saved to {save_path}")


def visualize_feature_maps(model, image, layer_name='conv1', save_path='feature_maps.png'):
    """可视化特征图"""
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())
    
    # 注册hook
    for name, layer in model.named_modules():
        if layer_name in name:
            handle = layer.register_forward_hook(hook_fn)
            break
    
    # 前向传播
    with torch.no_grad():
        _ = model(image.unsqueeze(0))
    
    handle.remove()
    
    if not activations:
        print(f"Layer {layer_name} not found")
        return
    
    feat_maps = activations[0][0]  # [C, H, W]
    n_maps = min(feat_maps.shape[0], 32)
    n_cols = 8
    n_rows = (n_maps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i in range(n_maps):
        fm = feat_maps[i, :, :]
        fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
        axes[i].imshow(fm, cmap='viridis')
        axes[i].set_title(f'Channel {i}')
        axes[i].axis('off')
    
    for i in range(n_maps, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Maps after {layer_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Feature map visualization saved to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载训练好的模型（需要先运行train.py）
    # 如果还没有训练好的模型，先创建一个随机初始化的模型演示
    print("Loading model (using random weights for demonstration)")
    model = MyCIFAR10Net().to(device)
    
    # 可视化conv1的卷积核
    visualize_filters(model, 'conv1', 'conv1_filters.png')
    
    # 可视化conv2的卷积核
    visualize_filters(model, 'conv2', 'conv2_filters.png')
    
    # 创建一个随机测试图像
    test_image = torch.randn(3, 32, 32).to(device)
    
    # 可视化conv1输出的特征图
    visualize_feature_maps(model, test_image, 'conv1', 'conv1_feature_maps.png')
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()