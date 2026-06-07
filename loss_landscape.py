"""
Loss Landscape visualization for Part 2
绘制不同学习率下的loss变化区域图
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm


def set_random_seeds(seed=42):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_loss_for_learning_rate(model_class, lr, train_loader, epochs=10, device='cpu'):
    """对给定的学习率训练模型并返回每个step的loss"""
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    step_losses = []
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            step_losses.append(loss.item())
    
    return step_losses


def plot_loss_landscape(losses_dict, save_path='loss_landscape.png'):
    """绘制loss landscape区域图"""
    if save_path is None:
        return
    
    plt.figure(figsize=(12, 6))
    
    # 找出所有loss的min和max范围
    all_losses = []
    for losses in losses_dict.values():
        all_losses.extend(losses)
    
    max_loss = min(max(all_losses), 5)
    min_loss = min(all_losses)
    
    for lr, losses in losses_dict.items():
        x = np.linspace(0, 1, len(losses))
        plt.plot(x, losses, label=f'LR = {lr}', alpha=0.7)
    
    plt.xlabel('Training Progress (normalized)')
    plt.ylabel('Loss')
    plt.title('Loss Landscape: Different Learning Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(min_loss - 0.5, max_loss + 0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss landscape saved to {save_path}")


def plot_loss_landscape_with_fill(losses_dict, save_path='loss_landscape_filled.png'):
    """绘制带填充区域的loss landscape（类似论文中的Figure 3）"""
    if save_path is None:
        return
    
    plt.figure(figsize=(12, 6))
    
    # 将所有loss对齐到相同的step数
    max_len = max(len(losses) for losses in losses_dict.values())
    aligned_losses = {}
    
    for lr, losses in losses_dict.items():
        x_old = np.linspace(0, 1, len(losses))
        x_new = np.linspace(0, 1, max_len)
        aligned = np.interp(x_new, x_old, losses)
        aligned_losses[lr] = aligned
    
    # 计算每个step的最大值和最小值
    all_aligned = np.array(list(aligned_losses.values()))
    min_curve = np.min(all_aligned, axis=0)
    max_curve = np.max(all_aligned, axis=0)
    
    # 绘制区域
    x = np.arange(max_len)
    plt.fill_between(x, min_curve, max_curve, alpha=0.3, color='blue', label='Loss Range')
    plt.plot(x, min_curve, 'b--', alpha=0.7, label='Min Loss')
    plt.plot(x, max_curve, 'r--', alpha=0.7, label='Max Loss')
    
    # 绘制平均线
    mean_curve = np.mean(all_aligned, axis=0)
    plt.plot(x, mean_curve, 'g-', alpha=0.8, label='Mean Loss')
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Loss Landscape with Fill Between (Different Learning Rates)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss landscape (filled) saved to {save_path}")


def main():
    set_random_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 使用部分数据加速训练（5000张图）
    train_loader = get_cifar_loader(train=True, batch_size=64, n_items=5000)
    
    # 测试不同的学习率
    learning_rates = [1e-3, 2e-3, 5e-4, 1e-4]
    epochs_per_lr = 15
    
    print("\n" + "="*50)
    print("Generating Loss Landscape for VGG_A (without BN)")
    print("="*50)
    
    losses_no_bn = {}
    for lr in learning_rates:
        print(f"\nTraining with LR = {lr}")
        losses = get_loss_for_learning_rate(VGG_A, lr, train_loader, epochs_per_lr, device)
        losses_no_bn[lr] = losses
        print(f"  Recorded {len(losses)} steps")
    
    print("\n" + "="*50)
    print("Generating Loss Landscape for VGG_A_BatchNorm (with BN)")
    print("="*50)
    
    losses_with_bn = {}
    for lr in learning_rates:
        print(f"\nTraining with LR = {lr}")
        losses = get_loss_for_learning_rate(VGG_A_BatchNorm, lr, train_loader, epochs_per_lr, device)
        losses_with_bn[lr] = losses
        print(f"  Recorded {len(losses)} steps")
    
    # 绘制结果
    print("\n" + "="*50)
    print("Plotting Results")
    print("="*50)
    
    # 无BN的loss landscape
    plot_loss_landscape(losses_no_bn, 'loss_landscape_no_bn.png')
    plot_loss_landscape_with_fill(losses_no_bn, 'loss_landscape_filled_no_bn.png')
    
    # 有BN的loss landscape
    plot_loss_landscape(losses_with_bn, 'loss_landscape_with_bn.png')
    plot_loss_landscape_with_fill(losses_with_bn, 'loss_landscape_filled_with_bn.png')
    
    # 对比图（并排显示）
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左边：无BN
    max_len_no_bn = max(len(losses) for losses in losses_no_bn.values())
    aligned_no_bn = []
    for losses in losses_no_bn.values():
        x_old = np.linspace(0, 1, len(losses))
        x_new = np.linspace(0, 1, max_len_no_bn)
        aligned_no_bn.append(np.interp(x_new, x_old, losses))
    all_aligned_no_bn = np.array(aligned_no_bn)
    min_no_bn = np.min(all_aligned_no_bn, axis=0)
    max_no_bn = np.max(all_aligned_no_bn, axis=0)
    
    axes[0].fill_between(np.arange(max_len_no_bn), min_no_bn, max_no_bn, alpha=0.3, color='blue')
    axes[0].plot(min_no_bn, 'b--', alpha=0.7)
    axes[0].plot(max_no_bn, 'r--', alpha=0.7)
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('VGG_A (No BN) - Loss Landscape')
    axes[0].grid(True, alpha=0.3)
    
    # 右边：有BN
    max_len_with_bn = max(len(losses) for losses in losses_with_bn.values())
    aligned_with_bn = []
    for losses in losses_with_bn.values():
        x_old = np.linspace(0, 1, len(losses))
        x_new = np.linspace(0, 1, max_len_with_bn)
        aligned_with_bn.append(np.interp(x_new, x_old, losses))
    all_aligned_with_bn = np.array(aligned_with_bn)
    min_with_bn = np.min(all_aligned_with_bn, axis=0)
    max_with_bn = np.max(all_aligned_with_bn, axis=0)
    
    axes[1].fill_between(np.arange(max_len_with_bn), min_with_bn, max_with_bn, alpha=0.3, color='blue')
    axes[1].plot(min_with_bn, 'b--', alpha=0.7)
    axes[1].plot(max_with_bn, 'r--', alpha=0.7)
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('VGG_A (With BN) - Loss Landscape')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_landscape_comparison.png', dpi=150)
    plt.close()
    
    print("\nAll plots saved!")
    print("- loss_landscape_no_bn.png")
    print("- loss_landscape_filled_no_bn.png")
    print("- loss_landscape_with_bn.png")
    print("- loss_landscape_filled_with_bn.png")
    print("- loss_landscape_comparison.png")


if __name__ == '__main__':
    main()