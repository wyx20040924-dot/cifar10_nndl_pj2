"""
Main training script for Part 1
训练不同配置的网络并记录结果
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.my_net import (
    MyCIFAR10Net, 
    MyCIFAR10NetNoBN, 
    MyCIFAR10NetDifferentChannels,
    MyCIFAR10NetLeakyReLU
)
from utils.nn import count_parameters


def set_random_seeds(seed=42):
    """设置随机种子保证可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_accuracy(model, loader, device):
    """计算准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)


def train_model(model, train_loader, val_loader, optimizer, criterion, 
                device, epochs=50, model_name='model'):
    """完整训练流程"""
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    print(f"\nTraining {model_name} on {device}")
    print(f"Parameters: {count_parameters(model):,}")
    
    for epoch in tqdm(range(epochs), desc=model_name):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # 验证
        val_acc = get_accuracy(model, val_loader, device)
        val_accuracies.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return train_losses, val_accuracies, best_val_acc


def plot_results(results_dict, save_path='training_results.png'):
    """绘制训练曲线对比"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Loss曲线
    for name, results in results_dict.items():
        axes[0].plot(results['losses'], label=f"{name} (Acc: {results['best_acc']:.1f}%)")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True)
    
    # 准确率曲线
    for name, results in results_dict.items():
        axes[1].plot(results['accuracies'], label=f"{name} (Acc: {results['best_acc']:.1f}%)")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy (%)')
    axes[1].set_title('Validation Accuracy Comparison')
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Figure saved to {save_path}")


def main():
    # 设置
    set_random_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    batch_size = 128
    train_loader = get_cifar_loader(train=True, batch_size=batch_size)
    val_loader = get_cifar_loader(train=False, batch_size=batch_size)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    criterion = nn.CrossEntropyLoss()
    epochs = 50
    
    results = {}
    
    # ========== 实验1: 有BN + 中等channel + Dropout + Adam (baseline) ==========
    print("\n" + "="*60)
    print("Experiment 1: With BN + Medium Channels + Dropout + Adam")
    print("="*60)
    
    model_bn = MyCIFAR10Net().to(device)
    optimizer_bn = Adam(model_bn.parameters(), lr=0.001, weight_decay=0)
    losses_bn, accs_bn, best_bn = train_model(
        model_bn, train_loader, val_loader, optimizer_bn, criterion, 
        device, epochs, "model_with_bn"
    )
    results['With BN (Baseline)'] = {'losses': losses_bn, 'accuracies': accs_bn, 'best_acc': best_bn}
    
    # ========== 实验2: 无BN + 中等channel + 无Dropout + Adam ==========
    print("\n" + "="*60)
    print("Experiment 2: Without BN + Medium Channels + No Dropout + Adam")
    print("="*60)
    
    model_no_bn = MyCIFAR10NetNoBN().to(device)
    optimizer_no_bn = Adam(model_no_bn.parameters(), lr=0.001, weight_decay=0)
    losses_no_bn, accs_no_bn, best_no_bn = train_model(
        model_no_bn, train_loader, val_loader, optimizer_no_bn, criterion, 
        device, epochs, "model_without_bn"
    )
    results['Without BN'] = {'losses': losses_no_bn, 'accuracies': accs_no_bn, 'best_acc': best_no_bn}
    
    # ========== 实验3: 有BN + 中等channel + Dropout + Adam + Weight Decay ==========
    print("\n" + "="*60)
    print("Experiment 3: With BN + Medium Channels + Dropout + Adam + Weight Decay")
    print("="*60)
    
    model_bn_wd = MyCIFAR10Net().to(device)
    optimizer_bn_wd = Adam(model_bn_wd.parameters(), lr=0.001, weight_decay=5e-4)
    losses_bn_wd, accs_bn_wd, best_bn_wd = train_model(
        model_bn_wd, train_loader, val_loader, optimizer_bn_wd, criterion, 
        device, epochs, "model_with_bn_weight_decay"
    )
    results['With BN + Weight Decay'] = {'losses': losses_bn_wd, 'accuracies': accs_bn_wd, 'best_acc': best_bn_wd}
    
    # ========== 实验4: 有BN + 小channel + Dropout + Adam ==========
    print("\n" + "="*60)
    print("Experiment 4: With BN + Small Channels + Dropout + Adam")
    print("="*60)
    
    model_small = MyCIFAR10NetDifferentChannels(channels=[16, 32, 64]).to(device)
    optimizer_small = Adam(model_small.parameters(), lr=0.001, weight_decay=0)
    losses_small, accs_small, best_small = train_model(
        model_small, train_loader, val_loader, optimizer_small, criterion, 
        device, epochs, "model_small"
    )
    results['Small (16-32-64)'] = {'losses': losses_small, 'accuracies': accs_small, 'best_acc': best_small}
    
    # ========== 实验5: 有BN + 大channel + Dropout + Adam ==========
    print("\n" + "="*60)
    print("Experiment 5: With BN + Large Channels + Dropout + Adam")
    print("="*60)
    
    model_large = MyCIFAR10NetDifferentChannels(channels=[64, 128, 256]).to(device)
    optimizer_large = Adam(model_large.parameters(), lr=0.001, weight_decay=0)
    losses_large, accs_large, best_large = train_model(
        model_large, train_loader, val_loader, optimizer_large, criterion, 
        device, epochs, "model_large"
    )
    results['Large (64-128-256)'] = {'losses': losses_large, 'accuracies': accs_large, 'best_acc': best_large}
    
    # ========== 实验6: 有BN + 中等channel + Dropout + Adam + LeakyReLU ==========
    print("\n" + "="*60)
    print("Experiment 6: With BN + Medium Channels + Dropout + Adam + LeakyReLU")
    print("="*60)
    
    model_leaky = MyCIFAR10NetLeakyReLU().to(device)
    optimizer_leaky = Adam(model_leaky.parameters(), lr=0.001, weight_decay=0)
    losses_leaky, accs_leaky, best_leaky = train_model(
        model_leaky, train_loader, val_loader, optimizer_leaky, criterion, 
        device, epochs, "model_leaky_relu"
    )
    results['LeakyReLU'] = {'losses': losses_leaky, 'accuracies': accs_leaky, 'best_acc': best_leaky}
    
    # ========== 实验7: 有BN + 中等channel + Dropout + SGD (动量) ==========
    print("\n" + "="*60)
    print("Experiment 7: With BN + Medium Channels + Dropout + SGD (Momentum)")
    print("="*60)
    
    model_sgd = MyCIFAR10Net().to(device)
    optimizer_sgd = SGD(model_sgd.parameters(), lr=0.01, momentum=0.9, weight_decay=0)
    losses_sgd, accs_sgd, best_sgd = train_model(
        model_sgd, train_loader, val_loader, optimizer_sgd, criterion, 
        device, epochs, "model_sgd"
    )
    results['SGD (Momentum)'] = {'losses': losses_sgd, 'accuracies': accs_sgd, 'best_acc': best_sgd}
    
    # 绘制所有结果对比
    plot_results(results, 'all_experiments_results.png')
    
    # 打印总结
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"{'Model':<40} {'Best Val Acc (%)':<20} {'Parameters':<15}")
    print("-" * 75)
    
    params_dict = {
        'With BN (Baseline)': count_parameters(MyCIFAR10Net()),
        'Without BN': count_parameters(MyCIFAR10NetNoBN()),
        'With BN + Weight Decay': count_parameters(MyCIFAR10Net()),
        'Small (16-32-64)': count_parameters(MyCIFAR10NetDifferentChannels(channels=[16,32,64])),
        'Large (64-128-256)': count_parameters(MyCIFAR10NetDifferentChannels(channels=[64,128,256])),
        'LeakyReLU': count_parameters(MyCIFAR10NetLeakyReLU()),
        'SGD (Momentum)': count_parameters(MyCIFAR10Net()),
    }

    for name, res in results.items():
        params = params_dict.get(name, 0)
        print(f"{name:<40} {res['best_acc']:<20.2f} {params:<15,}")
    
    # 打印额外对比分析
    print("\n" + "="*60)
    print("KEY COMPARISONS")
    print("="*60)
    print(f"BN Effect (Baseline vs Without BN): {best_bn:.2f}% vs {best_no_bn:.2f}% → Difference: {best_bn - best_no_bn:+.2f}%")
    print(f"Weight Decay Effect: {best_bn:.2f}% vs {best_bn_wd:.2f}% → Difference: {best_bn_wd - best_bn:+.2f}%")
    print(f"Channel Size Effect (Small vs Baseline vs Large): {best_small:.2f}% vs {best_bn:.2f}% vs {best_large:.2f}%")
    print(f"Activation Effect (ReLU vs LeakyReLU): {best_bn:.2f}% vs {best_leaky:.2f}% → Difference: {best_leaky - best_bn:+.2f}%")
    print(f"Optimizer Effect (Adam vs SGD): {best_bn:.2f}% vs {best_sgd:.2f}% → Difference: {best_sgd - best_bn:+.2f}%")
    
    # 保存结果到文件
    import json
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nAll models saved! Results saved to all_experiments_results.png")


if __name__ == '__main__':
    main()