"""
训练最佳配置：Large + Adam + lr=0.001 + weight_decay=1e-4
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from data.loaders import get_cifar_loader
from models.my_net import MyCIFAR10NetDifferentChannels
from utils.nn import count_parameters


def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_accuracy(model, loader, device):
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
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    print(f"\nTraining {model_name} on {device}")
    print(f"Parameters: {count_parameters(model):,}")
    
    for epoch in tqdm(range(epochs), desc=model_name):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        val_acc = get_accuracy(model, val_loader, device)
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_best.pth")
        
        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return train_losses, val_accuracies, best_val_acc


def plot_all_results(all_results, save_path='all_experiments_with_best.png'):
    """绘制所有实验（包括最佳配置）的对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 按准确率排序
    sorted_items = sorted(all_results.items(), key=lambda x: x[1]['best_acc'], reverse=True)
    
    # Loss曲线
    for name, results in sorted_items:
        axes[0].plot(results['losses'], linewidth=1.5, 
                    label=f"{name} ({results['best_acc']:.1f}%)")
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    for name, results in sorted_items:
        axes[1].plot(results['accuracies'], linewidth=1.5,
                    label=f"{name} ({results['best_acc']:.1f}%)")
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12)
    axes[1].set_title('Validation Accuracy Comparison', fontsize=14)
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('All Experiments Comparison (Including Best Configuration)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Figure saved to {save_path}")


def main():
    set_random_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    batch_size = 128
    train_loader = get_cifar_loader(train=True, batch_size=batch_size)
    val_loader = get_cifar_loader(train=False, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    epochs = 50
    
    # 加载之前保存的结果
    all_results = {}
    previous_results_file = 'training_results.json'
    if os.path.exists(previous_results_file):
        print(f"\nLoading previous results from {previous_results_file}")
        with open(previous_results_file, 'r') as f:
            previous_results = json.load(f)
        for name, data in previous_results.items():
            all_results[name] = {
                'losses': data['losses'],
                'accuracies': data['accuracies'],
                'best_acc': data['best_acc']
            }
        print(f"Loaded {len(previous_results)} previous experiments")
    
    # ========== 训练最佳配置 ==========
    print("\n" + "="*70)
    print("BEST CONFIG: Large Channels (64-128-256) + Adam + Weight Decay 1e-4")
    print("="*70)
    
    model_best = MyCIFAR10NetDifferentChannels(channels=[64, 128, 256]).to(device)
    optimizer_best = Adam(model_best.parameters(), lr=0.001, weight_decay=1e-4)
    
    losses_best, accs_best, best_acc = train_model(
        model_best, train_loader, val_loader, optimizer_best, criterion, 
        device, epochs, "best_config_large_adam_wd1e4"
    )
    
    all_results['BEST: Large + Adam + WD(1e-4)'] = {
        'losses': losses_best,
        'accuracies': accs_best,
        'best_acc': best_acc
    }
    
    # ========== 绘制所有结果对比 ==========
    print("\n" + "="*70)
    print("Plotting all results")
    print("="*70)
    
    plot_all_results(all_results, 'all_experiments_with_best.png')
    
    # ========== 打印总结 ==========
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL EXPERIMENTS (SORTED BY ACCURACY)")
    print("="*70)
    print(f"{'Model':<45} {'Best Val Acc (%)':<20} {'Parameters':<15}")
    print("-" * 80)
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_acc'], reverse=True)
    
    params_dict = {
        'Large (64-128-256)': '1,423,114',
        'BEST: Large + Adam + WD(1e-4)': '1,423,114',
        'LeakyReLU': '620,810',
        'SGD (Momentum)': '620,810',
        'With BN (Baseline)': '620,810',
        'With BN + Weight Decay': '620,810',
        'Without BN': '620,362',
        'Small (16-32-64)': '288,778'
    }
    
    for name, res in sorted_results:
        params = params_dict.get(name, '620,810')
        print(f"{name:<45} {res['best_acc']:<20.2f} {params:<15}")
    
    # 保存所有结果
    with open('all_results_with_best.json', 'w') as f:
        results_to_save = {}
        for name, res in all_results.items():
            results_to_save[name] = {
                'losses': [float(x) for x in res['losses']],
                'accuracies': [float(x) for x in res['accuracies']],
                'best_acc': float(res['best_acc'])
            }
        json.dump(results_to_save, f, indent=2)
    
    print("\n✅ Done!")
    print(f"   Best accuracy: {best_acc:.2f}%")
    print("   Figure: all_experiments_with_best.png")
    print("   Model: best_config_large_adam_wd1e4_best.pth")


if __name__ == '__main__':
    main()