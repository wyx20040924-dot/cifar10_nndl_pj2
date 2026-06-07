"""
Part 2: Compare VGG_A with and without BatchNorm
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
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters
from utils.nn import count_parameters


def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def train_model_full(model, train_loader, val_loader, optimizer, criterion, 
                      device, epochs=100, model_name='model', save_losses=True):
    """训练并记录所有loss"""
    model.to(device)
    train_losses = []
    val_accuracies = []
    step_losses = []  # 每个step的loss，用于loss landscape
    best_val_acc = 0
    
    print(f"\nTraining {model_name}")
    print(f"Parameters: {count_parameters(model):,}")
    
    for epoch in tqdm(range(epochs), desc=model_name):
        model.train()
        epoch_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if save_losses:
                step_losses.append(loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 每5个epoch验证一次
        if (epoch + 1) % 5 == 0:
            val_acc = get_accuracy(model, val_loader, device)
            val_accuracies.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"{model_name}_best.pth")
            
            print(f"\nEpoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    return train_losses, val_accuracies, step_losses, best_val_acc


def plot_comparison(results, save_path='bn_comparison.png'):
    """绘制BN vs 无BN的对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss曲线
    for name, res in results.items():
        axes[0].plot(res['losses'], label=f"{name} (Best: {res['best_acc']:.1f}%)")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss: With BN vs Without BN')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    for name, res in results.items():
        # 对齐准确率记录点（每5个epoch）
        epochs = [5 * i for i in range(len(res['accuracies']))]
        axes[1].plot(epochs, res['accuracies'], marker='o', label=f"{name}")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy (%)')
    axes[1].set_title('Validation Accuracy: With BN vs Without BN')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
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
    lr = 0.001
    
    results = {}
    
    # 训练 VGG_A (without BN)
    print("\n" + "="*50)
    print("Training VGG_A (without BatchNorm)")
    print("="*50)
    
    model_without_bn = VGG_A()
    optimizer_without_bn = Adam(model_without_bn.parameters(), lr=lr)
    
    losses_no_bn, accs_no_bn, steps_no_bn, best_no_bn = train_model_full(
        model_without_bn, train_loader, val_loader, optimizer_without_bn, 
        criterion, device, epochs, "vgg_a_no_bn"
    )
    results['VGG_A (No BN)'] = {
        'losses': losses_no_bn, 
        'accuracies': accs_no_bn, 
        'best_acc': best_no_bn,
        'step_losses': steps_no_bn
    }
    
    # 训练 VGG_A_BatchNorm (with BN)
    print("\n" + "="*50)
    print("Training VGG_A_BatchNorm (with BatchNorm)")
    print("="*50)
    
    model_with_bn = VGG_A_BatchNorm()
    optimizer_with_bn = Adam(model_with_bn.parameters(), lr=lr)
    
    losses_bn, accs_bn, steps_bn, best_bn = train_model_full(
        model_with_bn, train_loader, val_loader, optimizer_with_bn, 
        criterion, device, epochs, "vgg_a_with_bn"
    )
    results['VGG_A (With BN)'] = {
        'losses': losses_bn, 
        'accuracies': accs_bn, 
        'best_acc': best_bn,
        'step_losses': steps_bn
    }
    
    # 绘制对比图
    plot_comparison(results, 'vgg_bn_comparison.png')
    
    # 打印总结
    print("\n" + "="*50)
    print("VGG BN COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Model':<25} {'Best Val Acc (%)':<20} {'Parameters':<15}")
    print("-" * 60)
    print(f"{'VGG_A (No BN)':<25} {best_no_bn:<20.2f} {count_parameters(model_without_bn):<15,}")
    print(f"{'VGG_A (With BN)':<25} {best_bn:<20.2f} {count_parameters(model_with_bn):<15,}")
    
    # 保存step losses用于loss landscape
    np.savetxt('losses_no_bn.txt', steps_no_bn)
    np.savetxt('losses_with_bn.txt', steps_bn)
    print("\nStep losses saved to losses_no_bn.txt and losses_with_bn.txt")


if __name__ == '__main__':
    main()