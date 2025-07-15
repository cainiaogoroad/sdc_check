"""
主程序：演示注意力机制和MLP，并使用ABFT检测SDC
"""

import torch
import numpy as np
import argparse
import sys
import os

from models.attention import Attention
from models.mlp import MLP
from abft.detector import SDCDetector

def demo_attention(dim=32, num_heads=4, batch_size=2, seq_len=4, inject_fault=True):
    """
    演示注意力机制的ABFT检测SDC
    
    Args:
        dim: 输入维度
        num_heads: 注意力头数
        batch_size: 批量大小
        seq_len: 序列长度
        inject_fault: 是否注入故障
    """
    print("\n" + "="*50)
    print("注意力机制(Attention)演示")
    print("="*50)
    
    # 创建一个测试输入
    print(f"创建输入张量: [batch_size={batch_size}, seq_len={seq_len}, dim={dim}]")
    x = torch.randn(batch_size, seq_len, dim)
    
    # 创建注意力模型
    print(f"创建注意力模型: [dim={dim}, num_heads={num_heads}]")
    attn = Attention(dim=dim, num_heads=num_heads)
    
    # 前向传播（启用ABFT）
    print("前向传播（启用ABFT）...")
    output = attn(x, enable_abft=True)
    print(f"输出形状: {output.shape}")
    
    # 验证完整性（此时应该没有错误）
    print("\n检验计算完整性...")
    result = attn.verify_attn_integrity()
    print(f"是否检测到SDC: {result['is_corrupted']}")
    
    if inject_fault:
        # 人为注入错误
        print("\n注入人为错误到注意力权重矩阵...")
        original_attn_weights = attn.last_attn_weights
        corrupted_attn = original_attn_weights.clone()
        
        # 修改第一个批次，第一个头的注意力权重
        print("修改 batch=0, head=0, row=0, col=0 的权重值")
        original_val = corrupted_attn[0, 0, 0, 0].item()
        corrupted_attn[0, 0, 0, 0] += 0.1
        new_val = corrupted_attn[0, 0, 0, 0].item()
        print(f"原始值: {original_val:.6f}, 修改后: {new_val:.6f}")
        
        # 重新验证完整性（此时应该检测到错误）
        print("\n检验修改后的完整性...")
        result = attn.verify_attn_integrity(corrupted_attn)
        print(f"是否检测到SDC: {result['is_corrupted']}")
        if result['is_corrupted']:
            print("损坏详情:")
            print(f"- 行损坏: {result['row_corrupted']}")
            print(f"- 列损坏: {result['col_corrupted']}")
            if result['corrupted_indices']:
                print(f"- 损坏行索引: {result['corrupted_indices']['rows']}")
                print(f"- 损坏列索引: {result['corrupted_indices']['cols']}")

def demo_mlp(in_features=32, hidden_features=64, batch_size=2, inject_fault=True):
    """
    演示MLP的ABFT检测SDC
    
    Args:
        in_features: 输入特征数
        hidden_features: 隐藏层特征数
        batch_size: 批量大小
        inject_fault: 是否注入故障
    """
    print("\n" + "="*50)
    print("多层感知机(MLP)演示")
    print("="*50)
    
    # 创建一个测试输入
    print(f"创建输入张量: [batch_size={batch_size}, in_features={in_features}]")
    x = torch.randn(batch_size, in_features)
    
    # 创建MLP模型
    print(f"创建MLP模型: [in_features={in_features}, hidden_features={hidden_features}, out_features={in_features}]")
    mlp = MLP(in_features=in_features, hidden_features=hidden_features, out_features=in_features)
    
    # 前向传播（启用ABFT）
    print("前向传播（启用ABFT）...")
    output = mlp(x, enable_abft=True)
    print(f"输出形状: {output.shape}")
    
    # 验证完整性（此时应该没有错误）
    print("\n检验计算完整性...")
    result = mlp.verify_integrity(layer='fc1')
    print(f"是否检测到SDC: {result['is_corrupted']}")
    
    if inject_fault:
        # 人为注入错误
        print("\n注入人为错误到第一层输出...")
        original_fc1_output = mlp.last_fc1_output
        corrupted_output = original_fc1_output.clone()
        
        # 修改第一个批次的第一个特征
        print("修改 batch=0, feature=0 的输出值")
        original_val = corrupted_output[0, 0].item()
        corrupted_output[0, 0] += 0.1
        new_val = corrupted_output[0, 0].item()
        print(f"原始值: {original_val:.6f}, 修改后: {new_val:.6f}")
        
        # 重新验证完整性（此时应该检测到错误）
        print("\n检验修改后的完整性...")
        result = mlp.verify_integrity(layer='fc1', current_tensor=corrupted_output)
        print(f"是否检测到SDC: {result['is_corrupted']}")
        if result['is_corrupted']:
            print("损坏详情:")
            print(f"- 行损坏: {result['row_corrupted']}")
            print(f"- 列损坏: {result['col_corrupted']}")
            if result['corrupted_indices']:
                print(f"- 损坏行索引: {result['corrupted_indices']['rows']}")
                print(f"- 损坏列索引: {result['corrupted_indices']['cols']}")

def demo_sdc_detector(tensor_dim=32, inject_fault=True):
    """
    演示使用SDCDetector检测数据损坏
    
    Args:
        tensor_dim: 张量维度
        inject_fault: 是否注入故障
    """
    print("\n" + "="*50)
    print("SDC检测器演示")
    print("="*50)
    
    # 创建随机张量
    print(f"创建随机张量: [dim={tensor_dim}]")
    tensor = torch.randn(tensor_dim, tensor_dim)
    print(f"张量形状: {tensor.shape}")
    
    # 创建检测器
    print("创建SDC检测器...")
    detector = SDCDetector()
    
    # 注册张量
    print("注册张量到检测器...")
    checksums = detector.register_tensor('test_tensor', tensor)
    print(f"计算的校验和: 总和={checksums['sum']:.6f}, 均值={checksums['mean']:.6f}")
    
    # 验证（此时应该没有错误）
    print("\n检验张量完整性...")
    result = detector.verify_tensor('test_tensor')
    print(f"是否检测到SDC: {result['is_corrupted']}")
    
    if inject_fault:
        # 注入错误
        print("\n注入人为错误...")
        corrupted_tensor = detector.inject_fault('test_tensor', magnitude=0.1)
        
        # 找出被修改的位置
        diff = torch.abs(corrupted_tensor - tensor)
        modified_indices = torch.where(diff > 0.01)
        row, col = modified_indices[0].item(), modified_indices[1].item()
        original_val = tensor[row, col].item()
        new_val = corrupted_tensor[row, col].item()
        print(f"修改位置 [{row}, {col}]: 原始值={original_val:.6f}, 修改后={new_val:.6f}")
        
        # 重新验证（此时应该检测到错误）
        print("\n检验修改后的完整性...")
        result = detector.verify_tensor('test_tensor', corrupted_tensor)
        print(f"是否检测到SDC: {result['is_corrupted']}")
        if result['is_corrupted']:
            print("损坏详情:")
            print(f"- 总和损坏: {result['sum_corrupted']}")
            print(f"- 均值损坏: {result['mean_corrupted']}")
            print(f"- 行损坏: {result['row_corrupted']}")
            print(f"- 列损坏: {result['col_corrupted']}")
            if result['corrupted_rows']:
                print(f"- 损坏行索引: {result['corrupted_rows']}")
            if result['corrupted_cols']:
                print(f"- 损坏列索引: {result['corrupted_cols']}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ABFT检测SDC演示程序")
    parser.add_argument('--model', type=str, default='all', choices=['attention', 'mlp', 'detector', 'all'],
                       help='要演示的模型类型')
    parser.add_argument('--dim', type=int, default=32, help='模型维度/特征数')
    parser.add_argument('--no-fault', action='store_true', help='不注入故障')
    args = parser.parse_args()
    
    # 根据选择演示不同的模型
    if args.model == 'attention' or args.model == 'all':
        demo_attention(dim=args.dim, inject_fault=not args.no_fault)
        
    if args.model == 'mlp' or args.model == 'all':
        demo_mlp(in_features=args.dim, inject_fault=not args.no_fault)
        
    if args.model == 'detector' or args.model == 'all':
        demo_sdc_detector(tensor_dim=args.dim, inject_fault=not args.no_fault)

if __name__ == "__main__":
    main() 