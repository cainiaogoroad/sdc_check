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
from abft.fault_injection import (
    FaultType, FaultLocation, FaultPattern, FaultTiming,
    FaultInjector, BitFlipInjector, RandomValueInjector, 
    GaussianNoiseInjector, ZeroValueInjector, ConstantInjector,
    ScalingInjector, PermutationInjector
)

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
        # 使用错误注入器注入错误
        print("\n使用错误注入器注入错误...")
        
        # 创建随机值错误注入器
        injector = RandomValueInjector(
            min_val=-0.5, 
            max_val=0.5,
            fault_location=FaultLocation.SPECIFIED,
            fault_pattern=FaultPattern.SINGLE
        )
        
        # 获取注意力权重并注入错误
        original_attn_weights = attn.last_attn_weights
        # 指定错误位置：第一个批次，第一个头的注意力权重
        indices = (0, 0, 0, 0)  # batch=0, head=0, row=0, col=0
        
        print(f"注入位置: {indices}")
        original_val = original_attn_weights[indices].item()
        
        # 注入错误
        corrupted_attn = injector.inject(original_attn_weights.clone(), indices=indices)
        new_val = corrupted_attn[indices].item()
        
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
        # 使用错误注入器注入错误
        print("\n使用错误注入器注入错误...")
        
        # 创建位翻转错误注入器
        injector = BitFlipInjector(
            num_bits=2,  # 翻转2个位
            fault_location=FaultLocation.SPECIFIED,
            fault_pattern=FaultPattern.SINGLE
        )
        
        # 获取第一层输出并注入错误
        original_fc1_output = mlp.last_fc1_output
        # 指定错误位置：第一个批次的第一个特征
        indices = (0, 0)  # batch=0, feature=0
        
        print(f"注入位置: {indices}")
        original_val = original_fc1_output[indices].item()
        
        # 注入错误
        corrupted_output = injector.inject(original_fc1_output.clone(), indices=indices)
        new_val = corrupted_output[indices].item()
        
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
        # 使用错误注入器注入错误
        print("\n使用错误注入器注入错误...")
        
        # 创建高斯噪声错误注入器
        injector = GaussianNoiseInjector(
            mean=0.0,
            std=1.0,
            fault_pattern=FaultPattern.RANDOM_BLOCK
        )
        
        # 注入随机块高斯噪声
        print("注入随机块高斯噪声...")
        original_tensor = tensor.clone()
        corrupted_tensor = injector.inject(original_tensor, block_size=(3, 3))
        
        # 找出被修改的区域
        diff = torch.abs(corrupted_tensor - tensor)
        num_modified = torch.sum(diff > 0.001).item()
        max_diff = torch.max(diff).item()
        
        print(f"修改的元素数量: {num_modified}")
        print(f"最大修改幅度: {max_diff:.6f}")
        
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

def demo_fault_injectors(tensor_dim=10):
    """
    演示各种错误注入器
    
    Args:
        tensor_dim: 张量维度
    """
    print("\n" + "="*50)
    print("错误注入器演示")
    print("="*50)
    
    # 创建一个演示张量
    tensor = torch.ones(tensor_dim, tensor_dim)
    print(f"创建基础张量: [dim={tensor_dim}x{tensor_dim}], 值全为1.0")
    
    # 演示各种错误注入器
    injectors = [
        # 1. 位翻转错误
        {
            "name": "位翻转错误",
            "injector": BitFlipInjector(
                fault_pattern=FaultPattern.SINGLE,
                fault_location=FaultLocation.RANDOM
            )
        },
        # 2. 随机值错误
        {
            "name": "随机值错误",
            "injector": RandomValueInjector(
                min_val=-10.0,
                max_val=10.0,
                fault_pattern=FaultPattern.MULTIPLE,
                fault_location=FaultLocation.RANDOM
            ),
            "params": {"num_faults": 5}
        },
        # 3. 高斯噪声错误
        {
            "name": "高斯噪声错误",
            "injector": GaussianNoiseInjector(
                mean=0.0,
                std=2.0,
                fault_pattern=FaultPattern.ROW
            )
        },
        # 4. 零值错误
        {
            "name": "零值错误",
            "injector": ZeroValueInjector(
                fault_pattern=FaultPattern.COLUMN
            )
        },
        # 5. 常量错误
        {
            "name": "常量错误",
            "injector": ConstantInjector(
                value=999.0,
                fault_pattern=FaultPattern.DIAGONAL
            )
        },
        # 6. 缩放错误
        {
            "name": "缩放错误",
            "injector": ScalingInjector(
                scale_factor=-5.0,
                fault_pattern=FaultPattern.CROSS
            )
        },
        # 7. 置换错误
        {
            "name": "置换错误",
            "injector": PermutationInjector(
                fault_pattern=FaultPattern.RANDOM_BLOCK
            ),
            "params": {"block_size": (3, 3)}
        }
    ]
    
    # 遍历演示各种注入器
    for idx, inj_info in enumerate(injectors):
        print(f"\n{idx+1}. {inj_info['name']}:")
        
        # 复制原始张量
        test_tensor = tensor.clone()
        
        # 注入错误
        params = inj_info.get("params", {})
        corrupted = inj_info["injector"].inject(test_tensor, **params)
        
        # 计算统计信息
        diff = torch.abs(corrupted - tensor)
        num_modified = torch.sum(diff > 0.001).item()
        if num_modified > 0:
            max_diff = torch.max(diff).item()
            min_diff = torch.min(diff[diff > 0.001]).item()
            mean_diff = torch.mean(diff[diff > 0.001]).item()
        else:
            max_diff = min_diff = mean_diff = 0.0
        
        # 输出统计信息
        print(f"  - 修改的元素数量: {num_modified}")
        if num_modified > 0:
            print(f"  - 最小修改幅度: {min_diff:.6f}")
            print(f"  - 最大修改幅度: {max_diff:.6f}")
            print(f"  - 平均修改幅度: {mean_diff:.6f}")
        
        # 如果张量不太大，打印修改后的张量
        if tensor_dim <= 5:
            print(f"  - 修改后的张量:\n{corrupted}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ABFT检测SDC演示程序")
    parser.add_argument('--model', type=str, default='all', 
                       choices=['attention', 'mlp', 'detector', 'injectors', 'all'],
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
    
    if args.model == 'injectors' or args.model == 'all':
        # 对于错误注入器演示，使用较小的维度使输出更清晰
        demo_dim = min(args.dim, 10)
        demo_fault_injectors(tensor_dim=demo_dim)

if __name__ == "__main__":
    main() 