"""
测试注意力机制及其ABFT检测功能
"""

import torch
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.attention import Attention
from abft.detector import SDCDetector

def test_attention_forward():
    """测试注意力机制的前向传播"""
    # 创建一个小的测试输入
    batch_size, seq_len, dim = 2, 4, 32
    x = torch.randn(batch_size, seq_len, dim)
    
    # 创建注意力模型
    attn = Attention(dim=dim, num_heads=4)
    
    # 前向传播
    output = attn(x)
    
    # 检查输出形状
    assert output.shape == x.shape, f"输出形状 {output.shape} 与输入形状 {x.shape} 不匹配"
    
    print("注意力机制前向传播测试通过")
    return True

def test_attention_abft():
    """测试注意力机制的ABFT检测功能"""
    # 创建一个小的测试输入
    batch_size, seq_len, dim = 2, 4, 32
    x = torch.randn(batch_size, seq_len, dim)
    
    # 创建注意力模型
    attn = Attention(dim=dim, num_heads=4)
    
    # 前向传播（启用ABFT）
    output = attn(x, enable_abft=True)
    
    # 保存注意力权重的副本
    original_attn_weights = attn.last_attn_weights.clone()
    
    # 验证完整性（此时应该没有错误）
    result = attn.verify_attn_integrity()
    assert not result['is_corrupted'], "检测到错误的SDC"
    
    # 人为注入错误
    corrupted_attn = original_attn_weights.clone()
    # 修改第一个批次，第一个头的注意力权重
    corrupted_attn[0, 0, 0, 0] += 0.1
    
    # 重新验证完整性（此时应该检测到错误）
    result = attn.verify_attn_integrity(corrupted_attn)
    assert result['is_corrupted'], "未能检测到人为注入的错误"
    
    print("注意力机制ABFT检测测试通过")
    return True

def test_attention_sdc_detector():
    """测试使用SDCDetector检测注意力机制中的数据损坏"""
    # 创建一个小的测试输入
    batch_size, seq_len, dim = 2, 4, 32
    x = torch.randn(batch_size, seq_len, dim)
    
    # 创建注意力模型和检测器
    attn = Attention(dim=dim, num_heads=4)
    detector = SDCDetector()
    
    # 前向传播
    output = attn(x)
    
    # 注册注意力输出到检测器
    detector.register_tensor('attn_output', output)
    
    # 验证（此时应该没有错误）
    result = detector.verify_tensor('attn_output')
    assert not result['is_corrupted'], "检测到错误的SDC"
    
    # 注入错误
    corrupted_output = detector.inject_fault('attn_output', magnitude=0.1)
    
    # 重新验证（此时应该检测到错误）
    result = detector.verify_tensor('attn_output', corrupted_output)
    assert result['is_corrupted'], "未能检测到人为注入的错误"
    
    print("SDCDetector检测注意力机制数据损坏测试通过")
    return True

if __name__ == "__main__":
    # 运行所有测试
    test_attention_forward()
    test_attention_abft()
    test_attention_sdc_detector()
    
    print("所有测试通过!") 